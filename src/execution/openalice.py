import json
import hashlib
from datetime import datetime, timezone, timedelta
import math
import os
import asyncio
from dataclasses import dataclass
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker, UnsandboxedWorkflowRunner
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.langgraph_debater import execute_debate
from models.torchtrade_drone import execute_torchtrade_drone

# Single Source of Truth for Limits
# Single Source of Truth for Limits - Evolution Phase: Allowing Margin for Alpha
MAX_POSITION_SIZE = 1.0


@dataclass
class OrderParams:
    asset: str
    action: str
    size: float
    rationale_hash: str


def calc_kelly(probability: float, current_vix: float = None) -> float:
    """
    Leveraged Kelly Criterion calculation bounded by rigorous Security Circuit Breakers.
    """
    # 1. Type Safety & Domain Bounds
    if not isinstance(probability, (int, float)) or math.isnan(probability):
        print("[OpenAlice Guard] FATAL: Probability input is invalid/NaN. Forcing 0.0")
        return 0.0

    if probability < 0.0 or probability > 1.0:
        print(
            f"[OpenAlice Guard] FATAL: Probability {probability} out of bounds (0.0-1.0). Forcing 0.0"
        )
        return 0.0

    # 2. Volatility Sanity Bounds
    if current_vix is not None:
        if not isinstance(current_vix, (int, float)) or math.isnan(current_vix):
            print("[OpenAlice Guard] FATAL: VIX input is invalid. Forcing 0.0")
            return 0.0
        if current_vix < 5.0 or current_vix > 100.0:
            print(
                f"[OpenAlice Guard] FATAL: VIX ({current_vix}) outside bounds. Forcing 0.0"
            )
            return 0.0

    edge = probability - 0.5
    if edge <= 0:
        return 0.0

    size = max(0.01, edge * 4.0)

    # 3. Use Shared Configuration Limits
    return min(size, MAX_POSITION_SIZE)


class OpenAliceUTA:
    def __init__(self, account_id="alpha_fund"):
        self.account_id = account_id

        # [SECURITY] Fail-closed validation. Do not just print. RAISE.
        enforced_scope = os.getenv("ALPACA_KEY_ROLE", "UNVERIFIED")
        if enforced_scope != "RESTRICTED_TRADE_ONLY":
            raise PermissionError(
                f"FATAL: Alpaca API Keys lack RESTRICTED_TRADE_ONLY. Found: {enforced_scope}. Halting."
            )

        print(f"[OpenAlice] Connected to Unified Trading Account: {self.account_id}")

    # [ARCHITECTURAL FIX] Make execution a single, stateless, atomic pipeline.
    # Do not use instance variables to store transient trade state.
    def execute_order(
        self, asset: str, action: str, size: float, rationale_hash: str
    ) -> dict:
        """
        Atomically stages, commits, and pushes an order to prevent race conditions.
        """
        if size <= 0:
            print(f"[OpenAlice] Order ignored: Size must be > 0. Received {size}.")
            return None

        # 1. Stage
        order_payload = {
            "asset": asset,
            "action": action,
            "size": round(size, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # 2. Commit (Deterministic Hashing)
        # Using json.dumps with sort_keys=True ensures exact byte-for-byte reproducibility
        payload_bytes = json.dumps(order_payload, sort_keys=True).encode("utf-8")
        commit_id = hashlib.sha256(payload_bytes).hexdigest()

        committed_order = {
            "order": order_payload,
            "rationale_hash": rationale_hash,
            "commit_id": commit_id,
        }

        # 3. Guard & Push
        if committed_order["order"]["size"] > MAX_POSITION_SIZE:
            raise ValueError(
                f"GUARD BLOCK: Order size {committed_order['order']['size']} exceeds MAX {MAX_POSITION_SIZE}"
            )

        print(
            f"[OpenAlice] PUSH EXECUTED: Successfully simulated broker execution for commit {commit_id[:8]}."
        )

        return committed_order


@activity.defn
async def execute_order_activity(params: OrderParams) -> dict:
    """Temporal Activity wrapper mapping to the execution pipeline."""
    alice = OpenAliceUTA()
    return alice.execute_order(
        params.asset, params.action, params.size, params.rationale_hash
    )


@activity.defn
async def evaluate_market_activity() -> dict:
    """Temporal Activity wrapper to ping the LangGraph Syndicate for market direction."""
    # Sever RL Engine, trigger Boruta+TimesFM Debate Pipeline!
    final_state = execute_debate()
    torch_decision = execute_torchtrade_drone()

    raw_decision = final_state.get("final_decision", "NEUTRAL")

    # Parse the deterministic directive and enforce consensus
    langgraph_dir = "NEUTRAL"
    if "LONG" in raw_decision:
        langgraph_dir = "LONG"
    elif "SHORT" in raw_decision:
        langgraph_dir = "SHORT"

    direction = "NEUTRAL"
    if langgraph_dir != "NEUTRAL":
        if langgraph_dir == torch_decision:
            direction = langgraph_dir
        else:
            print(
                f"[OpenAlice Guard] ABORT: LangGraph({langgraph_dir}) and TorchTrade({torch_decision}) conflict. Forcing NEUTRAL."
            )

    return {
        "decision": direction,
        "rationale": {
            "macro_chief": final_state.get("macro_chief_analysis"),
            "risk_manager": final_state.get("risk_manager_analysis"),
            "raw_decision": raw_decision,
            "torchtrade_decision": torch_decision,
        },
    }


@workflow.defn
class OpenAliceTradeWorkflow:
    @workflow.run
    async def run(self) -> dict:
        """Durable workflow ensuring AI evaluation and trade execution complete safely."""
        # 1. Ping the Boruta+TimesFM NLP Syndicate
        market_eval = await workflow.execute_activity(
            evaluate_market_activity,
            start_to_close_timeout=timedelta(
                seconds=60
            ),  # Expanded timeout for API calls
        )

        decision = market_eval["decision"]
        rationale = market_eval["rationale"]

        if decision == "NEUTRAL":
            return {
                "status": "SKIPPED",
                "reason": "RL Brain inferred NEUTRAL market matrix. Holding Cash.",
            }

        # 2. Synthesize Order Params from AI Output
        # (We default to a standard 0.25 portfolio size for the test deployment)
        params = OrderParams(
            asset="SPY",
            action=decision,
            size=0.25,  # Static size fallback
            rationale_hash=rationale.get("raw_decision", "UNKNOWN_HASH")[:20],
        )

        # 3. Execute Trade
        execution_payload = await workflow.execute_activity(
            execute_order_activity,
            params,
            start_to_close_timeout=timedelta(seconds=15),
        )

        return {
            "status": "EXECUTED",
            "ai_evaluation": market_eval,
            "broker_payload": execution_payload,
        }


async def test_temporal_durable_execution():
    os.environ["ALPACA_KEY_ROLE"] = "RESTRICTED_TRADE_ONLY"
    print("\n[Temporal] Connecting to localhost:7233...")
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue="openalice-trade-queue",
        workflows=[OpenAliceTradeWorkflow],
        activities=[execute_order_activity, evaluate_market_activity],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )

    # Run worker asynchronously
    worker_task = asyncio.create_task(worker.run())

    # Dispatch autonomous AI-driven workflow (No manual OrderParams needed!)
    workflow_id = f"ai-trade-exec-{int(datetime.now().timestamp())}"

    print(f"[Temporal] Submitting Workflow ID: {workflow_id}")
    try:
        result = await client.execute_workflow(
            OpenAliceTradeWorkflow.run,
            id=workflow_id,
            task_queue="openalice-trade-queue",
        )
        print("[Temporal] AI Workflow fully executed and closed successfully!")
        print(json.dumps(result, indent=2))
    finally:
        worker_task.cancel()


if __name__ == "__main__":
    asyncio.run(test_temporal_durable_execution())

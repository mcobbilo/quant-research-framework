import json
import logging
from typing import Dict, Any, Callable

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] A2A RPC - %(message)s"
)


class AgentRPCProtocol:
    """
    Implements Agent-to-Agent JSON RPC for the Swarm.
    Prevents LLM drift and prompt injection by forcing specialized agents
    to communicate via strictly typed payloads.
    """

    def __init__(self):
        self.registry: Dict[str, Callable] = {}

    def register_method(self, method_name: str, handler: Callable):
        """Allows an agent to expose a capability to the swarm."""
        self.registry[method_name] = handler
        logging.info(f"Registered A2A RPC capability: '{method_name}'")

    def dispatch(self, rpc_payload: str) -> str:
        """
        Receives an A2A string, parses JSON-RPC 2.0 structure,
        routes to local capability, and responds.
        """
        try:
            req = json.loads(rpc_payload)
        except json.JSONDecodeError:
            return self._build_error(None, -32700, "Parse error")

        if req.get("jsonrpc") != "2.0":
            return self._build_error(req.get("id"), -32600, "Invalid Request")

        method = req.get("method")
        params = req.get("params", {})
        req_id = req.get("id")

        if not method or method not in self.registry:
            return self._build_error(req_id, -32601, "Method not found")

        try:
            handler = self.registry[method]
            if isinstance(params, dict):
                result = handler(**params)
            elif isinstance(params, list):
                result = handler(*params)
            else:
                result = handler(params)

            return json.dumps({"jsonrpc": "2.0", "result": result, "id": req_id})

        except Exception as e:
            logging.error(f"Execution error over RPC: {str(e)}")
            return self._build_error(req_id, -32000, "Server error", str(e))

    def _build_error(self, req_id, code: int, message: str, data: Any = None) -> str:
        err = {"code": code, "message": message}
        if data:
            err["data"] = data
        return json.dumps({"jsonrpc": "2.0", "error": err, "id": req_id})


# --- Usage Example within the Swarm ---
if __name__ == "__main__":
    protocol = AgentRPCProtocol()

    # 1. Backtesting Agent exposes its capabilities
    def run_historics(symbol: str, window: str):
        return {
            "status": "complete",
            "sharpe": 1.45,
            "symbol": symbol,
            "window": window,
        }

    protocol.register_method("backtest_strategy", run_historics)

    # 2. Executive Agent (LLM) generates an A2A JSON request instead of text prompting
    raw_llm_output = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "backtest_strategy",
            "params": {"symbol": "VIX", "window": "10y"},
            "id": 101,
        }
    )

    response = protocol.dispatch(raw_llm_output)
    print(f"RPC Response: {response}")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from celery.result import AsyncResult

# Import the celery app and tasks
from src.api.celery_worker import celery_app, train_rl_loop, execute_backtest
from src.api.mcp_registry import mcp

app = FastAPI(
    title="Quant Swarm Gateway API",
    description="The sole interface between the Swarm Orchestrator and the Quant Mathematical Engine.",
    version="1.0.0",
)

# Under the hood, FastMCP generates a Starlette app that perfectly aligns with FastAPI endpoints.
# By mounting it at "/mcp", Onyx can connect to its SSE transport natively.
app.mount("/mcp", mcp.sse_app)


# Input schemas for the LLM Gateway
class RLTrainingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float
    episodes: int
    symbol: str


class BacktestParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy_name: str
    start_date: str
    end_date: str


class MarketOrderParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    quantity: float
    side: str  # e.g., 'buy', 'sell'


# API Routes
@app.post(
    "/train/rl_loop", summary="Trigger Asynchronous RL Training", tags=["Training"]
)
async def trigger_rl_loop(params: RLTrainingParams):
    """
    Kicks off `zero_claw_rl_loop.py` inside the Celery cluster without hanging the LLM.
    Returns a UUID immediately.
    """
    # Defensive programming: type validation ensures the Swarm cannot hallucinate malicious kwargs
    if params.learning_rate <= 0:
        raise HTTPException(status_code=400, detail="Learning rate must be positive.")

    task = train_rl_loop.delay(
        lr=params.learning_rate, episodes=params.episodes, symbol=params.symbol
    )

    return {"message": "Training job queued successfully.", "task_uuid": task.id}


@app.post(
    "/execute/backtest", summary="Trigger Full Tactical Backtest", tags=["Execution"]
)
async def trigger_backtest(params: BacktestParams):
    task = execute_backtest.delay(
        strategy_name=params.strategy_name,
        start_date=params.start_date,
        end_date=params.end_date,
    )
    return {"message": "Backtest job queued successfully.", "task_uuid": task.id}


@app.post(
    "/execute/market_order", summary="Synchronous Live Market Order", tags=["Execution"]
)
async def market_order(params: MarketOrderParams):
    """
    Unlike training, firing a live market order must be entirely synchronous and low-latency.
    """
    if params.side not in ["buy", "sell"]:
        raise HTTPException(status_code=400, detail="Side must be 'buy' or 'sell'.")

    # In real life, this triggers the Alpaca/IBKR wrapper immediately.
    return {
        "status": "filled",
        "symbol": params.symbol,
        "quantity_filled": params.quantity,
        "execution_price": 405.15,
        "latency_ms": 14.2,
    }


@app.get(
    "/task_status/{task_id}", summary="Check Celery Task Status", tags=["Monitoring"]
)
async def get_task_status(task_id: str):
    """
    Allows the Swarm to periodically poll whether its deployed backtest has finished.
    """
    task_result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": task_result.status,
    }

    if task_result.state == "SUCCESS":
        response["result"] = task_result.result
    elif task_result.state == "FAILURE":
        response["error"] = str(task_result.info)

    return response

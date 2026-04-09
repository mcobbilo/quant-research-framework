# J-EPA Agent Manual (Nexus-Aware)

Welcome, Agent. You are operating within the **Tri-Modal J-EPA** quantitative ecosystem. This repository has been decoupled to allow for autonomous, asynchronous alpha discovery.

## 1. System Architecture

The project is divided into three distinct operational modes:

### **Mode A: The Brain (Attention Core)**

- **File**: `src/models/jepa_attention_engine.py`
- **Logic**: Multi-Scale J-EPA with a 5-regime `RegimeAttentionBias` module and a `PPOHead` for allocation.
- **Critical Path**: Refit cycles happen every 20 trading days in the backtest loop.
- **Uncertainty**: Uses Bayesian MC Dropout for confidence-scaled allocation.

### **Mode B: The Tools (MCP & Skills)**

- **File**: `src/core/mcp_data_server.py` & `src/skills/base_skill.py`
- **Logic**: A standardized Model Context Protocol (MCP) server that exposes market data and modular mathematical "Skills" (VPIN, Vol-Targeting).
- **Navigation**: Use `list_repos` and `query` via the MCP server to discover new data features.

### **Mode C: The Auditor (Agent Delta)**

- **File**: Managed via `curiosity_engine.py`
- **Logic**: A self-healing loop that monitors `last_failure.log`. Before committing a strategy to the backtest, Delta performs a "Stability Audit" to catch lookahead bias and numerical instability.

## 2. Development Commands

- **Backtest**: `./venv/bin/python3 src/experimental/jepa_extractor.py`
- **Skill Audit**: `./venv/bin/python3 src/skills/base_skill.py`
- **Curiosity Loop**: `./venv/bin/python3 curiosity_engine.py`
- **MCP Server**: `./venv/bin/python3 src/core/mcp_data_server.py`

## 3. Code Style

- **Imports**: Prefer `from src.core import ...` for internal modules.
- **Mathematical Implementation**: All indicators must be implemented from first principles using `numpy` and `pandas`. Do not use `talib`.
- **Typing**: Use static types (`float`, `np.ndarray`, `pd.DataFrame`) for all model functions.
- **Error Handling**: All model functions must contain guards for `NaN` and `Inf` to prevent RL gradient explosion.

## 4. Project Constraints

- **Phase 10 Alignment**: All alpha must be regime-aware and use the 20D (19D + VPIN) world model.
- **Audit Policy**: All code changes to the "Brain" or "Head" must be audited by **Agent Delta** if `curiosity_engine.py` is active.
- **Memory**: Always check [MEMORY.md](file:///Users/milocobb/Desktop/Recent%20Swarm%20Papers/quant_framework/MEMORY.md) before implementing a new hypothesis.

## 5. Context Snapshots

- **State**: Tri-Modal (Brain, Tools, Auditor).
- **Alpha Bridge**: 9.21% CAGR vs 12.85% SPY.
- **Goal**: Bridge the 3.64% gap through autonomous alpha discovery.

## 6. Decision Logic (Phase 25)

The current "Alpha Bridge" logic resides in `jepa_extractor.py`:

- **Offense**: `Waterall` capitulation recovery (1.2x leverage).
- **Defense**: `Linear True Risk Parity` ($1/\sigma \times (1-\rho)$) when J-EPA is off-equities.
- **Proxy**: `VPIN_Proxy` is used as a microstructure filter for high-toxicity regimes.

## 3. Operational Guardrails

- **No Halo-Data**: Do not use `talib` or `pandas_ta`. All indicators must be implemented from first principles.
- **Self-Healing**: If a script fails, always check `MEMORY.md` to see if a similar hypothesis was previously rejected.
- **Context Preservation**: Always update `MEMORY.md` with `[S]` (Success) or `[F]` (Failure) after every run.

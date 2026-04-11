# 1. Development Commands

- **Backtest**: `./venv/bin/python3 src/experimental/jepa_extractor.py`
- **Skill Audit**: `./venv/bin/python3 src/skills/base_skill.py`
- **Curiosity Loop**: `./venv/bin/python3 curiosity_engine.py`
- **MCP Server**: `./venv/bin/python3 src/core/mcp_data_server.py`

## 2. Code Style

- **Imports**: Prefer `from src.core import ...` for internal modules.
- **Mathematical Implementation**: All indicators must be implemented from first principles using `numpy` and `pandas`. Do not use `talib`.
- **Typing**: Use static types (`float`, `np.ndarray`, `pd.DataFrame`) for all model functions.
- **Error Handling**: All model functions must contain guards for `NaN` and `Inf` to prevent RL gradient explosion.

## 3. Project Constraints

- **Phase 10 Alignment**: All alpha must be regime-aware and use the 20D (19D + VPIN) world model.
- **Audit Policy**: All code changes to the "Brain" or "Head" must be audited by **Agent Delta** if `curiosity_engine.py` is active.
- **Memory**: Always check [MEMORY.md](file:///Users/milocobb/Desktop/Recent%20Swarm%20Papers/quant_framework/MEMORY.md) before implementing a new hypothesis.

## 4. Context Snapshots

- **State**: Tri-Modal (Brain, Tools, Auditor).
- **Alpha Bridge**: 9.21% CAGR vs 12.85% SPY.
- **Goal**: Bridge the 3.64% gap through autonomous alpha discovery.

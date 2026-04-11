# Unified Market Insights & Structural Grounding

This index consolidates manual research and 'Phase 20' audit findings to ensure and unified structural memory across the autonomous framework.

## 1. High-Alpha Regimes (Deterministic Grounding)

> *Source: Segmented Volatility Research (2000-2024)*

- **Panic Harvesting (VIX > 40)**: SPY exhibits a **98.3% 12-month win rate** with 37.5% mean returns. This is the system's primary 'Green Light' for maximum equity exposure.
- **Twin Calm (MOVE < 60 & VIX < 15)**: Strongest regime for persistent trend-following. Win rate 82.7%.
- **The Dead Zone (VIX 20-30)**: High-churn, low-probability regime. Recommended exposure: **Reduced (0.4x - 0.6x)**.

## 2. Structural Deficits & Failure Modes

> *Source: [[Structural_Deficits_in_Autonomous_Quantitative_Cod.md]]*

- **'Stubbing' Risk**: Autonomous code generation frequently omits core math. Mandate: **Strict Pipeline Scaffolding** (Data -> Signal -> Risk -> Size -> Exec).
- **Numerical Instability**: Mandatory protections against **NaN propagation** and **Division by Zero** in all rolling regressions.
- **Risk Omission**: Hardcoded statistical prerequisites (e.g., ADF stationarity checks) must be structurally integrated into signal validation.

## 3. Autonomous Execution Safety

> *Source: [[autonomous-execution-safety.md]]*

- **Deterministic Bounds**: Shift all textual and state logic from probabilistic prompts to deterministic backend code (Pydantic + Temporal).
- **Temporal Latency (~1,500ms)**: Fundamentally precludes HFT/Microstructure competition. The system is architected for **Daily/Weekly Tactical Allocation**.
- **Extreme Volatility Throttling**: Unresolved gap regarding how brokerage rate-limits during VIX > 40 interact with Temporal timeout daemons.

## 4. Key Cross-Asset Insights

> *Source: Cross-Volatility Matrix Research*

- **Bond Contagion (MOVE > 130)**: Acts as a structural headwind for SPY (Win rate drops to 71.3%). Use as a primary de-risking trigger regardless of VIX levels.
- **Flight-to-Quality**: While VUSTX has a 95.2% win rate during VIX > 40, SPY's absolute return is >5x higher. **Prioritize Panic Harvesting in SPY.**

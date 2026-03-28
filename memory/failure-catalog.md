_[[AutoDream Last Executed: 2026-03-26 18:38:28 UTC]]_

# 🛑 Failure Catalog & Anti-Patterns
> **STATE:** 2026-03-25 22:44:56 UTC | **DIRECTIVE:** Prevent redundant hypothesis execution.

| ID | Vector | Failure State / Root Cause | Enforced Constraint |
|:---|:---|:---|:---|
| **F-001** | Stat Thresholds | `±3.0σ` unreachable on `N<20` (e.g., `N=10` max limit is `2.84σ`). | **REQUIRE** `N ≥ 20` for any `≥ 3.0σ` rolling triggers. |
| **F-002** | Trade Friction | Scalping (e.g., DHO) bleeds via 10bps slippage, severely underperforming macro-swings (616% vs 1088%). | **BAN** high-frequency scalping. **ENFORCE** macro-swinging (buy crash → hold → sell crest). |

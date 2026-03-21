# Agent Failure Catalog
This document strictly indexes critical algorithmic and infrastructural failures to prevent future agents from wasting context or tokens re-testing broken hypotheses.

## [F-001] N=10 Rolling Standard Deviation Mathematical Bound Limit
- **Date:** 2026-03-20
- **Attempt:** Triggering a Buy signal off a `-3.0` Z-Score using a 10-day rolling mean/STD window (`N=10`).
- **Failure Mode:** Backtest yielded 0 trades. The strict mathematical maximum magnitude possible for a Z-Score in an `N=10` sample size is exactly `2.84`. 
- **Resolution:** Never attempt a >= `3.0` Sigma trigger algorithm on any sample window smaller than `N=20`.

## [F-002] High-Frequency Scalping Friction (Strategy F)
- **Date:** 2026-03-20
- **Attempt:** Modeling a 60-day Damped Harmonic Oscillator to actively scalp the +3/-3 secondary recovering ripples immediately following a VIX crash.
- **Failure Mode:** While mathematically sound (yielded 616% vs a static 574% 60-day hold), the active switching massively underperformed the Phase 9 "Single Swing" approach (1,088%). The transaction slippage (10bps) of scaling completely in and out of 2.0x margin leverage 3-4 separate times in 60 days destroyed the total alpha edge.
- **Resolution:** Hedge-fund execution friction universally destroys intermediate high-frequency retail scalping. Always constrain model algorithms to clean, singular Macro-swings (Buy the absolute crash bottom, Hold through the ripples, Sell the absolute recovery crest).

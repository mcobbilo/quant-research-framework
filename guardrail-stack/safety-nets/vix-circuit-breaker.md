# Safety Net: VIX Sanity Bounds & Data Corruption Guard
**Constraint:** Execution is entirely halted (forced size `0.0x`) if the incoming data stream reports a VIX value `<= 5` or `>= 100` or `NaN`.
**Why:** The historical structural bounds of the VIX index rest rigidly between ~9 and ~85. A VIX reading of `3` implies a massive upstream data corruption (like Yahoo Finance emitting null arrays). A VIX of `> 100` implies total structural macro-economic collapse beyond modeled physical algorithms bounds.
**Agent Enforcement:** `math.isnan(current_vix) or current_vix <= 5 or current_vix >= 100` strictly returns a Kelly size of `0.0`. Agents must not deploy backtest strategies that inherently ignore these core macroeconomic circuit breakers.

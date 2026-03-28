_[[AutoDream Last Executed: 2026-03-26 18:38:45 UTC]]_

# ARCHITECTURAL DECISIONS
> **AutoDream Sync:** 2026-03-25 22:45:16 UTC | **Scope:** Structural Pipeline Logic Verification

| ID [Date] / Ph | Architecture & Implementation | Rationale |
| :--- | :--- | :--- |
| **D-001** `03-20`<br>*(Ph 6)* | **Security:** Trap `math.isnan`, halt on `VIX > 100`. Enforce `pip-tools` with crypto-signed dependency hashes. | Mitigate runtime flash-crashes & supply-chain hijacks. |
| **D-002** `03-20`<br>*(Ph 9)* | **Fourier SPY Strategy:** Default 1.0x SPY. Scale to 2.0x Kelly Margin at 20d `Z_Score < -3.5σ`; revert to 1.0x at `+3.0σ`. | Optimize inflation capture (+638%) & Kelly efficiency (+1,088%). |

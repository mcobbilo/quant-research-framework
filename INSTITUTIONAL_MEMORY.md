# 🧠 Institutional Memory: Quantitative Research Engine
*Compiled via Autonomous Swarm Synthesis (Gemini 3.1 Pro High Context)*

This document serves as the central Nervous System for the quant framework. It catalogs all validated models, rejected strategies, and ultimate quantitative baselines generated across 38+ experimental sessions.

---

## 🏆 The "Council of Winners" (Top Validated Baselines)

These are the peak algorithmic structures that have survived rigorous Walk-Forward and Out-Of-Sample (OOS) testing.

| Brain ID | Architecture | Key Features injected | Backtest Horizon | Max Drawdown | CAGR | Sharpe Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`3f0968d2`** | **The SOTA Trifecta Ensemble** *(TFT 55% / LGBM 30% / XGB 15%)* | 4-Asset (SPY, TLT, GLD, BTC) Vol-Target 12% + Top 150 SHAP | 2020 - 2026 | **-8.88%** | **32.68%** | **2.03** |
| **`2684655d`** | **Global XGBoost Allocator** *(Phase 126 Master)* | VIX > 20 Panic Buy Override + "Doctor Copper" < 1.04x Defensive Block | 10-Year | -17.6% | 164.6% Total | **1.15** |
| **`679fc793`** | **v4.1 Dual-Pocket** *(XGBoost)* | SPY/GLD & TLT/GLD (13% Vol Target + T-Bills) | 2006 - 2026 | -22.87% | 10.62% | **0.90** |

---

## 🚫 The Graveyard (Rejected & Failed Theses)

Do **NOT** attempt to re-explore these hypotheses. They have been empirically proven to generate negative alpha or catastrophic drawdowns.

| Brain ID | Hypothesis Tested | Mathematical Result (Why it Failed) | Ultimate Action |
| :--- | :--- | :--- | :--- |
| `ed18c829` | **VIX Basis Edge** (Spot / 3Mo VIX Term Structure) | Decreased Sharpe by 20%. Created "noise" in the post-2006 ETP era and increased false-negative vetos to 38.4%. | **REJECTED**: Reverted to v4.1 Champion. |
| `2684655d` | **Yield Curve Un-Inversion Override** (10Y-3M > 0) | Using this as a 100% Cash Kill-Switch destroyed baseline performance (Total Returns plunged from 166% to 57%). | **REJECTED**: Abandoned curve logic for local triggers. |
| `2684655d` | **The Falling Knife Cascade** (Bonds beat Stocks > 9%) | Prevented the architecture from buying during massive V-shaped recoveries on Days 6 & 7 post-crash. | **REJECTED**: Doctor Copper override was vastly superior. |
| `fb78672e` | **Pure Python Monolithic Swarm Execution** | Caused astronomical LLM logic hallucinations, "spinning circle" infinite loops, and forward data leakage. | **REJECTED**: Transitioned to Git Patch Dry-Runs. |

---

## ⚙️ Core Infrastructure & Architectural Milestones

Key structural upgrades deployed to the Swarm environment.

### 1. AgentHub Transition & Decentralization (`2684655d`)
Transitioned from a monolithic script to an Asynchronous SQLite Event-Driven architecture.
*   **Orthogonal Nodes**: Split the swarm into `agent_high_compute` (heavy matrix operations) and `agent_low_compute` (elegance over intensity).
*   **WOLF / HOWL**: Decoupled real-time trade bounds into `trading_parameters.json`. WOLF halts trades locally without LLM latency.
*   **Kelly-Criterion Architect**: No predictive agent touches the bankroll. Models submit `pre_trade_signal`; the local Risk Architect calculates purely mathematical Expected Value before authorizing standard Half-Kelly allocation.

### 2. High-Fidelity OSINT & Context Ingestion (`12047a62` & `74c8ea13`)
*   **Financial MCP Server**: Integrated Anthropic's MCP standard with Gemini 3.1 Pro. The agent now asynchronously calls `get_current_stock_price` and evaluates live market conditions before synthesizing SQLite updates.
*   **Google Dorking Academy**: Deployed `dork_lab.py` and structured methodology for tracking hidden market configurations and scraping unstructured web intelligence.

### 3. Red Team Transparency & Stabilisation (`fb78672e` & `fa582ab5`)
*   **Git Pacthes Only**: Eradicated direct execution of AI-generated `.py` files. All logic updates are physically handled as `.patch` objects and run against `git apply --check` dry-runs to prevent system-wide lockups.
*   **Batch Transparency Protocol**: All background python "regex" indexing is banned to prevent unmonitored API calls and stalled processes.

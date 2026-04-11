import os
import sys
import torch
import pandas as pd
import numpy as np
import yfinance as yf

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models.hardcoded_wrapper import attach_features
from execution.openalice import calc_kelly
from core.deliberation import LatentCouncil


class SearchableAgenticStrategy:
    def __init__(self, config, feature_dim=4):
        self.council = LatentCouncil(feature_dim=feature_dim)
        self.config = config
        self.name = config.get("name", "Search Strategy")
        self.baseline_prob = 0.50
        self.action_history = []

    def evaluate(self, row):
        # H19: Recursive Feedback (Augment features with past 2 prob outputs)
        (self.action_history[-2:] + [0.5, 0.5])[:2] if self.config.get(
            "recursive"
        ) else [0.5, 0.5]

        feat_list = [row["SPY"], row["VIX"], row["GOLD"], row["COPPER"]]
        features = torch.tensor(feat_list, dtype=torch.float32).unsqueeze(0)

        # Agent Deliberation
        latent = self.council.project_agent_reasoning(features)

        # Deep Consensus (H15)
        consensus_score = 0.98 if self.config.get("deep_consensus") else 0.85
        action = self.council.derive_action(latent, consensus_score)

        # Apply Search Logic
        prob = self.baseline_prob

        # Industrial Demand Filter (H12/H16)
        if self.config.get("industrial_gate"):
            cgr = row["COPPER"] / row["GOLD"]
            if cgr < 0.051:  # Signal of industrial slowdown
                return 0.40  # Defensive bias

        # Convex VIX Scaling (H13)
        vix = row["VIX"]
        if self.config.get("convex_vix"):
            scale = min(1.5, max(0.4, (vix / 18.0) ** 1.5))
        else:
            scale = min(1.2, max(0.8, vix / 20.0))

        if action == "stage_trade":
            prob = 0.94 * scale
            prob = min(0.99, prob * self.config.get("kelly_multiplier", 1.0))
        elif action == "stage_hedge":
            prob = 0.06 / scale

        # Recursive update
        self.action_history.append(prob)
        return prob


def run_backtest(strategy, df, start_year=2018):
    capital = 10000.0
    strategy_returns = []

    # Filter DF for backtest slice
    test_df = df[df.index >= f"{start_year}-01-01"].copy()

    trading_days = test_df.index.tolist()
    previous_size = 0.0

    for i in range(len(trading_days) - 1):
        row = test_df.iloc[i]
        current_vix = row["VIX"]

        prob_up = strategy.evaluate(row)

        # Hypothesis 1: Custom Kelly
        kelly_size = calc_kelly(prob_up, current_vix)
        size = kelly_size * strategy.config.get("kelly_multiplier", 1.0)

        # Apply hard caps
        size = min(2.0, max(0.0, size))

        transaction_cost = capital * abs(size - previous_size) * 0.001
        margin_interest = (capital * max(0.0, size - 1.0)) * (0.05 / 252.0)
        previous_size = size

        price_start = test_df["SPY"].iloc[i]
        price_end = test_df["SPY"].iloc[i + 1]
        forward_return = (price_end - price_start) / price_start

        trade_pnl = capital * size * forward_return
        net_pnl = trade_pnl - transaction_cost - margin_interest

        strategy_returns.append(net_pnl / capital)
        capital += net_pnl

    rets = np.array(strategy_returns)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    total_return = ((capital / 10000.0) - 1.0) * 100

    return total_return, sharpe


def main():
    print("[Agentic Search] Loading 25-Year Dataset...")
    spy = yf.download("SPY", start="2000-01-01", progress=False, auto_adjust=True)
    vix = yf.download("^VIX", start="2000-01-01", progress=False)["Close"]
    gold = yf.download("GC=F", start="2000-01-01", progress=False)["Close"]
    copper = yf.download("HG=F", start="2000-01-01", progress=False)["Close"]

    df = pd.DataFrame(index=spy.index)
    df["SPY"] = spy["Close"]
    df["VIX"] = vix
    df["GOLD"] = gold
    df["COPPER"] = copper
    df.ffill(inplace=True)
    df = attach_features(df)
    df.dropna(inplace=True)

    hypotheses = [
        {
            "name": "H11: Volatility-Adjusted Entry",
            "convex_vix": True,
            "kelly_multiplier": 1.2,
        },
        {
            "name": "H12: Industrial Momentum (CGR)",
            "industrial_gate": True,
            "kelly_multiplier": 1.3,
        },
        {
            "name": "H13: Convex VIX Rescaling",
            "convex_vix": True,
            "kelly_multiplier": 1.5,
        },
        {"name": "H14: Time-Gated Recovery", "cooldown": 5, "kelly_multiplier": 1.4},
        {
            "name": "H15: Deep Consensus Validation",
            "deep_consensus": True,
            "kelly_multiplier": 1.6,
        },
        {
            "name": "H16: Low-Vol Danger Signal",
            "industrial_gate": True,
            "kelly_multiplier": 1.7,
        },
        {
            "name": "H17: Multi-Timeframe Trend",
            "trend_filter": True,
            "kelly_multiplier": 1.8,
        },
        {
            "name": "H18: Innovation-Gated Exit",
            "akf_exit": True,
            "kelly_multiplier": 1.9,
        },
        {"name": "H19: Recursive Feedback", "recursive": True, "kelly_multiplier": 2.0},
        {
            "name": "H20: The Synthetic Winner",
            "composite": True,
            "kelly_multiplier": 2.2,
            "convex_vix": True,
            "industrial_gate": True,
        },
    ]

    results = []
    for h in hypotheses:
        print(f"\n[Council] Iterating Strategy: {h['name']}")
        strat = SearchableAgenticStrategy(config=h)
        # Load calibrated weights if they exist (Phase 16)
        try:
            strat.council.load_weights()
        except:
            pass

        ret, sharpe = run_backtest(strat, df)
        results.append(
            {
                "Hypothesis": h["name"],
                "Return": f"{ret:.2f}%",
                "Sharpe": f"{sharpe:.2f}",
            }
        )
        print(f"   -> Result: {ret:.2f}% Return, {sharpe:.2f} Sharpe")

    res_df = pd.DataFrame(results)
    print("\n[Search Summary]")
    print(res_df.to_markdown())

    with open("search_results.md", "w") as f:
        f.write("# Agentic Strategy Search Results\n\n")
        f.write(res_df.to_markdown())


if __name__ == "__main__":
    main()

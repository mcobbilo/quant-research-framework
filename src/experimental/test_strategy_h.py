import os
import sys
import pandas as pd
import yfinance as yf

# Re-use our framework dependencies without permanently altering them
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hardcoded_wrapper import attach_features
from execution.openalice import calc_kelly


class StrategyH:
    def __init__(self, entry_z=-3.0, exit_z=3.0, baseline_prob=0.75):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.baseline_prob = baseline_prob
        self.in_trade = False

    def evaluate(self, row):
        z = row["Z_Score"]
        vix_h = row["VIX_HIGH"]
        bb_up = row["VIX_BB_UPPER"]
        ppo = row["VIX_PPO_7"]

        if not self.in_trade:
            # The Unified Triple-Confirmation Block
            if z <= self.entry_z and vix_h > bb_up and ppo > 15.00:
                self.in_trade = True
                return 1.0  # Fire 2.0x Kelly Margin
            return self.baseline_prob
        else:
            if z > self.exit_z:
                self.in_trade = False
                return self.baseline_prob
            return 1.0


def run_test():
    print("[Sandbox] Downloading baseline dataset...")
    spy_data = yf.download(
        "SPY", start="2000-01-01", end="2025-01-01", progress=False, auto_adjust=True
    )
    vix_data = yf.download("^VIX", start="2000-01-01", end="2025-01-01", progress=False)
    gold_data = yf.download(
        "GC=F", start="2000-01-01", end="2025-01-01", progress=False
    )
    copper_data = yf.download(
        "HG=F", start="2000-01-01", end="2025-01-01", progress=False
    )

    if isinstance(spy_data.columns, pd.MultiIndex):
        df = pd.DataFrame(index=spy_data.index)
        df["SPY"] = spy_data[("Close", "SPY")]
        df["VIX"] = vix_data[("Close", "^VIX")]
        df["VIX_OPEN"] = vix_data[("Open", "^VIX")]
        df["VIX_HIGH"] = vix_data[("High", "^VIX")]
        df["VIX_LOW"] = vix_data[("Low", "^VIX")]
        df["GOLD"] = gold_data[("Close", "GC=F")]
        df["COPPER"] = copper_data[("Close", "HG=F")]
    else:
        df = pd.DataFrame(index=spy_data.index)
        df["SPY"] = spy_data["Close"]
        df["VIX"] = vix_data["Close"]
        df["VIX_OPEN"] = vix_data["Open"]
        df["VIX_HIGH"] = vix_data["High"]
        df["VIX_LOW"] = vix_data["Low"]
        df["GOLD"] = gold_data["Close"]
        df["COPPER"] = copper_data["Close"]

    df.ffill(inplace=True)
    df.dropna(inplace=True)
    df = attach_features(df)
    df.dropna(inplace=True)

    model = StrategyH()
    capital = 100000.0
    shares = 0.0
    margin_borrowed = 0.0
    previous_size = 0.0

    print("[Sandbox] Simulating unified constraints...")

    for i in range(len(df)):
        row = df.iloc[i]
        price = row["SPY"]
        vix = row["VIX"]

        prob = model.evaluate(row)
        size = calc_kelly(prob, current_vix=vix)

        friction = capital * abs(size - previous_size) * 0.001
        capital -= friction

        if margin_borrowed > 0:
            interest = margin_borrowed * (0.05 / 252)
            capital -= interest

        target_exposure = capital * size
        current_exposure = shares * price

        delta_exposure = target_exposure - current_exposure
        shares += delta_exposure / price

        if size > 1.0:
            margin_borrowed = (size - 1.0) * capital
        else:
            margin_borrowed = 0.0

        previous_size = size

    final_value = (shares * df.iloc[-1]["SPY"]) - margin_borrowed
    total_ret = ((final_value - 100000.0) / 100000.0) * 100

    print("\n================= STRATEGY H SANDBOX YIELD =================")
    print(f"Strategy H (Unified Tri-Confirmation) | Total Return: {total_ret:,.2f}%")
    print("============================================================\n")


if __name__ == "__main__":
    run_test()

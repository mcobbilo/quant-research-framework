import yfinance as yf
import pandas as pd
import numpy as np


def calculate_vix_tranche_returns():
    print("[Research] Downloading 25 years of SPY and VIX data...")
    spy_data = yf.download(
        "SPY", start="2000-01-01", end="2025-01-01", progress=False, auto_adjust=True
    )
    vix_data = yf.download("^VIX", start="2000-01-01", end="2025-01-01", progress=False)

    df = pd.DataFrame(index=spy_data.index)
    df["SPY"] = spy_data["Close"]
    df["VIX"] = vix_data["Close"]
    df = df.ffill().dropna()

    # Calculate 12-month forward return (252 bars)
    # We use -252 to look forward into the future
    df["Forward_12M_Return"] = df["SPY"].shift(-252) / df["SPY"] - 1

    # Drop the last 252 days because we don't have the "future" data yet
    df = df.dropna(subset=["Forward_12M_Return"])

    # Define Tranches
    def get_tranche(vix):
        if vix > 40:
            return "1: VIX > 40 (Panic)"
        elif 30 < vix <= 40:
            return "2: VIX 30-40 (Capitulation)"
        elif 20 < vix <= 30:
            return "3: VIX 20-30 (Elevation)"
        elif 15 < vix <= 20:
            return "4: VIX 15-20 (Normal)"
        else:
            return "5: VIX < 15 (Complacency)"

    df["Tranche"] = df["VIX"].apply(get_tranche)

    # Statistics Synthesis
    summary = (
        df.groupby("Tranche")["Forward_12M_Return"]
        .agg(
            [
                ("Count", "count"),
                ("Mean Return (%)", lambda x: np.mean(x) * 100),
                ("Median Return (%)", lambda x: np.median(x) * 100),
                ("Win Rate (%)", lambda x: (x > 0).mean() * 100),
                ("Std Dev (%)", lambda x: np.std(x) * 100),
            ]
        )
        .sort_index()
    )

    print("\n" + "=" * 80)
    print("VIX TRANCHE ANALYSIS: 12-MONTH FORWARD SPY RETURNS (2000-2024)")
    print("=" * 80)
    print(summary.to_string())
    print("=" * 80)

    # Recommendation
    top_tranche = summary["Mean Return (%)"].idxmax()
    print(f"\n[RESEARCH FINDING] The highest average return comes from: {top_tranche}")
    print(f"Mean: {summary.loc[top_tranche, 'Mean Return (%)']:.2f}%")


if __name__ == "__main__":
    calculate_vix_tranche_returns()

import numpy as np
import pandas as pd


def generate_regime_data(n_samples=1000):
    """
    Generates synthetic market data with distinct regime shifts.
    Regime 0: Low-Vol Bull Trend
    Regime 1: High-Vol Mean Reversion
    Regime 2: Flash Crash & Recovery
    """
    np.random.seed(42)
    np.arange(n_samples)

    # Regime boundaries
    r0_end = 400
    r1_end = 800

    # Prices
    prices = np.zeros(n_samples)
    regimes = np.zeros(n_samples)

    # Regime 0: Steady Bull (0 -> 400)
    prices[0] = 100
    for i in range(1, r0_end):
        prices[i] = prices[i - 1] * (1 + np.random.normal(0.0005, 0.005))
        regimes[i] = 0

    # Regime 1: High-Vol Mean Reversion (400 -> 800)
    for i in range(r0_end, r1_end):
        target = 120
        prices[i] = (
            prices[i - 1] + 0.1 * (target - prices[i - 1]) + np.random.normal(0, 2.0)
        )
        regimes[i] = 1

    # Regime 2: Crash & Recovery (800 -> 1000)
    for i in range(r1_end, n_samples):
        if i < 850:  # The Crash
            prices[i] = prices[i - 1] * (1 - 0.02 - np.random.normal(0, 0.01))
        else:  # The Recovery
            prices[i] = prices[i - 1] * (1 + 0.01 + np.random.normal(0, 0.008))
        regimes[i] = 2

    df = pd.DataFrame(
        {
            "price": prices,
            "regime": regimes,
            "returns": pd.Series(prices).pct_change().fillna(0),
        }
    )

    # Feature for council (4 dim as expected by flow.py)
    df["vol_5"] = df["returns"].rolling(5).std().fillna(0)
    df["vol_20"] = df["returns"].rolling(20).std().fillna(0)
    df["mom_10"] = df["returns"].rolling(10).mean().fillna(0)
    df["drift"] = (df["price"] / df["price"].shift(20) - 1).fillna(0)

    return df


if __name__ == "__main__":
    df = generate_regime_data()
    df.to_csv("src/data/synthetic_market_regimes.csv", index=False)
    print(f"Generated {len(df)} samples of regime-shift data.")

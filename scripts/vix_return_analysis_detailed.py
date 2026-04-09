import yfinance as yf
import pandas as pd
import numpy as np

def calculate_detailed_vix_returns():
    print("[Research] Downloading 25 years of SPY and VIX data...")
    spy_data = yf.download("SPY", start="2000-01-01", end="2025-01-01", progress=False, auto_adjust=True)
    vix_data = yf.download("^VIX", start="2000-01-01", end="2025-01-01", progress=False)

    df = pd.DataFrame(index=spy_data.index)
    df["SPY"] = spy_data["Close"]
    df["VIX"] = vix_data["Close"]
    df = df.ffill().dropna()

    # Time Horizons
    # 6 Months (126 trading days)
    # 12 Months (252 trading days)
    df["Forward_6M_Return"] = df["SPY"].shift(-126) / df["SPY"] - 1
    df["Forward_12M_Return"] = df["SPY"].shift(-252) / df["SPY"] - 1

    # Tranches
    tranches = [
        ("VIX > 40", lambda x: x > 40),
        ("35 < VIX <= 40", lambda x: (x > 35) & (x <= 40)),
        ("30 < VIX <= 35", lambda x: (x > 30) & (x <= 35)),
        ("25 < VIX <= 30", lambda x: (x > 25) & (x <= 30)),
        ("20 < VIX <= 25", lambda x: (x > 20) & (x <= 25)),
        ("15 < VIX <= 20", lambda x: (x > 15) & (x <= 20)),
        ("VIX < 15", lambda x: x < 15),
        ("VIX < 12", lambda x: x < 12) # Specifically requested overlap
    ]

    results = []

    for name, condition in tranches:
        subset = df[condition(df["VIX"])]
        
        # 6-Month Stats
        s6 = subset["Forward_6M_Return"].dropna()
        # 12-Month Stats
        s12 = subset["Forward_12M_Return"].dropna()

        results.append({
            "Tranche": name,
            "Count": len(subset),
            "6M Mean (%)": np.mean(s6) * 100 if not s6.empty else 0,
            "6M Win Rate (%)": (s6 > 0).mean() * 100 if not s6.empty else 0,
            "12M Mean (%)": np.mean(s12) * 100 if not s12.empty else 0,
            "12M Win Rate (%)": (s12 > 0).mean() * 100 if not s12.empty else 0
        })

    summary = pd.DataFrame(results)

    print("\n" + "="*95)
    print("DETAILED VIX TRANCHE ANALYSIS: FORWARD SPY RETURNS (2000-2024)")
    print("="*95)
    print(summary.to_string(index=False))
    print("="*95)

if __name__ == "__main__":
    calculate_detailed_vix_returns()

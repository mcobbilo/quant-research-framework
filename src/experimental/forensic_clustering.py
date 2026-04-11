import os
import sys
import pandas as pd
import yfinance as yf
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hardcoded_wrapper import attach_features


def run_clustering():
    print("[Forensics] Indexing 25-Year Data History...")
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

    # Analyze the clusters of the deepest crashes (<= -3.0 Sigma)
    # Using -3.0 instead of -3.5 natively allows us to actually cluster
    # multiple macro events together for cross-referencing standard patterns.
    triggers = df[df["Z_Score"] <= -3.0].index

    print(
        f"\n[Forensics] Mathematically mapping {len(triggers)} isolated >3-Sigma Panics."
    )

    # Filter nearby duplicate triggers (within 10 days of each other) to isolate distinct crashes
    distinct_panics = []
    for t in triggers:
        if not distinct_panics or (t - distinct_panics[-1]) > timedelta(days=10):
            distinct_panics.append(t)

    print(
        f"[Forensics] De-duplicated into {len(distinct_panics)} distinctly isolated Macro events.\n"
    )

    for i, t_date in enumerate(distinct_panics):
        idx = df.index.get_loc(t_date)

        # Ensure we have boundary buffer
        if idx < 3 or idx > len(df) - 4:
            continue

        print(
            "=========================================================================="
        )
        print(
            f"                      CRASH EVENT {i + 1}: {t_date.strftime('%Y-%m-%d')}"
        )
        print(
            "=========================================================================="
        )

        window = df.iloc[idx - 2 : idx + 3]  # T-2 to T+2

        for r_idx, row in window.iterrows():
            day_offset = window.index.get_loc(r_idx) - idx
            day_str = (
                f"Day T{day_offset:+} " if day_offset != 0 else ">> T+0 (TRIGGER) <<"
            )

            ppo = row["VIX_PPO_7"]
            row["VIX"]
            row["VIX_OPEN"]
            vix_h = row["VIX_HIGH"]
            bb_up = row["VIX_BB_UPPER"]
            z = row["Z_Score"]

            # Identify candlestick overlaps with the Upper Bollinger Band
            bb_breach = "YES" if vix_h > bb_up else "NO"

            # Print structural diagnostics
            print(
                f"{day_str:>19} | Sigma: {z:>5.2f} | VIX High: {vix_h:>5.2f} (Breached BB_Upper? {bb_breach:<3}) | PPO: {ppo:>6.2f}%"
            )
        print("")


if __name__ == "__main__":
    run_clustering()

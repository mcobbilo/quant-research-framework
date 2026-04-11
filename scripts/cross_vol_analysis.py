import yfinance as yf
import pandas as pd
import numpy as np


def calculate_cross_vol_returns():
    print("[Research] Downloading 22+ years of SPY, VUSTX, VIX, and MOVE data...")
    tickers = ["SPY", "VUSTX", "^VIX", "^MOVE"]
    data = yf.download(
        tickers, start="2000-01-01", end="2025-01-01", progress=False, auto_adjust=True
    )

    # We use Close and ffill/dropna to align all 4 series
    df = data["Close"].ffill().dropna()

    print(
        f"[Research] Analyzing synchronized period: {df.index[0].date()} to {df.index[-1].date()}"
    )

    # Forward Returns
    # 3 Months (63d), 6 Months (126d), 12 Months (252d)
    for asset in ["SPY", "VUSTX"]:
        df[f"{asset}_3M_Ret"] = df[asset].shift(-63) / df[asset] - 1
        df[f"{asset}_6M_Ret"] = df[asset].shift(-126) / df[asset] - 1
        df[f"{asset}_12M_Ret"] = df[asset].shift(-252) / df[asset] - 1

    # Tranches for VIX
    vix_tranches = [
        ("VIX > 40", lambda x: x > 40),
        ("35 < VIX <= 40", lambda x: (x > 35) & (x <= 40)),
        ("30 < VIX <= 35", lambda x: (x > 30) & (x <= 35)),
        ("25 < VIX <= 30", lambda x: (x > 25) & (x <= 30)),
        ("20 < VIX <= 25", lambda x: (x > 20) & (x <= 25)),
        ("15 < VIX <= 20", lambda x: (x > 15) & (x <= 20)),
        ("VIX < 15", lambda x: x < 15),
        ("VIX < 12", lambda x: x < 12),
    ]

    # Tranches for MOVE
    move_tranches = [
        ("MOVE > 130", lambda x: x > 130),
        ("110 < MOVE <= 130", lambda x: (x > 110) & (x <= 130)),
        ("90 < MOVE <= 110", lambda x: (x > 90) & (x <= 110)),
        ("75 < MOVE <= 90", lambda x: (x > 75) & (x <= 90)),
        ("60 < MOVE <= 75", lambda x: (x > 60) & (x <= 75)),
        ("MOVE <= 60", lambda x: x <= 60),
    ]

    def summarize(df_source, indicator, tranches, target_asset):
        results = []
        for name, condition in tranches:
            subset = df_source[condition(df_source[indicator])]
            s3 = subset[f"{target_asset}_3M_Ret"].dropna()
            s6 = subset[f"{target_asset}_6M_Ret"].dropna()
            s12 = subset[f"{target_asset}_12M_Ret"].dropna()

            results.append(
                {
                    "Tranche": name,
                    "Count": len(subset),
                    "3M Mean (%)": np.mean(s3) * 100 if not s3.empty else 0,
                    "3M Win Rate (%)": (s3 > 0).mean() * 100 if not s3.empty else 0,
                    "6M Mean (%)": np.mean(s6) * 100 if not s6.empty else 0,
                    "6M Win Rate (%)": (s6 > 0).mean() * 100 if not s6.empty else 0,
                    "12M Mean (%)": np.mean(s12) * 100 if not s12.empty else 0,
                    "12M Win Rate (%)": (s12 > 0).mean() * 100 if not s12.empty else 0,
                }
            )
        return pd.DataFrame(results)

    # Study A: VIX vs VUSTX (Bonds)
    study_a = summarize(df, "^VIX", vix_tranches, "VUSTX")

    # Study B: MOVE vs SPY (Stocks)
    study_b = summarize(df, "^MOVE", move_tranches, "SPY")

    print("\n" + "=" * 115)
    print("STUDY A: VIX (STOCK VOL) vs. VUSTX (LONG BOND PORTFOLIO) FORWARD RETURNS")
    print("=" * 115)
    print(study_a.to_string(index=False))

    print("\n" + "=" * 115)
    print("STUDY B: MOVE (BOND VOL) vs. SPY (S&P 500 PORTFOLIO) FORWARD RETURNS")
    print("=" * 115)
    print(study_b.to_string(index=False))
    print("=" * 115)


if __name__ == "__main__":
    calculate_cross_vol_returns()

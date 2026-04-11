import yfinance as yf
import pandas as pd
import numpy as np


def calculate_move_vustx_returns():
    print("[Research] Downloading 25 years of VUSTX and MOVE Index data...")
    vustx_data = yf.download(
        "VUSTX", start="2000-01-01", end="2025-01-01", progress=False, auto_adjust=True
    )
    move_data = yf.download(
        "^MOVE", start="2000-01-01", end="2025-01-01", progress=False
    )

    df = pd.DataFrame(index=vustx_data.index)
    df["VUSTX"] = vustx_data["Close"]
    df["MOVE"] = move_data["Close"]
    df = df.ffill().dropna()

    # Time Horizons
    # 3 Months (63 trading days)
    # 6 Months (126 trading days)
    # 12 Months (252 trading days)
    df["Forward_3M_Return"] = df["VUSTX"].shift(-63) / df["VUSTX"] - 1
    df["Forward_6M_Return"] = df["VUSTX"].shift(-126) / df["VUSTX"] - 1
    df["Forward_12M_Return"] = df["VUSTX"].shift(-252) / df["VUSTX"] - 1

    # MOVE Tranches
    tranches = [
        ("MOVE > 130", lambda x: x > 130),
        ("110 < MOVE <= 130", lambda x: (x > 110) & (x <= 130)),
        ("90 < MOVE <= 110", lambda x: (x > 90) & (x <= 110)),
        ("75 < MOVE <= 90", lambda x: (x > 75) & (x <= 90)),
        ("60 < MOVE <= 75", lambda x: (x > 60) & (x <= 75)),
        ("MOVE <= 60", lambda x: x <= 60),
    ]

    results = []

    for name, condition in tranches:
        subset = df[condition(df["MOVE"])]

        # Horizons Stats
        s3 = subset["Forward_3M_Return"].dropna()
        s6 = subset["Forward_6M_Return"].dropna()
        s12 = subset["Forward_12M_Return"].dropna()

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

    summary = pd.DataFrame(results)

    print("\n" + "=" * 115)
    print("MOVE TRANCHE ANALYSIS: FORWARD VUSTX RETURNS (2000-2024)")
    print("=" * 115)
    print(summary.to_string(index=False))
    print("=" * 115)


if __name__ == "__main__":
    calculate_move_vustx_returns()

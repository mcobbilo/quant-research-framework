import pandas as pd
import sqlite3
import os
import warnings

warnings.filterwarnings("ignore")


def tsi_analysis():
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "market_data.db"
    )
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM core_market_table", conn, index_col="Date")
    df.index = pd.to_datetime(df.index)

    # Ensure NYA200R is mapped dynamically
    df = df.dropna(subset=["NYA200R"])

    # Shift prices backwards to calculate forward future returns from any given day T
    df["Fwd_5D"] = (df["SPY_CLOSE"].shift(-5) / df["SPY_CLOSE"]) - 1
    df["Fwd_10D"] = (df["SPY_CLOSE"].shift(-10) / df["SPY_CLOSE"]) - 1
    df["Fwd_20D"] = (df["SPY_CLOSE"].shift(-20) / df["SPY_CLOSE"]) - 1
    df["Fwd_60D"] = (df["SPY_CLOSE"].shift(-60) / df["SPY_CLOSE"]) - 1

    top_50 = df.nlargest(50, "NYA200R")
    bottom_50 = df.nsmallest(50, "NYA200R")

    print("\n=======================================================")
    print("  NYSE % ABOVE 200-DAY SMA (NYA200R) EXTREMES ANALYSIS")
    print("=======================================================")
    print("Top 50 HIGHEST NYA200R Readings (Euphoric Overbought):")
    print(f"Average % Above 200D SMA:  {top_50['NYA200R'].mean():.2f}%")
    print(
        f"-> Forward  5-Day Return Average: {top_50['Fwd_5D'].mean() * 100:>6.2f}%  (Win Rate: {(top_50['Fwd_5D'] > 0).mean() * 100:.1f}%)"
    )
    print(
        f"-> Forward 10-Day Return Average: {top_50['Fwd_10D'].mean() * 100:>6.2f}%  (Win Rate: {(top_50['Fwd_10D'] > 0).mean() * 100:.1f}%)"
    )
    print(
        f"-> Forward 20-Day Return Average: {top_50['Fwd_20D'].mean() * 100:>6.2f}%  (Win Rate: {(top_50['Fwd_20D'] > 0).mean() * 100:.1f}%)"
    )
    print(
        f"-> Forward 60-Day Return Average: {top_50['Fwd_60D'].mean() * 100:>6.2f}%  (Win Rate: {(top_50['Fwd_60D'] > 0).mean() * 100:.1f}%)"
    )

    print("\n-------------------------------------------------------")
    print("Bottom 50 LOWEST NYA200R Readings (Structural Capitulation):")
    print(f"Average % Above 200D SMA:  {bottom_50['NYA200R'].mean():.2f}%")
    print(
        f"-> Forward  5-Day Return Average: {bottom_50['Fwd_5D'].mean() * 100:>6.2f}%  (Win Rate: {(bottom_50['Fwd_5D'] > 0).mean() * 100:.1f}%)"
    )
    print(
        f"-> Forward 10-Day Return Average: {bottom_50['Fwd_10D'].mean() * 100:>6.2f}%  (Win Rate: {(bottom_50['Fwd_10D'] > 0).mean() * 100:.1f}%)"
    )
    print(
        f"-> Forward 20-Day Return Average: {bottom_50['Fwd_20D'].mean() * 100:>6.2f}%  (Win Rate: {(bottom_50['Fwd_20D'] > 0).mean() * 100:.1f}%)"
    )
    print(
        f"-> Forward 60-Day Return Average: {bottom_50['Fwd_60D'].mean() * 100:>6.2f}%  (Win Rate: {(bottom_50['Fwd_60D'] > 0).mean() * 100:.1f}%)"
    )
    print("=======================================================\n")


if __name__ == "__main__":
    tsi_analysis()

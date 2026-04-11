import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime
import os
import warnings

warnings.filterwarnings("ignore")


def generate_liquidity_chart():
    # Calculate dates
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=5 * 365)

    print(
        f"Fetching Data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )

    # Fetch SPY
    print("Fetching SPY...")
    spy = yf.download(
        "SPY", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
    )

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)

    # Fetch FRED
    print("Fetching FRED...")
    fred_vars = ["WALCL", "WTREGEN", "RRPONTSYD"]
    fred_df = web.DataReader(fred_vars, "fred", start_date, end_date)
    fred_df["RRPONTSYD"] = fred_df["RRPONTSYD"].fillna(
        0
    )  # Reverse repo was often 0 historically
    fred_df = fred_df.ffill().bfill()

    # Values are in Millions of Dollars. Divide by 1,000,000 for Trillions.
    fred_df["Net_Liquidity_Trillions"] = (
        fred_df["WALCL"] - fred_df["WTREGEN"] - fred_df["RRPONTSYD"]
    ) / 1000000

    # Merge into a single daily trading calendar
    df = pd.merge(
        spy,
        fred_df[["Net_Liquidity_Trillions"]],
        left_index=True,
        right_index=True,
        how="outer",
    )
    df = (
        df.ffill().bfill()
    )  # Forward fill liquidity on weekends/holidays, backfill the start
    df = df.dropna(subset=["Close"])  # Restrict back to SPY trading days

    print("Generating Chart...")
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1.5]}
    )

    # Top panel: Liquidity
    ax1.plot(
        df.index,
        df["Net_Liquidity_Trillions"],
        color="#00d2ff",
        linewidth=2,
        label="Net Dollar Liquidity (Trillions)",
    )
    ax1.set_title(
        "Global Net Dollar Liquidity vs S&P 500 (5-Year Window)",
        fontsize=16,
        color="white",
        pad=20,
    )
    ax1.set_ylabel("Net Liquidity ($ Trillions)", fontsize=12, color="white")
    ax1.grid(color="#2A3459", linestyle="--", alpha=0.7)
    ax1.legend(loc="upper left")

    # Bottom panel: SPY Candlestick
    up = df[df["Close"] >= df["Open"]]
    down = df[df["Close"] < df["Open"]]

    width = 1.0
    width2 = 0.2

    # Plot up candles
    ax2.bar(
        up.index, up["Close"] - up["Open"], width, bottom=up["Open"], color="#26a69a"
    )
    ax2.bar(
        up.index, up["High"] - up["Close"], width2, bottom=up["Close"], color="#26a69a"
    )
    ax2.bar(
        up.index, up["Low"] - up["Open"], width2, bottom=up["Open"], color="#26a69a"
    )

    # Plot down candles
    ax2.bar(
        down.index,
        down["Close"] - down["Open"],
        width,
        bottom=down["Open"],
        color="#ef5350",
    )
    ax2.bar(
        down.index,
        down["High"] - down["Open"],
        width2,
        bottom=down["Open"],
        color="#ef5350",
    )
    ax2.bar(
        down.index,
        down["Low"] - down["Close"],
        width2,
        bottom=down["Close"],
        color="#ef5350",
    )

    ax2.set_ylabel("SPY Price", fontsize=12, color="white")
    ax2.grid(color="#2A3459", linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save artifact
    out_path = "/Users/milocobb/.gemini/antigravity/brain/c33063f0-f712-49cd-b420-4b0183d4e862/artifacts/liquidity_spy_5yr.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0D1117")
    print(f"Chart saved perfectly to {out_path}")


if __name__ == "__main__":
    generate_liquidity_chart()

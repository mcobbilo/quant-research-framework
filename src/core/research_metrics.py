import pandas as pd
import yfinance as yf
import numpy as np
import traceback


def main():
    try:
        print("Loading cv_df...")
        df = pd.read_csv("xLSTM_full_backtest_results.csv")
        df["ds"] = pd.to_datetime(df["ds"])
        df["cutoff"] = pd.to_datetime(df["cutoff"])

        df_1d = df.sort_values(["cutoff", "ds"]).groupby("cutoff").first().reset_index()

        print("Fetching SPY to get actual cutoff prices...")
        spy = yf.Ticker("SPY").history(period="25y").reset_index()
        spy["Date"] = pd.to_datetime(spy["Date"]).dt.tz_localize(None)
        spy.rename(columns={"Date": "cutoff", "Close": "cutoff_price"}, inplace=True)

        df_1d = pd.merge(
            df_1d, spy[["cutoff", "cutoff_price"]], on="cutoff", how="inner"
        )

        df_1d["signal"] = np.where(df_1d["xLSTM-median"] > df_1d["cutoff_price"], 1, -1)
        df_1d["asset_return"] = (df_1d["y"] - df_1d["cutoff_price"]) / df_1d[
            "cutoff_price"
        ]
        df_1d["strategy_return"] = df_1d["signal"] * df_1d["asset_return"]

        df_1d["cum_asset"] = (1 + df_1d["asset_return"]).cumprod()
        df_1d["cum_strat"] = (1 + df_1d["strategy_return"]).cumprod()

        print(f"Total Asset Return: {df_1d['cum_asset'].iloc[-1]:.2f}x")
        print(f"Total Strat Return: {df_1d['cum_strat'].iloc[-1]:.2f}x")

        def get_mdd(returns):
            cum = (1 + returns).cumprod()
            peak = cum.expanding(min_periods=1).max()
            dd = (cum / peak) - 1
            return dd.min()

        print(f"Asset MDD: {get_mdd(df_1d['asset_return']):.2%}")
        print(f"Strat MDD: {get_mdd(df_1d['strategy_return']):.2%}")

        df_1d["year"] = df_1d["ds"].dt.year
        yearly = df_1d.groupby("year").apply(
            lambda x: pd.Series(
                {
                    "Asset": (1 + x["asset_return"]).prod() - 1,
                    "Model": (1 + x["strategy_return"]).prod() - 1,
                }
            )
        )
        print("YEARLY BREAKDOWN:")
        print((yearly * 100).round(2).to_string())
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()

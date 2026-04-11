import pandas as pd
import numpy as np

print("Loading Multi-Variate CV predictions...")
cv_df = pd.read_csv("xLSTM_backtest_results.csv")
cv_df["ds"] = pd.to_datetime(cv_df["ds"])
cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])

df_pred = cv_df.groupby(["unique_id", "cutoff"]).last().reset_index()
df_pred_pivot = df_pred.pivot(
    index="cutoff", columns="unique_id", values="xLSTM"
).reset_index()
df_pred_pivot = df_pred_pivot.rename(
    columns={
        "cutoff": "signal_date",
        "SPY": "prob_up_spy",
        "VUSTX": "prob_up_vustx",
        "GLD": "prob_up_gld",
        "VIX": "prob_up_vix",
    }
)

import sqlite3
import os

db_path = os.path.join("src", "data", "market_data.db")
conn = sqlite3.connect(db_path)
df_core = pd.read_sql_query(
    "SELECT Date as ds, SPY_CLOSE as y, VUSTX_CLOSE as vustx_close, GLD_CLOSE as gld_close FROM core_market_table",
    conn,
)
conn.close()

df_core["ds"] = pd.to_datetime(df_core["ds"]).dt.tz_localize(None)
df_core["spy_close"] = df_core["y"]
df_returns = df_core[["ds", "vustx_close", "gld_close"]].copy()
df_returns["vustx_ret"] = df_returns["vustx_close"].pct_change()
df_returns["gld_ret"] = df_returns["gld_close"].pct_change()
df_returns["vustx_close_prev"] = df_returns["vustx_close"].shift(1)
df_returns["gld_close_prev"] = df_returns["gld_close"].shift(1)
df_returns["vustx_sma200_prev"] = df_returns["vustx_close_prev"].rolling(200).mean()
df_returns["gld_sma200_prev"] = df_returns["gld_close_prev"].rolling(200).mean()

df_cont = df_core[["ds", "spy_close"]].copy().dropna()
df_cont["y_prev"] = df_cont["spy_close"].shift(1)
df_cont = df_cont.dropna(subset=["y_prev"]).copy()

df_cont = pd.merge(df_cont, df_returns, on="ds", how="left")
df_cont["asset_return"] = (df_cont["spy_close"] - df_cont["y_prev"]) / df_cont["y_prev"]

df_cont = pd.merge(
    df_cont, df_pred_pivot, left_on="ds", right_on="signal_date", how="left"
)
df_cont[["prob_up_spy", "prob_up_vustx", "prob_up_gld", "prob_up_vix"]] = df_cont[
    ["prob_up_spy", "prob_up_vustx", "prob_up_gld", "prob_up_vix"]
].ffill()
df_cont = df_cont.dropna(subset=["prob_up_spy"])

spy_thresholds = np.arange(0.48, 0.60, 0.01)
vix_thresholds = np.arange(0.55, 0.76, 0.05)

results = []
print(
    "SPY_Thresh | VIX_Thresh | Strat Return | Max DD | SPY Exposure | VUSTX Expo | GLD Expo | CASH Expo"
)
print("-" * 100)

for s_t in spy_thresholds:
    for v_t in vix_thresholds:
        alloc = []
        rets = []
        for idx, row in df_cont.iterrows():
            if row["prob_up_vix"] > v_t:
                alloc.append("CASH")
                rets.append(0.0)
            elif row["prob_up_spy"] >= s_t:
                alloc.append("SPY")
                rets.append(row["asset_return"])
            else:
                if (
                    row["prob_up_vustx"] > row["prob_up_gld"]
                    and row["prob_up_vustx"] > 0.50
                ):
                    alloc.append("VUSTX")
                    rets.append(row["vustx_ret"])
                elif (
                    row["prob_up_gld"] > row["prob_up_vustx"]
                    and row["prob_up_gld"] > 0.50
                ):
                    alloc.append("GLD")
                    rets.append(row["gld_ret"])
                elif row["vustx_close_prev"] > row["vustx_sma200_prev"]:
                    alloc.append("VUSTX")
                    rets.append(row["vustx_ret"])
                elif row["gld_close_prev"] > row["gld_sma200_prev"]:
                    alloc.append("GLD")
                    rets.append(row["gld_ret"])
                else:
                    alloc.append("CASH")
                    rets.append(0.0)

        df_cont["strategy_allocation"] = alloc
        df_cont["strategy_return"] = rets

        cum_strat = (1 + df_cont["strategy_return"]).cumprod()
        final_ret = cum_strat.iloc[-1]

        cum = (1 + df_cont["strategy_return"]).cumprod()
        peak = cum.expanding(min_periods=1).max()
        dd = (cum / peak) - 1
        mdd = dd.min()

        alloc_counts = df_cont["strategy_allocation"].value_counts(normalize=True)
        spy_exp = alloc_counts.get("SPY", 0)
        vustx_exp = alloc_counts.get("VUSTX", 0)
        gld_exp = alloc_counts.get("GLD", 0)
        cash_exp = alloc_counts.get("CASH", 0)

        results.append(
            {
                "spy_t": s_t,
                "vix_t": v_t,
                "ret": final_ret,
                "mdd": mdd,
                "spy": spy_exp,
                "vustx": vustx_exp,
                "gld": gld_exp,
                "cash": cash_exp,
            }
        )

        print(
            f"{s_t:.2f}       | {v_t:.2f}       | {final_ret:>10.2f}x | {mdd:>6.2%} | {spy_exp:>12.1%} | {vustx_exp:>10.1%} | {gld_exp:>8.1%} | {cash_exp:>9.1%}"
        )

results_df = pd.DataFrame(results)
best = results_df.loc[results_df["ret"].idxmax()]
print(f"\nOPTIMAL THRESHOLDS --> SPY: {best['spy_t']:.2f} | VIX: {best['vix_t']:.2f}")
print(f"Total Yield: {best['ret']:.2f}x (Max DD: {best['mdd']:.2%})")

import os
import pandas as pd
from dotenv import load_dotenv
import logging
import sqlite3
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.xlstm_wrapper import xLSTMForecast

logging.basicConfig(level=logging.INFO)


def execute_backtest():
    logging.info(
        "[Backtest] Initiating Full-Timeline Walk-Forward Backtest on MULTI-ASSET Global Pipeline..."
    )

    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(env_path)

    logging.info("[Backtest] Extracting SQLite Core Market Database...")
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "market_data.db")
    conn = sqlite3.connect(db_path)

    query = """
        SELECT Date as ds, 
               SPY_CLOSE as spy_close,
               VUSTX_CLOSE as vustx_close,
               GLD_CLOSE as gld_close,
               VIX_CLOSE as vix_close,
               SPY_VOLUME as volume,
               SPY_HIGH as high,
               SPY_LOW as low,
               T10Y2Y as t10y2y,
               VIX_BB_WIDTH as vix_bb_width,
               SPY_VWAP_20 as spy_vwap_20,
               NET_NYHGH_NYLOW_SMA_10 as nyhgh_nylow_10d,
               "World_CentralBank_BalSh_45d%Chg" as world_cb_liq,
               VIX_TNX_PCT_FROM_200 as vix_tnx_pct_200,
               SPY_PCT_FROM_200 as spy_pct_200,
               NYA200R as nya200r,
               VIX_TNX_TSI as vix_tnx_tsi,
               VIX_TSI as vix_tsi,
               VIX_VVIX_RATIO as vix_vvix_ratio,
               SPY_PPO_SIGNAL as spy_ppo_signal,
               VIX_VVIX_RATIO_Z as vix_vvix_ratio_z,
               SENT_CROSSASSET_VOL_RATIO as sent_crossasset_vol_ratio,
               SENT_0DTE_REPRESSION_Z as sent_0dte_repression_z,
               VIX_VXV_RATIO_Z as vix_vxv_ratio_z,
               SPY_VOL_RATIO_21_252 as spy_vol_ratio_21_252,
               SPY_ACCEL_MOM as spy_accel_mom,
               SPY_ATR_14 as spy_atr_14,
               SPY_ATR_PCT as spy_atr_pct,
               MARKET_BREADTH_ZSCORE_252D as market_breadth_zscore_252d,
               VIX_TO_10Y_MOM21 as vix_to_10y_mom21,
               CORR_SPY_VUSTX_63 as corr_spy_vustx_63
        FROM core_market_table
    """
    df_core = pd.read_sql_query(query, conn)
    conn.close()

    df_core["ds"] = pd.to_datetime(df_core["ds"]).dt.tz_localize(None)

    # Pre-calculate safe-haven returns and historical SMAs on the full absolute timeline
    df_returns = df_core[["ds", "vustx_close", "gld_close"]].copy()
    df_returns["vustx_ret"] = df_returns["vustx_close"].pct_change()
    df_returns["gld_ret"] = df_returns["gld_close"].pct_change()
    df_returns["vustx_close_prev"] = df_returns["vustx_close"].shift(1)
    df_returns["gld_close_prev"] = df_returns["gld_close"].shift(1)
    df_returns["vustx_sma200_prev"] = df_returns["vustx_close_prev"].rolling(200).mean()
    df_returns["gld_sma200_prev"] = df_returns["gld_close_prev"].rolling(200).mean()

    # Build Panel Data Blocks
    panels = []
    assets_map = {
        "SPY": "spy_close",
        "VUSTX": "vustx_close",
        "GLD": "gld_close",
        "VIX": "vix_close",
    }

    for uid, target_col in assets_map.items():
        df_sub = df_core.copy()
        df_sub["unique_id"] = uid
        df_sub["PRICEMAP"] = df_sub[target_col]
        df_sub["y"] = (df_sub["PRICEMAP"] > df_sub["PRICEMAP"].shift(10)).astype(int)
        panels.append(df_sub)

    df_panel = pd.concat(panels, ignore_index=True)

    hist_exog = [
        "volume",
        "high",
        "low",
        "t10y2y",
        "vix_bb_width",
        "spy_vwap_20",
        "nyhgh_nylow_10d",
        "world_cb_liq",
        "vix_tnx_pct_200",
        "vix_tnx_tsi",
        "vix_tsi",
        "vix_vvix_ratio",
        "spy_ppo_signal",
        "vix_vvix_ratio_z",
        "sent_crossasset_vol_ratio",
        "sent_0dte_repression_z",
        "vix_vxv_ratio_z",
        "spy_vol_ratio_21_252",
        "spy_accel_mom",
        "spy_atr_14",
        "spy_atr_pct",
        "market_breadth_zscore_252d",
        "vix_to_10y_mom21",
        "corr_spy_vustx_63",
        "nya200r",
    ]

    df_model = df_panel[["unique_id", "ds", "y"] + hist_exog].dropna()
    logging.info(
        f"[Backtest] Target Matrix Size (Panel Data): {len(df_model)} Total Rows across 4 Assets"
    )

    HORIZON = 10
    WINDOWS = 400
    STEP_SIZE = 10

    model = xLSTMForecast(
        h=HORIZON, input_size=30, max_steps=10, hist_exog_list=hist_exog, freq="B"
    )

    logging.info("[Backtest] Commencing multi-asset walk-forward permutations ...")
    cv_df = model.cross_validation(df_model, n_windows=WINDOWS, step_size=STEP_SIZE)

    output_path = "xLSTM_backtest_results.csv"
    cv_df.to_csv(output_path, index=False)

    print("==================================================================")
    print(" BACKTEST OOS SAMPLE (Out of Sample Ground Truth vs Quantile Map)")
    print("==================================================================")

    if "xLSTM" in cv_df.columns:
        cv_df["ds"] = pd.to_datetime(cv_df["ds"])
        cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])

        # Grab h=10 prediction
        df_pred = cv_df.groupby(["unique_id", "cutoff"]).last().reset_index()
        # Pivot the df to get columns for each asset probability
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

        # Ground truth for metrics and plotting (We track SPY core returns base)
        df_cont = df_core[["ds", "spy_close"]].copy().dropna()
        df_cont["y_prev"] = df_cont["spy_close"].shift(1)
        df_cont = df_cont.dropna(subset=["y_prev"]).copy()

        # Returns merging
        df_cont = pd.merge(df_cont, df_returns, on="ds", how="left")
        df_cont["asset_return"] = (df_cont["spy_close"] - df_cont["y_prev"]) / df_cont[
            "y_prev"
        ]

        df_cont = pd.merge(
            df_cont, df_pred_pivot, left_on="ds", right_on="signal_date", how="left"
        )
        # Forward fill all probabilities
        df_cont[["prob_up_spy", "prob_up_vustx", "prob_up_gld", "prob_up_vix"]] = (
            df_cont[
                ["prob_up_spy", "prob_up_vustx", "prob_up_gld", "prob_up_vix"]
            ].ffill()
        )
        df_cont = df_cont.dropna(subset=["prob_up_spy"])

        # =========================================================================
        # CROSS ASSET LOGIC
        # =========================================================================
        alloc = []
        rets = []
        for idx, row in df_cont.iterrows():
            if row["prob_up_vix"] > 0.60:
                # Contagion trigger overrides everything -> CASH
                alloc.append("CASH")
                rets.append(0.0)
            elif row["prob_up_spy"] >= 0.51:
                # Standard equity clearance
                alloc.append("SPY")
                rets.append(row["asset_return"])
            else:
                # Equity rejected: Compare bonds and gold directly by raw algorithmic prob
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

        df_cont["cum_asset"] = (1 + df_cont["asset_return"]).cumprod()
        df_cont["cum_strat"] = (1 + df_cont["strategy_return"]).cumprod()

        def calculate_mdd(returns):
            cum = (1 + returns).cumprod()
            peak = cum.expanding(min_periods=1).max()
            dd = (cum / peak) - 1
            return dd.min()

        print(
            f"-> Total Asset Return                 : {df_cont['cum_asset'].iloc[-1]:.2f}x"
        )
        print(
            f"-> Total Multi-Asset Strategy Return  : {df_cont['cum_strat'].iloc[-1]:.2f}x"
        )
        print(
            f"-> Asset Max Drawdown                 : {calculate_mdd(df_cont['asset_return']):.2%}"
        )
        print(
            f"-> Strategy Max Drawdown              : {calculate_mdd(df_cont['strategy_return']):.2%}"
        )

        initial_investment = 100000
        print(
            f"\n-> $100k invested in S&P 500 becomes  : ${initial_investment * df_cont['cum_asset'].iloc[-1]:,.2f}"
        )
        print(
            f"-> $100k invested in Model becomes    : ${initial_investment * df_cont['cum_strat'].iloc[-1]:,.2f}"
        )

        alloc_counts = df_cont["strategy_allocation"].value_counts(normalize=True) * 100
        print("\n--- Strategy Allocation Exposure (%) ---")
        print(alloc_counts.round(1).to_string())

        df_cont["year"] = df_cont["ds"].dt.year
        yearly = df_cont.groupby("year").apply(
            lambda x: pd.Series(
                {
                    "Asset (%)": ((1 + x["asset_return"]).prod() - 1) * 100,
                    "Model (%)": ((1 + x["strategy_return"]).prod() - 1) * 100,
                }
            )
        )
        print("\n--- Year-by-Year Performance ---")
        print(yearly.round(2).to_string())
        print("==================================================================")


if __name__ == "__main__":
    execute_backtest()

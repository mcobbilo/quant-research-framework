import os
import pandas as pd
import numpy as np
import logging
import sqlite3
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.xlstm_wrapper import xLSTMForecast
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.ERROR)


def run_audit(seed=42, noise_std=0.0, guillotine=False, test_name=""):
    print(f"\n[{test_name}] INIT...")

    db_path = os.path.join("src", "data", "market_data.db")
    conn = sqlite3.connect(db_path)

    query = """
        SELECT Date as ds, 
               SPY_CLOSE as spy_close, VUSTX_CLOSE as vustx_close, GLD_CLOSE as gld_close, VIX_CLOSE as vix_close,
               SPY_VOLUME as volume, SPY_HIGH as high, SPY_LOW as low, T10Y2Y as t10y2y, VIX_BB_WIDTH as vix_bb_width, SPY_VWAP_20 as spy_vwap_20,
               NET_NYHGH_NYLOW_SMA_10 as nyhgh_nylow_10d, "World_CentralBank_BalSh_45d%Chg" as world_cb_liq,
               VIX_TNX_PCT_FROM_200 as vix_tnx_pct_200, SPY_PCT_FROM_200 as spy_pct_200, NYA200R as nya200r,
               VIX_TNX_TSI as vix_tnx_tsi, VIX_TSI as vix_tsi, VIX_VVIX_RATIO as vix_vvix_ratio, SPY_PPO_SIGNAL as spy_ppo_signal,
               VIX_VVIX_RATIO_Z as vix_vvix_ratio_z, SENT_CROSSASSET_VOL_RATIO as sent_crossasset_vol_ratio,
               SENT_0DTE_REPRESSION_Z as sent_0dte_repression_z, VIX_VXV_RATIO_Z as vix_vxv_ratio_z,
               SPY_VOL_RATIO_21_252 as spy_vol_ratio_21_252, SPY_ACCEL_MOM as spy_accel_mom, SPY_ATR_14 as spy_atr_14,
               SPY_ATR_PCT as spy_atr_pct, MARKET_BREADTH_ZSCORE_252D as market_breadth_zscore_252d,
               VIX_TO_10Y_MOM21 as vix_to_10y_mom21, CORR_SPY_VUSTX_63 as corr_spy_vustx_63
        FROM core_market_table
    """
    df_core = pd.read_sql_query(query, conn)
    conn.close()

    df_core["ds"] = pd.to_datetime(df_core["ds"]).dt.tz_localize(None)

    # GUILLOTINE TEST: We drop the last 1000 days entirely from history before ANY evaluation
    if guillotine:
        latest = df_core["ds"].max()
        cutoff_date = latest - pd.Timedelta(days=1000)
        df_core = df_core[df_core["ds"] < cutoff_date].copy()

    df_returns = df_core[["ds", "vustx_close", "gld_close"]].copy()
    df_returns["vustx_ret"] = df_returns["vustx_close"].pct_change()
    df_returns["gld_ret"] = df_returns["gld_close"].pct_change()
    df_returns["vustx_close_prev"] = df_returns["vustx_close"].shift(1)
    df_returns["gld_close_prev"] = df_returns["gld_close"].shift(1)
    df_returns["vustx_sma200_prev"] = df_returns["vustx_close_prev"].rolling(200).mean()
    df_returns["gld_sma200_prev"] = df_returns["gld_close_prev"].rolling(200).mean()

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

    # NOISE INJECTION
    if noise_std > 0:
        np.random.seed(seed)
        for col in hist_exog:
            # Inject noise proportional to the standard deviation of the feature
            std_dev = df_panel[col].std()
            df_panel[col] = df_panel[col] + np.random.normal(
                0, std_dev * noise_std, len(df_panel)
            )

    df_model = df_panel[["unique_id", "ds", "y"] + hist_exog].dropna()

    WINDOWS = 400 if not guillotine else 300  # Shorter walkforward for truncated data
    HORIZON = 10
    STEP_SIZE = 10

    model = xLSTMForecast(
        h=HORIZON,
        input_size=30,
        max_steps=10,
        hist_exog_list=hist_exog,
        freq="B",
        random_seed=seed,
    )

    cv_df = model.cross_validation(df_model, n_windows=WINDOWS, step_size=STEP_SIZE)

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

    df_cont = df_core[["ds", "spy_close"]].copy().dropna()
    df_cont["y_prev"] = df_cont["spy_close"].shift(1)
    df_cont = df_cont.dropna(subset=["y_prev"]).copy()

    df_cont = pd.merge(df_cont, df_returns, on="ds", how="left")
    df_cont["asset_return"] = (df_cont["spy_close"] - df_cont["y_prev"]) / df_cont[
        "y_prev"
    ]

    df_cont = pd.merge(
        df_cont, df_pred_pivot, left_on="ds", right_on="signal_date", how="left"
    )
    df_cont[["prob_up_spy", "prob_up_vustx", "prob_up_gld", "prob_up_vix"]] = df_cont[
        ["prob_up_spy", "prob_up_vustx", "prob_up_gld", "prob_up_vix"]
    ].ffill()
    df_cont = df_cont.dropna(subset=["prob_up_spy"])

    alloc = []
    rets = []
    for idx, row in df_cont.iterrows():
        if row["prob_up_vix"] > 0.60:
            alloc.append("CASH")
            rets.append(0.0)
        elif row["prob_up_spy"] >= 0.51:
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
                row["prob_up_gld"] > row["prob_up_vustx"] and row["prob_up_gld"] > 0.50
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

    cum_asset = (1 + df_cont["asset_return"]).cumprod()
    base_ret = cum_asset.iloc[-1]

    cum = (1 + df_cont["strategy_return"]).cumprod()
    peak = cum.expanding(min_periods=1).max()
    mdd = ((cum / peak) - 1).min()

    print(
        f"[{test_name}] COMPLETE | Asset Yield: {base_ret:.2f}x | Model Yield: {final_ret:.2f}x | Max DD: {mdd:.2%}"
    )
    return final_ret, mdd


if __name__ == "__main__":
    print("================================")
    print(" RED TEAM AUDIT INITIALIZED")
    print("================================")
    run_audit(seed=0, noise_std=0.0, guillotine=False, test_name="SEED_0_PERTURBATION")
    run_audit(
        seed=1337, noise_std=0.0, guillotine=False, test_name="SEED_1337_PERTURBATION"
    )
    run_audit(
        seed=42, noise_std=0.05, guillotine=False, test_name="5PCT_NOISE_INJECTION"
    )
    run_audit(
        seed=42, noise_std=0.0, guillotine=True, test_name="DATA_GUILLOTINE_TIMETRAVEL"
    )

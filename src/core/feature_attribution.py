import os
import pandas as pd
import numpy as np
import logging
import sys

# Ensure correct pathing when running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.xlstm_wrapper import xLSTMForecast
import sqlite3


def run_attribution():
    # Keep massive Nixtla PyTorch output out of terminal unless breaking
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    print("\n=======================================================")
    print("         PERMUTATION FEATURE IMPORTANCE ENGINE         ")
    print("=======================================================")

    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "market_data.db")
    conn = sqlite3.connect(db_path)

    query = """
        SELECT Date as ds, 
               SPY_CLOSE as y,
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
               NYA200R as nya200r
        FROM core_market_table
    """
    df_core = pd.read_sql_query(query, conn)
    conn.close()

    df_core["ds"] = pd.to_datetime(df_core["ds"]).dt.tz_localize(None)
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
    ]

    df_core = df_core[["unique_id", "ds", "y"] + hist_exog].dropna()

    # Optimization: Calculate over last 800 days for high-speed cross-validation loops
    df_recent = df_core.tail(800).reset_index(drop=True)

    WINDOWS = 6

    def evaluate_model(df, verbose=False):
        model = xLSTMForecast(
            h=5, input_size=12, max_steps=30, hist_exog_list=hist_exog, freq="B"
        )
        if not verbose:
            import warnings

            warnings.filterwarnings("ignore")
        try:
            cv_df = model.cross_validation(df, n_windows=WINDOWS, step_size=5)
            # Calculate Ground Truth RMSE vs Prediction
            rmse = ((cv_df["y"] - cv_df["xLSTM-median"]) ** 2).mean() ** 0.5
            return rmse
        except Exception as e:
            print(f"Failure during cv: {e}")
            return 9999.0

    print(f"-> Phase 1: Calculating Baseline Truth Network ({WINDOWS} CV Windows)...")
    base_rmse = evaluate_model(df_recent, verbose=True)
    print(f"-> Baseline Out-of-Sample RMSE: {base_rmse:.4f}\n")

    importance = {}

    print("-> Phase 2: Firing Stochastic Feature Disruption...")
    for feature in hist_exog:
        df_shuffled = df_recent.copy()

        # Obliterate mathematical continuity for this single vector completely randomly
        np.random.seed(42)
        df_shuffled[feature] = np.random.permutation(df_shuffled[feature].values)

        shuff_rmse = evaluate_model(df_shuffled)

        # Positive Delta = Essential structural feature
        # Negative Delta = Pure noise feature blinding the network
        delta_rmse = (shuff_rmse - base_rmse) / base_rmse * 100
        importance[feature] = delta_rmse
        print(f"   [ {feature.ljust(18)} ] ━━━━ Error Shift: {delta_rmse:+.2f}%")

    print("\n=======================================================")
    print("             XLSTM FEATURE LEADERBOARD                 ")
    print("=======================================================")
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, delta in sorted_features:
        if delta > 1.0:
            print(f"{feat.ljust(20)} | {delta:+.2f}% Error Spike (CRITICAL)")
        elif delta > 0.0:
            print(f"{feat.ljust(20)} | {delta:+.2f}% Error Spike (MINOR)")
        else:
            print(f"{feat.ljust(20)} | {delta:+.2f}% Error Drop (NOISE ALLOCATION)")
    print("=======================================================\n")


if __name__ == "__main__":
    run_attribution()

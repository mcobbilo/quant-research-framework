import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import logging
import argparse
import gc
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.timesfm_wrapper import TimesFMWrapper
from src.api.karpathy_ide import dispatch_obsidian_response

logging.basicConfig(level=logging.INFO)

DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "market_data.db")
)

BASE_FEATURES = [
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

# Mapping the original Nixtla base features to the actual SQLite columns
REAL_COLUMN_MAP = {
    "volume": "SPY_VOLUME",
    "high": "SPY_HIGH",
    "low": "SPY_LOW",
    "t10y2y": "T10Y2Y",
    "vix_bb_width": "VIX_BB_WIDTH",
    "spy_vwap_20": "SPY_VWAP_20",
    "nyhgh_nylow_10d": "NET_NYHGH_NYLOW_SMA_10",
    "world_cb_liq": "World_CentralBank_BalSh_45d%Chg",
    "vix_tnx_pct_200": "VIX_TNX_PCT_FROM_200",
}


def get_candidates():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT * FROM core_market_table LIMIT 1")
    cols = [description[0] for description in cursor.description]
    conn.close()

    ignore_cols = ["Date", "SPY_CLOSE", "unique_id", "ds", "y"] + list(
        REAL_COLUMN_MAP.values()
    )

    candidates = [c for c in cols if c not in ignore_cols]
    return candidates


def calculate_rmse(cv_df):
    if "y" in cv_df.columns and "xLSTM-median" in cv_df.columns:
        return ((cv_df["y"] - cv_df["xLSTM-median"]) ** 2).mean() ** 0.5
    return float("inf")


def evaluate_subset(exog_subset, n_windows, max_steps):
    """
    Evaluates the backend sequence wrapper.
    NOTE: Designed modularly so xLSTMForecast can be hot-swapped for
    a TimesFMWrapper (google-research/timesfm) instantly based on recent feedback.
    """
    conn = sqlite3.connect(DB_PATH)

    query_selects = ["Date as ds", "SPY_CLOSE as y"]

    for friendly_name, real_col in REAL_COLUMN_MAP.items():
        query_selects.append(f'"{real_col}" as {friendly_name}')

    for f in exog_subset:
        if f not in REAL_COLUMN_MAP.values():
            query_selects.append(f'"{f}"')

    query = f"SELECT {', '.join(query_selects)} FROM core_market_table"

    df_core = pd.read_sql_query(query, conn)
    conn.close()

    df_core["ds"] = pd.to_datetime(df_core["ds"]).dt.tz_localize(None)
    df_core["unique_id"] = "SPY"

    hist_exog = BASE_FEATURES + exog_subset

    df_core = df_core[["unique_id", "ds", "y"] + hist_exog].dropna()

    # We enforce a hard floor of observations
    if len(df_core) < 100:
        return float("inf")

    # Hot-swapped xLSTM with TimesFMWrapper logic.
    model = TimesFMWrapper(
        h=10,
        input_size=128,  # Increased minimum context required for foundation models
        max_steps=max_steps,
        hist_exog_list=hist_exog,
        freq="B",
    )

    try:
        cv_df = model.cross_validation(df_core, n_windows=n_windows, step_size=40)
        return calculate_rmse(cv_df)
    except Exception as e:
        logging.error(f"Error evaluating {exog_subset}: {e}")
        return float("inf")


def run_arena(fast_fail=True):
    candidates = get_candidates()
    logging.info(
        f"[Arena] Found {len(candidates)} potential candidate features for mutation."
    )

    n_windows = 2 if fast_fail else 35
    max_steps = 5 if fast_fail else 50

    # Evaluate Baseline
    logging.info("[Arena] Establishing Baseline Root Mean Square Error (RMSE)...")
    baseline_rmse = evaluate_subset([], n_windows, max_steps)
    logging.info(f"[Arena] Baseline RMSE encoded: {baseline_rmse:.4f}")

    results = []

    if fast_fail:
        # Testing 2 random candidates out of the gate for this execution cycle
        np.random.seed(42)
        test_subset = list(np.random.choice(candidates, size=2, replace=False))
    else:
        test_subset = candidates

    for candidate in test_subset:
        logging.info(f"[Arena] Genetic Mutation -> Injecting: {candidate}")
        rmse = evaluate_subset([candidate], n_windows, max_steps)

        improvement = baseline_rmse - rmse
        results.append(
            {"mutation": candidate, "rmse": rmse, "improvement": improvement}
        )

        # Free MPS memory to prevent swapping / lockups
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    results.sort(key=lambda x: x["rmse"])

    markdown_lines = []
    markdown_lines.append("# Combinatorial Optimization Sweep")
    markdown_lines.append(
        f"Tested mutating existing 9-feature baseline with `{len(test_subset)}` new SQLite candidates."
    )
    markdown_lines.append(f"*Note: Evaluated in fast_fail={fast_fail} mode.*")
    markdown_lines.append("\n**Top Mutations Identified:**")

    for r in results:
        sym = "🟢" if r["improvement"] > 0 else "🔴"
        markdown_lines.append(
            f"- {sym} **{r['mutation']}**: RMSE: {r['rmse']:.4f} (Diff vs Base: {r['improvement']:.4f})"
        )

    markdown_lines.append(
        "\n*Model configuration remains modular to accept continuous TimesFM upgrades.*"
    )

    md_content = "\n".join(markdown_lines)
    dispatch_obsidian_response("Identify optimal feature genetic mutations", md_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="fast_fail")
    args = parser.parse_args()

    run_arena(fast_fail=(args.mode == "fast_fail"))

import sys
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Pytorch Forecasting dependencies
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

# Connect Kronos repo
sys.path.append("/Users/milocobb/tft_model/Kronos_repo")
from model import KronosTokenizer, Kronos, KronosPredictor

# Import Liquidity Telemetry
sys.path.append("/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/src/data")
from liquidity_telemetry import fetch_net_dollar_liquidity


def get_phase126_compass_and_vetoes(test_dates):
    """
    Phase 2: Train the 126 Allocator and calculate macro vetoes.
    Training window iterates exactly out of sample preventing lookahead.
    """
    print("\nInitializing Phase 126 Compass & Breadth Vetoes...")
    import sqlite3
    db_path = "src/data/market_data.db"
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT Date as ds, 
               SPY_CLOSE as spy_close, VUSTX_CLOSE as vustx_close, GLD_CLOSE as gld_close, VIX_CLOSE as vix_close, HG_CLOSE as hg_close,
               SPY_VOLUME as volume, SPY_HIGH as high, SPY_LOW as low, T10Y2Y as t10y2y, VIX_BB_WIDTH as vix_bb_width, SPY_VWAP_20 as spy_vwap_20,
               NET_NYHGH_NYLOW_SMA_10 as nyhgh_nylow_10d, "World_CentralBank_BalSh_45d%Chg" as world_cb_liq,
               VIX_TNX_PCT_FROM_200 as vix_tnx_pct_200, SPY_PCT_FROM_200 as spy_pct_200, NYA200R as nya200r,
               VIX_TNX_TSI as vix_tnx_tsi, VIX_TSI as vix_tsi, VIX_VVIX_RATIO as vix_vvix_ratio, SPY_PPO_SIGNAL as spy_ppo_signal,
               VIX_VVIX_RATIO_Z as vix_vvix_ratio_z, SENT_CROSSASSET_VOL_RATIO as sent_crossasset_vol_ratio,
               SENT_0DTE_REPRESSION_Z as sent_0dte_repression_z, VIX_VXV_RATIO_Z as vix_vxv_ratio_z,
               SPY_VOL_RATIO_21_252 as spy_vol_ratio_21_252, SPY_ACCEL_MOM as spy_accel_mom, SPY_ATR_14 as spy_atr_14,
               SPY_ATR_PCT as spy_atr_pct, MARKET_BREADTH_ZSCORE_252D as market_breadth_zscore_252d,
               VIX_TO_10Y_MOM21 as vix_to_10y_mom21, CORR_SPY_VUSTX_63 as corr_spy_vustx_63, SKEW_ZSCORE_252 as skew_zscore_252
        FROM core_market_table
    """
    df_core = pd.read_sql_query(query, conn)
    conn.close()

    df_core["ds"] = pd.to_datetime(df_core["ds"]).dt.tz_localize(None)
    df_core.sort_values("ds", inplace=True)
    df_core.reset_index(drop=True, inplace=True)

    df_core["hg_close_prev"] = df_core["hg_close"].shift(1)
    df_core["hg_sma200_prev"] = df_core["hg_close_prev"].rolling(200).mean()
    df_core["vix_close_prev"] = df_core["vix_close"].shift(1)

    features = [
        "volume", "high", "low", "t10y2y", "vix_bb_width", "spy_vwap_20", "nyhgh_nylow_10d", "world_cb_liq",
        "vix_tnx_pct_200", "vix_tnx_tsi", "vix_tsi", "vix_vvix_ratio", "spy_ppo_signal", "vix_vvix_ratio_z",
        "sent_crossasset_vol_ratio", "sent_0dte_repression_z", "vix_vxv_ratio_z", "spy_vol_ratio_21_252",
        "spy_accel_mom", "spy_atr_14", "spy_atr_pct", "market_breadth_zscore_252d", "vix_to_10y_mom21",
        "corr_spy_vustx_63", "nya200r"
    ]
    df_core[features] = df_core[features].ffill().bfill()
    HORIZON = 10
    df_core["y_spy"] = np.where(df_core["spy_close"].shift(-HORIZON) > df_core["spy_close"], 1, 0)
    df_core["y_vustx"] = np.where(df_core["vustx_close"].shift(-HORIZON) > df_core["vustx_close"], 1, 0)
    df_core["y_gld"] = np.where(df_core["gld_close"].shift(-HORIZON) > df_core["gld_close"], 1, 0)
    
    # Calculate Macro Vetoes
    df_core["copper_strength"] = np.where(df_core["hg_sma200_prev"] > 0, df_core["hg_close_prev"] / df_core["hg_sma200_prev"], 1.0)
    df_core["Veto_DrCopper"] = df_core["copper_strength"] < 1.04
    df_core["Veto_Breadth"] = df_core["nya200r"] < 0.10
    
    df_core["gex_proxy"] = df_core["skew_zscore_252"] + df_core["vix_vxv_ratio_z"] + df_core["sent_0dte_repression_z"]
    df_core["Veto_VIX_Panic"] = (df_core["vix_close_prev"] > 20.0) & (df_core["gex_proxy"] <= 1.5)
    
    df_core["XGBoost_Prob"] = np.nan
    df_core["XGBoost_Prob_VUSTX"] = np.nan
    df_core["XGBoost_Prob_GLD"] = np.nan
    
    total_dates = pd.to_datetime(test_dates).sort_values()
    
    base_params = {
        "n_estimators": 150, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8,
        "colsample_bytree": 0.8, "eval_metric": "logloss", "random_state": 42, "n_jobs": -1
    }
    clf_spy = xgb.XGBClassifier(**base_params)
    clf_vustx = xgb.XGBClassifier(**base_params)
    clf_gld = xgb.XGBClassifier(**base_params)
    
    from tqdm import tqdm
    step_size = 21  # Retrain model every ~month (21 trading days)
    next_train_date = total_dates[0]
    
    print("\nExecuting Data Guillotine Expanding Window...")
    for dt in tqdm(total_dates):
        if dt >= next_train_date:
            # Drop last 10 days to enforce time-travel prevention
            train_df = df_core[df_core["ds"] <= dt].iloc[:-10]
            train_df = train_df.iloc[-1260:] # 5-Year fixed horizon trailing
            X_train = train_df[features]
            if len(X_train) > 0:
                clf_spy.fit(X_train, train_df["y_spy"])
                clf_vustx.fit(X_train, train_df["y_vustx"])
                clf_gld.fit(X_train, train_df["y_gld"])
            next_train_date = dt + pd.Timedelta(days=step_size)
        
        # OOS Inference for just the current date
        if hasattr(clf_spy, "classes_"):
            X_test_row = df_core[df_core["ds"] == dt][features]
            if len(X_test_row) > 0:
                prob_spy = clf_spy.predict_proba(X_test_row)[0][1]
                prob_vustx = clf_vustx.predict_proba(X_test_row)[0][1]
                prob_gld = clf_gld.predict_proba(X_test_row)[0][1]
                idx = df_core[df_core["ds"] == dt].index
                df_core.loc[idx, "XGBoost_Prob"] = prob_spy
                df_core.loc[idx, "XGBoost_Prob_VUSTX"] = prob_vustx
                df_core.loc[idx, "XGBoost_Prob_GLD"] = prob_gld

    df_core["Date"] = pd.to_datetime(df_core["ds"])
    return df_core[["Date", "XGBoost_Prob", "XGBoost_Prob_VUSTX", "XGBoost_Prob_GLD", "Veto_DrCopper", "Veto_Breadth", "Veto_VIX_Panic"]]


def get_kronos_predictions(test_dates):
    """
    Phase 3: Run Kronos Autoregressive Engine
    """
    print("\nInitializing Kronos Autoregressive Micro-Momentum Engine...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(
        device
    )
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base").to(device)
    predictor = KronosPredictor(model, tokenizer, device=device)

    # Load raw Yahoo data for OHLCV
    import yfinance as yf

    spy = yf.download("SPY", start="2024-01-01", end="2026-03-01", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.columns = [
        c.lower() for c in spy.columns
    ]  # kronos expects ['open', 'high', 'low', 'close', 'volume', 'amount']
    spy["amount"] = spy["volume"] * ((spy["open"] + spy["close"]) / 2.0)

    kronos_preds = []

    # For every day out of sample, feed the predictor trailing history.
    for dt in test_dates:
        # History UP TO AND INCLUDING `dt`
        hist_df = spy[spy.index <= dt].copy()

        # We need future timestamps correctly aligned
        # In real backtests, weekends don't trade. But we just grab the next 5 actual trading days of `spy` as temporal index
        future_dt = spy[spy.index > dt].index[:5]
        if len(future_dt) < 5:
            # Fake it if we ran out of calendar
            future_dt = pd.date_range(dt + pd.Timedelta(days=1), periods=5, freq="B")

        # Truncate context to 512
        hist_df = hist_df.iloc[-512:]

        x_timestamp = pd.Series(hist_df.index)
        y_timestamp = pd.Series(future_dt)

        # generate 5 day projection
        pred_df = predictor.predict(
            hist_df,
            x_timestamp,
            y_timestamp,
            pred_len=5,
            T=0.8,
            top_p=0.9,
            sample_count=3,
            verbose=False,
        )

        # Calculate expected micro-return over 5 days (Proj_Close / Current_Close - 1)
        curr_close = hist_df["close"].iloc[-1]
        proj_close = pred_df["close"].iloc[-1]
        five_d_ret = (proj_close / curr_close) - 1.0

        kronos_preds.append(five_d_ret)

    res_df = pd.DataFrame({"Date": test_dates, "Kronos_5D_Prob": kronos_preds})
    return res_df


def run_dual_brain_backtest():
    print("=" * 60)
    print(" INITIALIZING DUAL-BRAIN LIVE EXECUTION (V4.3) ")
    print("=" * 60)

    # ----------------------------------------------------
    # LOAD GLOBAL DATAFRAME
    # ----------------------------------------------------
    print("Loading Core Master Features...")
    df = pd.read_parquet(
        "/Users/milocobb/tft_model/clean_aligned_features_27yr.parquet"
    )
    df.columns = [c.replace(".", "_") for c in df.columns]

    if "Kronos_S1" in df.columns:
        df["Kronos_S1"] = df["Kronos_S1"].apply(lambda x: np.nan if pd.isna(x) else str(int(float(x))))
        df["Kronos_S2"] = df["Kronos_S2"].apply(lambda x: np.nan if pd.isna(x) else str(int(float(x))))

    if "directional_impact" in df.columns:
        df["directional_impact"] = df["directional_impact"].ffill().fillna(0.0)
        df["fomc_regime"] = df["directional_impact"].ewm(span=21, adjust=False).mean()

    df["SPY_Log_Return"] = np.log(df["SPY"] / df["SPY"].shift(1)).fillna(0)
    df["Daily_Return"] = df["SPY"].pct_change().fillna(0)
    df["VUSTX_Daily_Return"] = df["TLT"].pct_change().fillna(0)
    df["GLD_Daily_Return"] = df["GLD"].pct_change().fillna(0)
    df = df.fillna(0)

    if df.index.name == "Date":
        df = df.reset_index()

    df["time_idx"] = np.arange(len(df))
    df["group_id"] = "portfolio"

    # Define validation Out-of-Sample boundaries
    # Eradicates the 21-day target latency boundary, allowing flush prediction up to t
    training_cutoff = df["time_idx"].max() - 126
    oos_start_idx = training_cutoff + 1
    true_oos_dates = df[df["time_idx"] >= oos_start_idx]["Date"].tolist()

    # ----------------------------------------------------
    # PHASE 1: TFT MACRO EXTRACTION
    # ----------------------------------------------------
    print("\nPhase 1: Computing Temporal Fusion Transformer (Macro) Sequences...")

    ckpt_path = "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/lightning_logs/version_1166/checkpoints/epoch=10-step=2596.ckpt"
    model = TemporalFusionTransformer.load_from_checkpoint(
        ckpt_path, map_location="cpu", weights_only=False
    )

    dataset = TimeSeriesDataSet.from_parameters(
        model.dataset_parameters, df, min_prediction_idx=oos_start_idx
    )
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    model.eval()
    preds_list = []
    idx_list = []

    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            p = model.to_prediction(out)
            final_p = p.sum(dim=1).numpy()
            final_idx = x["decoder_time_idx"][:, 0].numpy() - 1
            preds_list.extend(final_p)
            idx_list.extend(final_idx)

    tft_df = pd.DataFrame(
        {"time_idx": np.array(idx_list), "TFT_21D_Return": np.array(preds_list)}
    )
    base_eval_df = pd.merge(
        df[["time_idx", "Date", "SPY", "Daily_Return", "VUSTX_Daily_Return", "GLD_Daily_Return"]],
        tft_df,
        on="time_idx",
        how="inner",
    )
    base_eval_df = base_eval_df[base_eval_df["time_idx"] >= oos_start_idx].copy()

    # ----------------------------------------------------
    # PHASE 2 & 3: XGBOOST COMPASS & KRONOS
    # ----------------------------------------------------
    compass_df = get_phase126_compass_and_vetoes(true_oos_dates)
    kronos_df = get_kronos_predictions(true_oos_dates)

    # ----------------------------------------------------
    # PHASE 4: CONSENSUS MATRIX FUSION & MACRO LIQUIDITY
    # ----------------------------------------------------
    print("\nFusing Telemetries across logical boundaries...")
    final_df = base_eval_df.merge(compass_df, on="Date").merge(kronos_df, on="Date")
    
    print("\nFetching Net Dollar Liquidity Telemetry (FRED)...")
    start_date = final_df["Date"].min().strftime("%Y-%m-%d")
    liquidity_df = fetch_net_dollar_liquidity(start_date=start_date)
    liquidity_df["Date"] = pd.to_datetime(liquidity_df["Date"])
    final_df["Date"] = pd.to_datetime(final_df["Date"])
    
    final_df = final_df.merge(liquidity_df[["Date", "Liquidity_Guillotine", "NDL_ROC_20D", "Net_Dollar_Liquidity"]], on="Date", how="left")
    final_df["Liquidity_Guillotine"].fillna(False, inplace=True)

    # Evaluate Phase 126 Master Protocol
    final_df["Pos_TFT"] = final_df["TFT_21D_Return"] > 0.0
    final_df["Pos_Kronos"] = final_df["Kronos_5D_Prob"] > 0.0
    final_df["Pos_XGB"] = final_df["XGBoost_Prob"] > 0.51

    def evaluate_position(row):
        # Generational Buying Overrides (Buy the Blood)
        if row["Veto_Breadth"]:
            return "SPY"
        if row["Veto_VIX_Panic"]:
            return "SPY"
            
        prob_v = row.get("XGBoost_Prob_VUSTX", 0)
        prob_g = row.get("XGBoost_Prob_GLD", 0)
        
        # Dr Copper primary rotational veto
        if row["Veto_DrCopper"]:
            if prob_v > prob_g and prob_v > 0.50:
                return "VUSTX"
            elif prob_g > prob_v and prob_g > 0.50:
                return "GLD"
            else:
                return "CASH"
        
        # XGBoost is the dominant logic. TFT/Kronos are supplementary confidence flags.
        if row["Pos_XGB"]:
            if row["Liquidity_Guillotine"]:
                if row["Pos_TFT"] and row["Pos_Kronos"]:
                    return "SPY"
                else:
                    if prob_v > prob_g and prob_v > 0.50:
                        return "VUSTX"
                    elif prob_g > prob_v and prob_g > 0.50:
                        return "GLD"
                    else:
                        return "CASH"
            return "SPY"
        else:
            if prob_v > prob_g and prob_v > 0.50:
                return "VUSTX"
            elif prob_g > prob_v and prob_g > 0.50:
                return "GLD"
            else:
                return "CASH"

    final_df["Final_Position"] = final_df.apply(evaluate_position, axis=1)

    # Plot curve
    def get_strat_return(row):
        pos = row["Final_Position_Prev"]
        if pos == "SPY":
            return row["Daily_Return"]
        elif pos == "VUSTX":
            return row["VUSTX_Daily_Return"]
        elif pos == "GLD":
            return row["GLD_Daily_Return"]
        else:
            return 0.0

    final_df["Final_Position_Prev"] = final_df["Final_Position"].shift(1)
    final_df["Strategy_Return"] = final_df.apply(get_strat_return, axis=1)
    final_df["Equity_Curve"] = (1 + final_df["Strategy_Return"].fillna(0)).cumprod()
    final_df["Buy_Hold"] = (1 + final_df["Daily_Return"].fillna(0)).cumprod()

    # ----------------------------------------------------
    # OUTPUTS
    # ----------------------------------------------------
    final_eq = final_df["Equity_Curve"].iloc[-1]
    final_bh = final_df["Buy_Hold"].iloc[-1]
    alpha = (final_eq - final_bh) / final_bh * 100

    print("\n" + "=" * 60)
    print(" TRI-AGENT OUT-OF-SAMPLE 2026 LIVE PERFORMANCE")
    print("=" * 60)
    print(f"Total True Evaluation Days:    {len(final_df)}")
    print(f"Tri-Agent Strategy Equity:     {final_eq:.3f}x")
    print(f"Naive Baseline (Buy/Hold):     {final_bh:.3f}x")
    print(f"Absolute Alpha Generated:      {alpha:+.2f}%")
    print("\nExecution Matrix Generated Successfully. Plotting Trajectory...")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(
        final_df["Date"],
        final_df["Buy_Hold"],
        color="#555555",
        linewidth=2,
        label="SPY Baseline (Buy & Hold)",
    )
    ax.plot(
        final_df["Date"],
        final_df["Equity_Curve"],
        color="#00ff00",
        linewidth=2.5,
        label="Master Tri-Agent Trajectory",
    )

    ax.set_title(
        "Out-of-Sample 2026 Dual-Brain Live Execution",
        fontsize=16,
        color="white",
        pad=20,
    )
    ax.set_ylabel("Sequential Capital Multiple", fontsize=12, color="white")
    ax.grid(color="#2A3459", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", fontsize=12)

    plt.tight_layout()
    out_path = "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/tri_agent_2026_curve.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved High-Fidelity Path Trajectory to {out_path}")


if __name__ == "__main__":
    run_dual_brain_backtest()

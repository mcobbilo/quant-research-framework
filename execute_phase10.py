import os
import pandas as pd
import numpy as np
import sqlite3
import xgboost as xgb
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings("ignore")


def build_phase10_engine():
    print("\n[PHASE 10 ENGINE] INITIALIZING...")

    db_path = os.path.join("src", "data", "market_data.db")
    conn = sqlite3.connect(db_path)

    query = "SELECT * FROM core_market_table"
    df_core = pd.read_sql_query(query, conn)
    conn.close()

    df_core["ds"] = pd.to_datetime(df_core["Date"]).dt.tz_localize(None)
    df_core.sort_values("ds", inplace=True)
    df_core.reset_index(drop=True, inplace=True)

    df_core = df_core.rename(
        columns={
            "SPY_CLOSE": "spy_close",
            "VUSTX_CLOSE": "vustx_close",
            "GLD_CLOSE": "gld_close",
            "VIX_CLOSE": "vix_close",
        }
    )

    HORIZON = 45  # The strict Phase 10 long-horizon definition

    # ---------------------------------------------------------
    # 1. 45-DAY TARGETS & METRICS
    # ---------------------------------------------------------
    df_core["y_cls_45d"] = (
        df_core["spy_close"].shift(-HORIZON) > df_core["spy_close"]
    ).astype(int)
    df_core["y_reg_45d"] = (
        df_core["spy_close"].shift(-HORIZON) - df_core["spy_close"]
    ) / df_core["spy_close"]

    # Inverse Correlation Trackers (Rolling 63-day)
    spy_ret = df_core["spy_close"].pct_change()
    df_core["corr_spy_vustx"] = spy_ret.rolling(63).corr(
        df_core["vustx_close"].pct_change()
    )
    df_core["corr_spy_gld"] = spy_ret.rolling(63).corr(
        df_core["gld_close"].pct_change()
    )

    exclude_cols = [
        "ds",
        "Date",
        "spy_close",
        "vustx_close",
        "gld_close",
        "vix_close",
        "PRICEMAP",
        "unique_id",
        "y_cls_45d",
        "y_reg_45d",
        "corr_spy_vustx",
        "corr_spy_gld",
    ]
    num_features = [c for c in df_core.columns if c not in exclude_cols]
    num_features = (
        df_core[num_features].select_dtypes(include=[np.number]).columns.tolist()
    )

    df_core[num_features] = df_core[num_features].ffill().bfill()
    df_core["corr_spy_vustx"] = df_core["corr_spy_vustx"].ffill().bfill()
    df_core["corr_spy_gld"] = df_core["corr_spy_gld"].ffill().bfill()

    # ---------------------------------------------------------
    # 2. DYNAMIC L1 FEATURE EXTRACTION
    # ---------------------------------------------------------
    valid_train = df_core.iloc[:-HORIZON].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(valid_train[num_features])
    y_target = valid_train["y_cls_45d"].values

    print("Extracting top 60 DNA Features via L1 Regularization...")
    lr = LogisticRegression(
        penalty="l1", solver="liblinear", C=1.0, random_state=42, max_iter=200
    )
    lr.fit(X_scaled, y_target)

    feature_importances = pd.DataFrame(
        {"feature": num_features, "importance": np.abs(lr.coef_[0])}
    )
    feature_importances = feature_importances.sort_values("importance", ascending=False)
    top_60_features = feature_importances.head(60)["feature"].tolist()

    # ---------------------------------------------------------
    # 3. EXPANDING WINDOW WITH GMM REGIME ROUTING
    # ---------------------------------------------------------
    total_steps = 150  # Stepping by 10 days for ~150 periods to cover OOS
    step_size = 10
    end_idx = len(df_core) - HORIZON
    start_idx = end_idx - (total_steps * step_size)

    print("\n[PHASE 10] EXECUTING DUAL-STAGE OOS BACKTEST...")
    results = []

    for i in tqdm(range(total_steps)):
        current_cutoff_idx = start_idx + i * step_size
        train_end_idx = current_cutoff_idx - HORIZON  # Physical OOS Barrier Truncation

        df_train = df_core.iloc[:train_end_idx].copy()
        df_test = df_core.iloc[[current_cutoff_idx]].copy()

        # Train GMM Regime Detector
        # Using SPY Volatility & VIX to determine if we are in High Vol or Low Vol
        gmm_features = ["SPY_ATR_PCT", "vix_close"]
        gmm = GaussianMixture(n_components=2, random_state=42)

        train_gmm_data = df_train[gmm_features].values
        gmm.fit(train_gmm_data)

        # Identify which regime mapped to danger (higher mean VIX)
        means = gmm.means_
        danger_regime = 0 if means[0][1] > means[1][1] else 1

        # Assign Regimes to Train / Test
        df_train["regime"] = gmm.predict(train_gmm_data)
        df_train["regime"] = (df_train["regime"] == danger_regime).astype(
            int
        )  # 1 = Danger, 0 = Safe

        test_gmm_data = df_test[gmm_features].values
        current_regime = 1 if gmm.predict(test_gmm_data)[0] == danger_regime else 0

        # Train Regime-Specific Dual-Stage Experts
        X_train_r0 = df_train[df_train["regime"] == 0][top_60_features]  # Low Vol
        y_train_r0_cls = df_train[df_train["regime"] == 0]["y_cls_45d"]
        y_train_r0_reg = df_train[df_train["regime"] == 0]["y_reg_45d"]

        X_train_r1 = df_train[df_train["regime"] == 1][top_60_features]  # High Vol
        y_train_r1_cls = df_train[df_train["regime"] == 1]["y_cls_45d"]
        y_train_r1_reg = df_train[df_train["regime"] == 1]["y_reg_45d"]

        # ---------------------------------------------------------
        # EXPERT MODELS (Classifier + Custom Magnitude Regressor)
        # ---------------------------------------------------------
        clf_args = dict(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            eval_metric="logloss",
        )
        reg_args = dict(
            n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.8
        )

        X_test = df_test[top_60_features]

        # Asymmetric Weights: Penalize negative target under-prediction 3x
        def custom_asymmetric_weights(y_true):
            return np.where(y_true < 0, 3.0, 1.0)

        prob_up = 0.0
        magnitude = 0.0
        conformal_buffer = 0.0

        if current_regime == 0 and len(X_train_r0) > 50:
            cls_model = xgb.XGBClassifier(**clf_args, random_state=42)
            cls_model.fit(X_train_r0, y_train_r0_cls)
            prob_up = cls_model.predict_proba(X_test)[0][1]

            reg_model = xgb.XGBRegressor(**reg_args, random_state=42)
            reg_weights = custom_asymmetric_weights(y_train_r0_reg)
            reg_model.fit(X_train_r0, y_train_r0_reg, sample_weight=reg_weights)
            magnitude = reg_model.predict(X_test)[0]

            # Conformal Margin of Error on training set
            train_preds = reg_model.predict(X_train_r0)
            residuals = y_train_r0_reg - train_preds
            conformal_buffer = np.std(residuals)

        elif current_regime == 1 and len(X_train_r1) > 50:
            cls_model = xgb.XGBClassifier(**clf_args, random_state=42)
            cls_model.fit(X_train_r1, y_train_r1_cls)
            prob_up = cls_model.predict_proba(X_test)[0][1]

            reg_model = xgb.XGBRegressor(**reg_args, random_state=42)
            reg_weights = custom_asymmetric_weights(y_train_r1_reg)
            reg_model.fit(X_train_r1, y_train_r1_reg, sample_weight=reg_weights)
            magnitude = reg_model.predict(X_test)[0]

            train_preds = reg_model.predict(X_train_r1)
            residuals = y_train_r1_reg - train_preds
            conformal_buffer = np.std(residuals)

        results.append(
            {
                "ds": df_test["ds"].values[0],
                "regime": current_regime,
                "prob_up": prob_up,
                "pred_magnitude": magnitude,
                "conformal_buffer": conformal_buffer,
                "spy_vustx_corr": df_test["corr_spy_vustx"].values[0],
                "spy_gld_corr": df_test["corr_spy_gld"].values[0],
            }
        )

    df_res = pd.DataFrame(results)

    # ---------------------------------------------------------
    # 4. EXECUTION PROTOCOL
    # ---------------------------------------------------------
    df_cont = df_core[["ds", "spy_close", "vustx_close", "gld_close"]].copy().dropna()
    df_cont["spy_ret"] = df_cont["spy_close"].pct_change()
    df_cont["vustx_ret"] = df_cont["vustx_close"].pct_change()
    df_cont["gld_ret"] = df_cont["gld_close"].pct_change()

    df_cont = pd.merge(df_cont, df_res, on="ds", how="inner")

    alloc = []
    rets = []

    for idx, row in df_cont.iterrows():
        # Conformal Execution: Expected edge must exceed statistical margin of error
        has_edge = row["pred_magnitude"] > row["conformal_buffer"]

        # Regime 0 (Safe) -> Can enter SPY if probability handles it, else CASH
        if row["regime"] == 0:
            if row["prob_up"] > 0.50 and has_edge:
                alloc.append("SPY")
                rets.append(row["spy_ret"])
            else:
                alloc.append("CASH")
                rets.append(0.0)

        # Regime 1 (Danger) -> Enforce Asymmetric Rotation
        else:
            if row["pred_magnitude"] < -row["conformal_buffer"]:
                # The model is confident in a severe crash. Defend via max inverse correlation.
                if (
                    row["spy_vustx_corr"] < row["spy_gld_corr"]
                    and row["spy_vustx_corr"] < -0.1
                ):
                    alloc.append("VUSTX")
                    rets.append(row["vustx_ret"])
                elif (
                    row["spy_gld_corr"] < row["spy_vustx_corr"]
                    and row["spy_gld_corr"] < -0.1
                ):
                    alloc.append("GLD")
                    rets.append(row["gld_ret"])
                else:
                    alloc.append("CASH")
                    rets.append(0.0)
            else:
                # Ambiguous Danger. Flat.
                alloc.append("CASH")
                rets.append(0.0)

    df_cont["allocation"] = alloc
    df_cont["strat_ret"] = rets

    cum_strat = (1 + df_cont["strat_ret"]).cumprod()
    strat_yield = cum_strat.iloc[-1]
    peak = cum_strat.expanding(min_periods=1).max()
    mdd = ((cum_strat / peak) - 1).min()

    cum_spy = (1 + df_cont["spy_ret"]).cumprod()
    spy_yield = cum_spy.iloc[-1]
    spy_peak = cum_spy.expanding(min_periods=1).max()
    spy_mdd = ((cum_spy / spy_peak) - 1).min()

    print("\n================================")
    print(" PHASE 10 GROUND TRUTH RESULTS")
    print("================================")
    print(f"SPY BUY & HOLD: {spy_yield:.2f}x (MDD: {spy_mdd:.2%})")
    print(f"PHASE 10 ENGINE: {strat_yield:.2f}x (MDD: {mdd:.2%})")


if __name__ == "__main__":
    build_phase10_engine()

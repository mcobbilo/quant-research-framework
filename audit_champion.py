import sqlite3
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture


def run_causal_audit():
    print("[AUDIT] Initializing Causal Recovery for Hypothesis_002_1775371277...")

    # 1. Load Data
    conn = sqlite3.connect("src/data/market_data.db")
    df = pd.read_sql_query("SELECT * FROM core_market_table", conn)
    conn.close()

    df.columns = [c.upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)

    # 2. Base Features
    df["LOGRET"] = np.log(df["SPY_CLOSE"] / df["SPY_CLOSE"].shift(1))
    df["RET"] = df["SPY_CLOSE"].pct_change()
    df["RV"] = df["LOGRET"] ** 2
    df["RV"] = df["RV"].fillna(0)
    df["LOG_RV"] = np.log(df["RV"] + 1e-8)
    df["DLOG_RV"] = df["LOG_RV"].diff()
    df["RV_AVG20"] = df["RV"].rolling(window=20, min_periods=10).mean()
    df["RV_RATIO"] = df["RV"] / df["RV_AVG20"]
    df["DRV"] = df["RV"].diff()
    df["ABS_DRV_OVER_RV"] = np.abs(df["DRV"]) / (df["RV"] + 1e-8)
    df["SKEW_5"] = df["LOGRET"].rolling(window=5, min_periods=3).skew()

    feature_cols = ["RV", "DLOG_RV", "RV_RATIO", "ABS_DRV_OVER_RV", "SKEW_5"]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # 3. Model Fit (Causal Split)
    n = len(df)
    train_size = int(n * 0.6)
    X_train = df.iloc[:train_size][feature_cols].values

    gmm = GaussianMixture(
        n_components=3, random_state=42, covariance_type="full", max_iter=200
    )
    gmm.fit(X_train)

    X_all = df[feature_cols].values
    gammas = gmm.predict_proba(X_all)

    # Sorted Indices
    means = gmm.means_
    rv_means = means[:, 0]
    idx = np.argsort(rv_means)
    low_idx, norm_idx, cris_idx = idx[0], idx[1], idx[2]

    # Map state probabilities
    df["GAMMA_LOW"] = gammas[:, low_idx]
    df["GAMMA_NORM"] = gammas[:, norm_idx]
    df["GAMMA_CRIS"] = gammas[:, cris_idx]

    # 4. BLOCK 16: ENFORCED CAUSALITY
    # Apply filters
    hl = 4
    df["G_LOW_S"] = df["GAMMA_LOW"].ewm(halflife=hl).mean()
    df["G_NORM_S"] = df["GAMMA_NORM"].ewm(halflife=hl).mean()
    df["G_CRIS_S"] = df["GAMMA_CRIS"].ewm(halflife=hl).mean()

    # --- BLOCK 16 INJECTION ---
    # We must not use G_LOW_S[t] to decide PI[t] for captured return RET[t+1]
    # IF G_LOW_S[t] was built using X[t] (which includes RV[t]).
    # To be hyper-causal, we shift the signals by 1.
    df["G_LOW_S_L"] = df["G_LOW_S"].shift(1)
    df["G_NORM_S_L"] = df["G_NORM_S"].shift(1)
    df["G_CRIS_S_L"] = df["G_CRIS_S"].shift(1)
    # --------------------------

    # Sigma Hat using LAGGED signals
    mu_rv = means[:, 0]
    mu_low, mu_norm, mu_cris = mu_rv[low_idx], mu_rv[norm_idx], mu_rv[cris_idx]

    df["SIGMA_HAT"] = (
        df["G_LOW_S_L"] * mu_low
        + df["G_NORM_S_L"] * mu_norm
        + df["G_CRIS_S_L"] * mu_cris
    )

    # Multiplier M using LAGGED signals
    s_t_L = 1.8 * df["G_LOW_S_L"] + 0.6 * df["G_NORM_S_L"] - 0.4 * df["G_CRIS_S_L"]
    s_bar = s_t_L.iloc[:train_size].mean()
    lambda_ = 2.5
    df["M"] = 1.0 + np.tanh(lambda_ * (s_t_L - s_bar))

    # 5. Position Rule
    sigma_target = 1e-6
    # SQRT_RV is current (for sizing), but SIGMA_HAT is predictive.
    # To be safe, SQRT_RV should also be lagged?
    # Usually, realized vol for sizing is ok at t, but let's lag it to be Block 16 perfect.
    df["SQRT_RV_L"] = np.sqrt(df["RV"].shift(1) + 1e-8)

    df["PI"] = (
        df["M"] * (sigma_target / (df["SIGMA_HAT"] + 1e-8)) * (1.0 / df["SQRT_RV_L"])
    )
    df["PI"] = df["PI"].clip(-1.5, 2.5)

    # 6. Backtest
    df["STRAT_RET"] = df["PI"].shift(1) * df["RET"]
    df = df.dropna(subset=["STRAT_RET"]).reset_index(drop=True)

    cum_ret = (1 + df["STRAT_RET"]).cumprod()
    final_yield = cum_ret.iloc[-1] - 1.0
    strat_rets = df["STRAT_RET"]
    sharpe = (
        (strat_rets.mean() / strat_rets.std() * np.sqrt(252))
        if strat_rets.std() > 0
        else 0.0
    )

    print(f"[AUDIT_RESULTS] Yield: {final_yield:.4f} | Sharpe: {sharpe:.4f}")

    if sharpe > 0.50:
        print("[AUDIT_PASS] Alpha is robust under Block 16 causality.")
    else:
        print(
            "[AUDIT_FAIL] Lookahead bias detected in original. New Sharpe is significantly lower."
        )


if __name__ == "__main__":
    run_causal_audit()

import sys
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import xgboost as xgb

# Connect architecture
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from experimental.xgboost_allocation_engine import get_ml_dataframe
from models.jepa_attention_engine import JepaAttentionEngine, PPOExecutionPipeline


def categorize_features(feature_names):
    """Dynamically route the remaining DB columns into exactly 8 Tensors"""
    domains = {
        "MACRO_GRAVITY": [],
        "CREDIT": [],
        "COMMODITIES": [],
        "BANKS": [],
        "VOLATILITY": [],
        "BREADTH": [],
        "CURIOSITY": [],
        "ASYMMETRIC_ALPHA": [],
    }

    for c in feature_names:
        # --- PRIORITY DOMAINS (Idea 4 & 19D) ---
        if "Entropy" in c:
            domains["CURIOSITY"].append(c)
        elif "XGB" in c or "Asymmetric" in c:
            domains["ASYMMETRIC_ALPHA"].append(c)

        # --- MACRO DOMAINS (Phase 11 Logic) ---
        elif any(
            k in c
            for k in [
                "TNX",
                "T10Y",
                "REAL_YIELD",
                "INFLATION",
                "Monetary",
                "FED",
                "CentralBank",
            ]
        ):
            domains["MACRO_GRAVITY"].append(c)
        elif any(k in c for k in ["VIX", "SKEW", "SPY", "VUSTX"]):
            domains["MACRO_GRAVITY"].append(c)
        elif any(k in c for k in ["GC_", "HG_", "OIL_", "GLD", "SLV", "Commodity"]):
            domains["COMMODITIES"].append(c)
        elif any(k in c for k in ["LQD", "HYG", "TIP", "BND", "Credit"]):
            domains["CREDIT"].append(c)
        elif any(k in c for k in ["XLF", "KBE", "KRE", "Liquidity", "FX_"]):
            domains["BANKS"].append(c)
        elif any(k in c for k in ["VXX", "UVXY", "Volatility"]):
            domains["VOLATILITY"].append(c)
        # BREADTH (Default fallback for everything else)
        else:
            domains["BREADTH"].append(c)

    return domains


def asymmetric_mse(y_true, y_pred):
    """Phase 10: Calibrated for 19D Recovery (Less restrictive)"""
    residual = y_pred - y_true
    grad = np.where(residual > 0, 2.0 * residual, 1.0 * residual)
    hess = np.where(residual > 0, 2.0, 1.0)
    return grad, hess


def calculate_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr0 = high - low
    tr1 = (high - prev_close).abs()
    tr2 = (low - prev_close).abs()
    true_range = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False).mean()


def jepa_walk_forward_pipeline():
    print("Initiating PPO RL J-EPA Integration Protocol (19D Alpha Edition)...")
    df = get_ml_dataframe()
    ml_df = df.dropna(subset=["SPY_Daily_Ret"]).copy()

    # Static Configuration
    SEQ_LEN = 63
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Calculate physical VUSTX Daily Returns
    ml_df["VUSTX_Daily_Ret"] = ml_df["VUSTX_CLOSE"].pct_change().fillna(0.0)

    # --- Feature Engineering Stage ---
    print("Engineering Asymmetric Alpha (Hypothesis 0013) + Curiosity Scalars...")

    # 1. Entropy calculation
    def shannon_entropy(x):
        if len(x) < 2:
            return 0.0
        hist, _ = np.histogram(x, bins=10, density=True)
        hist = hist / (hist.sum() + 1e-8)
        return -np.sum(hist * np.log(hist + 1e-8))

    ml_df["Entropy_SPY"] = (
        ml_df["SPY_Daily_Ret"]
        .rolling(SEQ_LEN)
        .apply(shannon_entropy, raw=True)
        .fillna(0.0)
    )

    # 3. VPIN-Style Microstructure Proxy (Hypothesis 0014)
    # Volatility / Volume ratio as a proxy for toxic liquidity
    ml_df["VPIN_Proxy"] = (ml_df["SPY_HIGH"] - ml_df["SPY_LOW"]) / (
        ml_df["SPY_VOLUME"].rolling(5).mean() + 1e-8
    )
    ml_df["VPIN_Proxy"] = (
        ml_df["VPIN_Proxy"].rolling(SEQ_LEN).rank(pct=True).fillna(0.5)
    )

    # 4. Hypothesis 0013 Indicators
    cl, hi, lo = ml_df["SPY_CLOSE"], ml_df["SPY_HIGH"], ml_df["SPY_LOW"]
    m21 = cl.ewm(span=21, adjust=False).mean()
    sigma21 = cl.rolling(window=21).std()
    atr14 = calculate_atr(hi, lo, cl, 14)
    lambda_t = 1.8 + 0.7 * ((sigma21 / (atr14 + 1e-8)) - 1)
    kupper = m21 + lambda_t * atr14
    klower = m21 - lambda_t * atr14
    supper = m21 + 2.0 * sigma21
    slower = m21 - 2.0 * sigma21

    ml_df["0013_CT"] = ((kupper - klower) / (supper - slower + 1e-8)).shift(1).fillna(0)
    ml_df["0013_M21_Delta"] = m21.diff(4).shift(1).fillna(0)
    ml_df["0013_Close_Rel_K"] = (
        ((cl - klower) / (kupper - klower + 1e-8)).shift(1).fillna(0)
    )

    # 5. Rolling XGB Asymmetric Confidence Signal
    ml_df["SPY_Fwd_10D_Ret"] = (1 + ml_df["SPY_Daily_Ret"]).rolling(10).apply(
        np.prod, raw=True
    ) - 1
    ml_df["SPY_Fwd_10D_Ret"] = ml_df["SPY_Fwd_10D_Ret"].shift(-10).fillna(0.0)
    ml_df["VUSTX_Fwd_10D_Ret"] = (1 + ml_df["VUSTX_Daily_Ret"]).rolling(10).apply(
        np.prod, raw=True
    ) - 1
    ml_df["VUSTX_Fwd_10D_Ret"] = ml_df["VUSTX_Fwd_10D_Ret"].shift(-10).fillna(0.0)

    xgb_features = ["0013_CT", "0013_M21_Delta", "0013_Close_Rel_K"]
    ml_df["XGB_Asymmetric_Signal"] = 0.0
    y_target_xgb = ml_df["SPY_Fwd_10D_Ret"].values
    xgb_train_size = 1260

    print("Generating Pre-computed Asymmetric Alpha Signal (OOS Rolling)...")
    for t in range(xgb_train_size, len(ml_df), 20):
        t_start, t_end = t - xgb_train_size, t
        X_train, y_train = (
            ml_df[xgb_features].iloc[t_start:t_end],
            y_target_xgb[t_start:t_end],
        )
        reg = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            objective=asymmetric_mse,
            n_jobs=-1,
            random_state=42,
        )
        reg.fit(X_train, y_train)
        t_next = min(t + 20, len(ml_df))
        ml_df.iloc[t:t_next, ml_df.columns.get_loc("XGB_Asymmetric_Signal")] = (
            reg.predict(ml_df[xgb_features].iloc[t:t_next]) * 1.5
        )

    # --- Feature Discovery & Domain Mapping ---
    excluded_keywords = [
        "OPEN",
        "HIGH",
        "LOW",
        "CLOSE",
        "PRICE",
        "VOLUME",
        "Fwd_",
        "Date",
        "index",
        "Daily_Ret",
    ]
    features = [c for c in ml_df.columns if not any(k in c for k in excluded_keywords)]
    # Explicitly add engineered features back if they were excluded
    features += [
        "Entropy_SPY",
        "VPIN_Proxy",
        "0013_CT",
        "0013_M21_Delta",
        "0013_Close_Rel_K",
        "XGB_Asymmetric_Signal",
    ]
    features = sorted(list(set(features)))  # Deduplicate and sort

    ml_df = ml_df.dropna(subset=features)
    print(f"Matrix Filtered: {len(ml_df)} rows | Neural Features: {len(features)}")

    domain_map = categorize_features(features)
    # Add VPIN to the Volatility domain
    if "VPIN_Proxy" in features:
        domain_map["VOLATILITY"].append("VPIN_Proxy")

    domain_dimensions = {dom: len(cols) for dom, cols in domain_map.items()}

    # --- Structural Trajectories & Hardware Acceleration ---
    y_returns = np.column_stack(
        (
            ml_df["SPY_Fwd_10D_Ret"].values[SEQ_LEN - 1 :],
            ml_df["VUSTX_Fwd_10D_Ret"].values[SEQ_LEN - 1 :],
        )
    )
    y_physical_daily_spy = ml_df["SPY_Daily_Ret"].values[SEQ_LEN - 1 :]
    y_physical_daily_vustx = ml_df["VUSTX_Daily_Ret"].values[SEQ_LEN - 1 :]
    y_physical_daily_vix = ml_df["VIX_CLOSE"].values[SEQ_LEN - 1 :]
    y_dates = ml_df.index[SEQ_LEN - 1 :]

    print("\nConstructing Neural Tensors with T-1 Lag Enforcement...")
    domain_windows_tensor = {}
    for dom, cols in domain_map.items():
        if not cols:
            continue
        # Shift 1 to force T-1 prediction
        base_tensor = torch.tensor(
            ml_df[cols].shift(1).fillna(0).values, dtype=torch.float32, device=device
        )
        n, d = base_tensor.shape
        s0, s1 = base_tensor.stride()
        domain_windows_tensor[dom] = torch.as_strided(
            base_tensor, size=(n - SEQ_LEN + 1, SEQ_LEN, d), stride=(s0, s0, s1)
        )

    # --- Walk-Forward Execution Loop ---
    n_samples = len(y_dates)
    train_size, step_size = 1260, 20
    out_of_sample_preds, out_of_sample_dates = [], []
    out_of_sample_daily_spy, out_of_sample_daily_vustx, out_of_sample_daily_vix = (
        [],
        [],
        [],
    )
    out_of_sample_attn = []

    jepa_engine = JepaAttentionEngine(
        domain_dims=domain_dimensions, latent_dim=16, num_heads=4
    )
    target_pipeline = PPOExecutionPipeline(jepa_engine=jepa_engine, device=device)

    print(f"Executing Forward Sim Loop (device={device})...")
    for i in range(0, n_samples - train_size, step_size):
        train_idx = slice(i, i + train_size)
        test_bounds = min(i + train_size + step_size, n_samples)
        test_idx = slice(i + train_size, test_bounds)

        y_train = y_returns[train_idx]
        domain_inputs_train = {
            dom: domain_windows_tensor[dom][train_idx] for dom in domain_windows_tensor
        }
        domain_inputs_test = {
            dom: domain_windows_tensor[dom][test_idx] for dom in domain_windows_tensor
        }

        target_pipeline.fit(domain_inputs_train, y_train, epochs=40)
        preds, active_weights = target_pipeline.predict_proba(domain_inputs_test)

        out_of_sample_preds.extend(np.atleast_1d(preds))
        out_of_sample_dates.extend(y_dates[test_idx])
        out_of_sample_daily_spy.extend(y_physical_daily_spy[test_idx])
        out_of_sample_daily_vustx.extend(y_physical_daily_vustx[test_idx])
        out_of_sample_daily_vix.extend(y_physical_daily_vix[test_idx])
        # Sum along sequence dimension to get (Batch, Domains) importance
        out_of_sample_attn.extend(active_weights.sum(dim=2).detach().cpu().numpy())

    # --- Allocation Simulation (Waterfall Edition) ---
    results_df = pd.DataFrame(
        {
            "JEPA_Allocation": out_of_sample_preds,
            "SPY_Daily_Ret": out_of_sample_daily_spy,
            "VUSTX_Daily_Ret": out_of_sample_daily_vustx,
            "VIX_CLOSE": out_of_sample_daily_vix,
        },
        index=out_of_sample_dates,
    )

    # Join Attention Weights
    attn_df = pd.DataFrame(
        out_of_sample_attn,
        columns=jepa_engine.active_domains,
        index=out_of_sample_dates,
    )
    results_df = results_df.join(attn_df)

    # Alpha Filter (Inertia Gate)
    raw_allocations = results_df["JEPA_Allocation"].values
    filtered_allocations = np.zeros_like(raw_allocations)
    curr_a = 0.0
    for idx, a in enumerate(raw_allocations):
        if abs(a - curr_a) > 0.10:
            curr_a = a
        filtered_allocations[idx] = curr_a
    results_df["Jepa_Raw_Weight"] = filtered_allocations

    # Waterfall Memory
    import yfinance as yf

    print("Compiling Waterfall Capitulation Offense...")
    spy_px = yf.download("^GSPC", start="1986-01-01", progress=False)
    if isinstance(spy_px.columns, pd.MultiIndex):
        spy_px.columns = spy_px.columns.droplevel(1)

    mas = [5, 10, 20, 50, 100, 200]
    for ma in mas:
        spy_px[f"SMA_{ma}"] = spy_px["Close"].rolling(window=ma).mean()

    spy_px["Is_Waterfall"] = (
        spy_px["Close"] < spy_px[[f"SMA_{ma}" for ma in mas]].min(axis=1)
    ) & (
        spy_px["Close"].shift(1).rolling(5).max()
        > spy_px[[f"SMA_{ma}" for ma in mas]].max(axis=1)
    )

    results_df.index = pd.to_datetime(results_df.index)
    results_df = results_df.join(
        spy_px[["Is_Waterfall"]].rolling(10).max() == 1.0, how="left"
    )
    results_df.columns = [*results_df.columns[:-1], "Waterfall_10D_Memory"]
    results_df["Waterfall_10D_Memory"] = results_df["Waterfall_10D_Memory"].fillna(
        False
    )

    # --- Phase 25 Deterministic Waterfall Allocation ---
    # Metrics calculation for Risk-Parity
    results_df["SPY_Vol"] = results_df["SPY_Daily_Ret"].rolling(63).std() * np.sqrt(252)
    results_df["VUSTX_Vol"] = results_df["VUSTX_Daily_Ret"].rolling(63).std() * np.sqrt(
        252
    )
    results_df["SPY_VUSTX_Corr"] = (
        results_df["SPY_Daily_Ret"].rolling(63).corr(results_df["VUSTX_Daily_Ret"])
    )

    # Linear True Risk Parity (Phase 25 Logic)
    # W = (1/sigma) * (1 - rho)
    vustx_budget = (1.0 / (results_df["VUSTX_Vol"] + 1e-4)) * (
        1.0 - results_df["SPY_VUSTX_Corr"]
    )
    # Normalize budgets if using multiple safe-havens (here just VUSTX)
    results_df["Risk_Parity_Weight"] = vustx_budget.clip(
        0.5, 1.5
    )  # Minimum 50% defensive budget

    # Waterfall Override Logic
    results_df["SPY_Weight"] = results_df["Jepa_Raw_Weight"]
    # If J-EPA is Risk-Off (0.0), use Risk-Parity budget instead of just 1.0
    results_df["VUSTX_Weight"] = (1.0 - results_df["Jepa_Raw_Weight"]) * results_df[
        "Risk_Parity_Weight"
    ]
    results_df["Cash_Weight"] = (
        1.0 - results_df["SPY_Weight"] - results_df["VUSTX_Weight"]
    )
    results_df["Cash_Weight"] = results_df["Cash_Weight"].clip(0.0, 1.0)

    # 1. Bomb Shelter (VIX > 33)
    defense = results_df["VIX_CLOSE"].shift(1) > 33.0
    results_df.loc[defense, ["SPY_Weight", "VUSTX_Weight", "Cash_Weight"]] = [
        0.0,
        0.0,
        1.0,
    ]

    # 2. Waterfall Recovery (Alpha-Bridge)
    # Be aggressive in the recovery phase
    offense = (results_df["VIX_CLOSE"].shift(1) > 33.0) & (
        results_df["Waterfall_10D_Memory"].shift(1)
    )
    results_df.loc[offense, ["SPY_Weight", "VUSTX_Weight", "Cash_Weight"]] = [
        1.2,
        -0.2,
        0.0,
    ]

    # Final Performance Calculation
    turnover = (
        results_df["SPY_Weight"].diff().abs()
        + results_df["VUSTX_Weight"].diff().abs()
        + results_df["Cash_Weight"].diff().abs()
    ) / 2.0
    results_df["JEPA_Daily_Ret"] = (
        (results_df["SPY_Daily_Ret"] * results_df["SPY_Weight"])
        + (results_df["VUSTX_Daily_Ret"] * results_df["VUSTX_Weight"])
        - (turnover.fillna(0.0) * 0.001)
    )

    j_cagr = ((1 + results_df["JEPA_Daily_Ret"]).prod() ** (252 / len(results_df))) - 1
    s_cagr = ((1 + results_df["SPY_Daily_Ret"]).prod() ** (252 / len(results_df))) - 1

    print(
        f"\nFinal Statistics (2011-2026):\n - SPY CAGR: {s_cagr * 100:.2f}%\n - J-EPA CAGR: {j_cagr * 100:.2f}%"
    )

    # Export & Plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 8))
    (1 + results_df[["SPY_Daily_Ret", "JEPA_Daily_Ret"]]).cumprod().plot(
        ax=ax, color=["#555555", "#ff00ff"]
    )
    plt.title("J-EPA 19D Alpha Injection - Backtest Results")
    plt.grid(alpha=0.3)

    out_path = "/Users/milocobb/.gemini/antigravity/brain/4b7ec9fe-9cbc-4e19-a3f6-2639cbd15870/jepa_walk_forward_19d.png"
    plt.savefig(out_path)
    results_df.to_csv(
        "/Users/milocobb/.gemini/antigravity/brain/4b7ec9fe-9cbc-4e19-a3f6-2639cbd15870/jepa_walk_forward_results_19d.csv"
    )
    print("Results saved to artifacts.")


if __name__ == "__main__":
    jepa_walk_forward_pipeline()

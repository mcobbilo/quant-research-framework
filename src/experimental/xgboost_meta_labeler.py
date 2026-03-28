import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import scipy.stats as stats
import shap
import warnings
warnings.filterwarnings('ignore')

def download_orthogonal_features(penalty_weight=49.0):
    import sqlite3
    import os
    print(f"\n[Meta-Labeler] Connecting to Phase 123 SQLite Infrastructure (Penalty Weight: {penalty_weight}x)...")
    
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'market_data.db')
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT Date, SPY_CLOSE, VIX_CLOSE, TLT_CLOSE, BAMLC0A0CM, GC_CLOSE,
           CVR3_BUY_SIGNAL, CVR3_SELL_SIGNAL, VIX_BB_WIDTH, VIX_DIST_UPPER, VIX_DIST_LOWER,
           VIX_MOVE_SPREAD_10D, Credit_Acceleration_30D, TED_Acceleration_30D, SPY_RSP_MOMENTUM_60D,
           Global_Liquidity_Velocity_21d, Fed_Liquidity_Surprise,
           FX_DXY_Velocity_20d, FX_Yen_Shock_5d
    FROM core_market_table
    WHERE Date >= '2005-01-01'
    """
    df = pd.read_sql(query, conn, index_col='Date')
    conn.close()
    
    df.index = pd.to_datetime(df.index)
    
    df["SPY"] = df["SPY_CLOSE"]
    df["VIX"] = df["VIX_CLOSE"]
    
    # 1. Volatility Regime
    df["VIX_Spike"] = df["VIX"].pct_change(3)
    
    # 2. Flight to Safety Velocity
    df["TLT_Momentum"] = df["TLT_CLOSE"].pct_change(10)
    
    # 3. High Yield Corporate Stress Spread (BAMLC0A0CM spreads)
    df["Credit_Velocity"] = df["BAMLC0A0CM"].pct_change(5)
    
    # 4. Gold Safe Haven vs Equity
    df["Safe_Haven_Ratio"] = df["GC_CLOSE"] / df["SPY_CLOSE"]
    
    # [NEW] Phase 125 Tri-Modal Safe Haven Parameters
    df["TLT"] = df["TLT_CLOSE"]
    df["GLD"] = df["GC_CLOSE"]
    df["TLT_SMA200"] = df["TLT"].rolling(200).mean()
    df["GLD_SMA200"] = df["GLD"].rolling(200).mean()
    
    spy_log = np.log(df["SPY"] / df["SPY"].shift(1))
    tlt_log = np.log(df["TLT"] / df["TLT"].shift(1))
    gld_log = np.log(df["GLD"] / df["GLD"].shift(1))
    
    df["CORR_SPY_TLT_63"] = spy_log.rolling(63).corr(tlt_log)
    df["CORR_SPY_GLD_63"] = spy_log.rolling(63).corr(gld_log)
    
    df["TLT_Mom_63"] = df["TLT"].pct_change(63)
    df["GLD_Mom_63"] = df["GLD"].pct_change(63)
    
    # =========================================================================
    # PHASE 119: HEAVY-TAIL TRANSFORMATION PIPELINE
    # Mapping raw velocity and spread arrays into structural Inverse-Normal domains
    # by parsing them through a Student-T (df=3) cumulative density function.
    # =========================================================================
    print("[Meta-Labeler] Executing Phase 119 Heavy-Tail transformations...")
    
    for col in ["VIX_Spike", "TLT_Momentum", "Credit_Velocity", "Safe_Haven_Ratio", "VIX_MOVE_SPREAD_10D"]:
        roll_mean = df[col].rolling(252).mean().shift(1)
        roll_std = df[col].rolling(252).std().shift(1)
        
        # Raw theoretical Gaussian Z-Score
        raw_z = (df[col] - roll_mean) / roll_std
        
        # Fat-Tail Probability -> Inverse Gaussian Re-mapping
        df[f"HT_{col}"] = stats.norm.ppf(stats.t.cdf(raw_z, df=3))
    
    # =========================================================================
    # PHASE 115: THE BOOLEAN 89-MATRIX INSIGHT FLAGS
    # =========================================================================
    df["Insight_VIX_Shock"] = (df["VIX_Spike"] > 0.20).astype(int)
    df["Insight_Credit_Trap"] = (df["Credit_Velocity"] > 0.03).astype(int)
    
    # Phase 126: VIX vs MOVE Cross-Asset Volatility Divergence
    df["Insight_Credit_Fracture"] = (df["VIX_MOVE_SPREAD_10D"] > 0.15).astype(int)
    
    # Phase 128: Global Macro Acceleration (Jerk)
    df["Insight_Credit_Acceleration"] = (df["Credit_Acceleration_30D"] > 0).astype(int)
    df["Insight_TED_Acceleration"] = (df["TED_Acceleration_30D"] > 0).astype(int)
    
    # Phase 129: Market Breadth Divergence (Hollow Rallies)
    df["Insight_Breadth_Divergence"] = (df["SPY_RSP_MOMENTUM_60D"] > 0.05).astype(int)
    
    # Phase 130: Principal Component Analysis (Global Liquidity Velocity)
    df["Insight_Liquidity_Contraction"] = (df["Global_Liquidity_Velocity_21d"] < 0).astype(int)
    
    # Phase 131: The Liquidity Surprise Indicator
    df["Insight_Fed_Tightening"] = (df["Fed_Liquidity_Surprise"] < 0).astype(int)
    
    # Phase 134: Global FX Shock Absorbers
    df["Insight_FX_Shock"] = ((df["FX_DXY_Velocity_20d"] > 3.0) | (df["FX_Yen_Shock_5d"] < -4.0)).astype(int)
    
    _safe_haven_sma = df["Safe_Haven_Ratio"].rolling(50).mean()
    df["Insight_Safe_Haven_Flight"] = (df["Safe_Haven_Ratio"] > _safe_haven_sma * 1.05).astype(int)
    
    _spy_sma20 = df["SPY"].rolling(20).mean()
    df["Insight_SPY_Oversold_Flush"] = (df["SPY"] < _spy_sma20 * 0.95).astype(int)
    
    df["Insight_TLT_Capitulation"] = (df["TLT_Momentum"] < -0.05).astype(int)
    # =========================================================================
    
    # Target: BLACK SWAN RADAR
    # We are not predicting upside. We are explicitly training the model to detect 
    # imminent structural drawdowns (-5% drop in the next 15 days).
    future_ret = df["SPY"].shift(-15) / df["SPY"] - 1.0
    df["Target_Crash"] = (future_ret < -0.05).astype(int)
    
    # [REMOVED] Artifical DataFrame Weighting. Machine will now use Asymmetric Calculus.
    df.dropna(inplace=True)
    return df

def backtest_meta_labeler(penalty_weight=49.0):
    df = download_orthogonal_features(penalty_weight)
    
    train_size = 5 * 252 # 5 year training memory
    purge_period = 30  # 30 day strict CPCV firewall
    
    # Phase 132: Boruta "Shadow Noise" Execution
    # We strip outcome signals and base prices to avoid data leakage
    drop_cols = ["Target_Crash", "SPY", "SPY_CLOSE", "TLT_CLOSE", "GC_CLOSE", "BAMLC0A0CM", "VIX_CLOSE"]
    raw_features = [c for c in df.columns if c not in drop_cols and c != "Date"]
    
    # Extract the first 5-Year Train Block to discover universal structural truths cleanly
    discovery_index = train_size
    X_discovery = df.iloc[:discovery_index][raw_features]
    y_discovery = df.iloc[:discovery_index]["Target_Crash"]
    
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.pruning import run_boruta_pruning
    
    features = run_boruta_pruning(X_discovery, y_discovery)
    
    # Phase 133: The GMM Macro Router
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    
    print("\n[Phase 133] Initializing Unsupervised Tri-Regime GMM Router...")
    gmm_feats = ["BAMLC0A0CM", "VIX_BB_WIDTH", "SPY_RSP_MOMENTUM_60D", "Global_Liquidity_Velocity_21d"]
    scaler = StandardScaler()
    
    # Mathematically isolate the router to only learn from the 2005-2010 discovery set 
    X_discovery_gmm = df.iloc[:discovery_index][gmm_feats].fillna(0)
    scaler.fit(X_discovery_gmm)
    
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(scaler.transform(X_discovery_gmm))
    
    # Physically apply the router forward through time
    df["Macro_Regime"] = gmm.predict(scaler.transform(df[gmm_feats].fillna(0)))
    print(f"[Phase 133] 15-Year Market Regimes Segmented:\n{df['Macro_Regime'].value_counts()}\n")
    
    # Phase 127: Asymmetric Objective Injection
    # We strip 'objective': 'binary:logistic' from params to output raw logits.
    params = {
        'max_depth': 4,
        'learning_rate': 0.05,
        'eval_metric': 'aucpr', # Area Under the Precision-Recall Curve (optimal for heavily imbalanced rare events)
        'n_jobs': -1,
        'random_state': 42
    }
    
    def asymmetric_logloss(preds, dtrain):
        labels = dtrain.get_label()
        
        # XGBoost outputs raw margin logits when using custom obj. Transform to probability via sigmoid.
        probs = 1.0 / (1.0 + np.exp(-preds))
        
        # Base Log-Loss Gradient & Hessian
        grad = probs - labels
        hess = probs * (1.0 - probs)
        
        # Exponential Asymmetry Penalty
        # If label is 1 (Crash), mathematically hyper-penalize false negatives.
        grad = np.where(labels == 1, grad * penalty_weight, grad)
        hess = np.where(labels == 1, hess * penalty_weight, hess)
        
        return grad, hess
    
    capital = 100000.0
    shares_spy = 0.0
    shares_tlt = 0.0
    shares_gld = 0.0
    
    dates = []
    equity_curve = []
    spy_curve = []
    spy_allocs = []
    tlt_allocs = []
    gld_allocs = []
    cash_allocs = []
    
    print("\n[Meta-Labeler] Initializing Purged Black-Swan Detection (CPCV)...")
    
    for i in range(train_size + purge_period, len(df), 60):  
        train_end_idx = i - purge_period
        train_data = df.iloc[train_end_idx - train_size : train_end_idx]
        
        test_end_idx = min(i + 60, len(df))
        test_data = df.iloc[i : test_end_idx]
        
        # Phase 133: Tri-Model Segregation
        models = {}
        explainers = {}
        for r in range(3):
            train_r = train_data[train_data["Macro_Regime"] == r]
            if len(train_r) < 30:
                # If regime is historically rare, fall back to global memory
                dtrain_r = xgb.DMatrix(train_data[features], label=train_data["Target_Crash"])
            else:
                dtrain_r = xgb.DMatrix(train_r[features], label=train_r["Target_Crash"])
                
            model_r = xgb.train(params, dtrain_r, num_boost_round=30, obj=asymmetric_logloss)
            models[r] = model_r
            explainers[r] = shap.TreeExplainer(model_r)
        
        prob_crash = np.zeros(len(test_data))
        
        for r in range(3):
            mask = test_data["Macro_Regime"] == r
            if mask.any():
                dtest_r = xgb.DMatrix(test_data.loc[mask, features])
                # Phase 127 custom logit translation -> probability
                raw_logits_r = models[r].predict(dtest_r)
                prob_crash[mask] = 1.0 / (1.0 + np.exp(-raw_logits_r))
        
        for j in range(len(test_data)):
            today = test_data.iloc[j]
            prob = prob_crash[j]
            price_spy = today["SPY"]
            price_tlt = today["TLT"]
            price_gld = today["GLD"]
            
            # MTM Valuation
            portfolio_value = capital + (shares_spy * price_spy) + (shares_tlt * price_tlt) + (shares_gld * price_gld)
            
            target_spy = 0.0
            target_tlt = 0.0
            target_gld = 0.0
            target_cash = 0.0
            
            # Phase 125: Tri-Modal Fractional Allocation routing
            if prob > 0.80: # 🚨 ABSOLUTE PANIC TRIGGER 🚨
                # Tri-Modal Defense Screener Execution
                tlt_valid = (today["TLT"] > today["TLT_SMA200"]) and (today["CORR_SPY_TLT_63"] < 0.0)
                gld_valid = (today["GLD"] > today["GLD_SMA200"]) and (today["CORR_SPY_GLD_63"] < 0.0)
                
                if tlt_valid and gld_valid:
                    if today["TLT_Mom_63"] > today["GLD_Mom_63"]:
                        target_tlt = portfolio_value
                    else:
                        target_gld = portfolio_value
                elif tlt_valid:
                    target_tlt = portfolio_value
                elif gld_valid:
                    target_gld = portfolio_value
                else:
                    target_cash = portfolio_value # Total system failure, retreat to 0.0 yield
                    
            elif prob > 0.45: # ⚠️ ELEVATED RISK ⚠️
                target_spy = portfolio_value * 0.5
                target_cash = portfolio_value * 0.5
            else: # 🟢 REGIME STABLE 🟢
                target_spy = portfolio_value * 1.0
                
            # Delta Rebalancing
            delta_spy = target_spy - (shares_spy * price_spy)
            delta_tlt = target_tlt - (shares_tlt * price_tlt)
            delta_gld = target_gld - (shares_gld * price_gld)
            
            rebalance_threshold = portfolio_value * 0.05
            
            if abs(delta_spy) > rebalance_threshold or abs(delta_tlt) > rebalance_threshold or abs(delta_gld) > rebalance_threshold:
                # Liquidations first
                if delta_spy < 0:
                    capital -= delta_spy 
                    shares_spy += (delta_spy / price_spy)
                    capital -= abs(delta_spy) * 0.0002
                if delta_tlt < 0:
                    capital -= delta_tlt
                    shares_tlt += (delta_tlt / price_tlt)
                    capital -= abs(delta_tlt) * 0.0002
                if delta_gld < 0:
                    capital -= delta_gld
                    shares_gld += (delta_gld / price_gld)
                    capital -= abs(delta_gld) * 0.0002
                    
                # Purchases next
                if delta_spy > 0:
                    shares_spy += (delta_spy / price_spy)
                    capital -= delta_spy
                    capital -= abs(delta_spy) * 0.0002
                if delta_tlt > 0:
                    shares_tlt += (delta_tlt / price_tlt)
                    capital -= delta_tlt
                    capital -= abs(delta_tlt) * 0.0002
                if delta_gld > 0:
                    shares_gld += (delta_gld / price_gld)
                    capital -= delta_gld
                    capital -= abs(delta_gld) * 0.0002
                
                # --- PHASE 120 FORENSIC SHAP DIAGNOSTIC EXTRACTION ---
                r = int(test_data["Macro_Regime"].iloc[j])
                explainer_r = explainers[r]
                row = test_data[features].iloc[[j]]
                shap_vals = explainer_r(row)
                vals = np.abs(shap_vals.values[0])
                top_indices = np.argsort(vals)[-3:][::-1]
                top_features = test_data[features].columns[top_indices].tolist()
                
                if target_spy == portfolio_value:
                    shift_type = "ROTATION -> 100% SPY (Risk-On)"
                elif target_tlt == portfolio_value:
                    shift_type = "ROTATION -> 100% TLT (Tri-Modal Treasury Defense)"
                elif target_gld == portfolio_value:
                    shift_type = "ROTATION -> 100% GLD (Tri-Modal Gold Defense)"
                elif target_cash == portfolio_value:
                    shift_type = "ROTATION -> 100% CASH (Total System Panic)"
                else:
                    shift_type = "ROTATION -> 50/50 SPY/CASH (Elevated De-Risking)"
                
                msg = f"\n=================================================="
                msg += f"\n| FORENSIC SHAP TRIGGER [{today.name.strftime('%Y-%m-%d')}]"
                msg += f"\n| SHIFT: {shift_type}"
                msg += f"\n| SYSTEMIC CRASH PROBABILITY: {prob:.1f}% (Target Matrix Threshold: 80.0%)"
                msg += f"\n| ALGO RATIONALE: Top 3 Absolute SHAP Feature Weights"
                
                for rank, feat in enumerate(top_features):
                    val = row[feat].iloc[0]
                    shap_w = shap_vals.values[0][top_indices[rank]]
                    msg += f"\n|  {rank+1}. {feat:<28} (Value: {val:>+6.2f}) | SHAP Impact: {shap_w:>+7.3f}"
                
                msg += f"\n=================================================="
                print(msg)
                # -----------------------------------------------------
            
            mtm = capital + (shares_spy * price_spy) + (shares_tlt * price_tlt) + (shares_gld * price_gld)
            
            dates.append(today.name)
            equity_curve.append(mtm)
            spy_curve.append(price_spy)
            spy_allocs.append((shares_spy * price_spy) / mtm * 100)
            tlt_allocs.append((shares_tlt * price_tlt) / mtm * 100)
            gld_allocs.append((shares_gld * price_gld) / mtm * 100)
            cash_allocs.append((capital / mtm) * 100)
            
    # Compile the final data matrices into a time-series
    results = pd.DataFrame({
        "Date": dates, 
        "Radar_Nav": equity_curve, 
        "SPY_Price": spy_curve,
        "SPY_Alloc": spy_allocs,
        "TLT_Alloc": tlt_allocs,
        "GLD_Alloc": gld_allocs,
        "Cash_Alloc": cash_allocs
    })
    results["Year"] = results["Date"].dt.year
    
    # Year-by-Year Aggregation
    yearly_metrics = []
    
    for year, group in results.groupby("Year"):
        meta_ret = ((group["Radar_Nav"].iloc[-1] / group["Radar_Nav"].iloc[0]) - 1.0) * 100
        spy_ret = ((group["SPY_Price"].iloc[-1] / group["SPY_Price"].iloc[0]) - 1.0) * 100
        
        alpha = meta_ret - spy_ret
        yearly_metrics.append((year, meta_ret, spy_ret, alpha))
        
    print("\n=================== 15-YEAR ALGORITHMIC TEAR SHEET ====================")
    print(f"{'Year':<6} | {'Meta-Labeler (%)':<18} | {'S&P 500 (%)':<16} | {'Alpha (%)':<10}")
    print("-" * 65)
    for year, meta, spy, alpha in yearly_metrics:
        # Formatting for clear visual display
        meta_str = f"{meta:+.2f}%"
        spy_str = f"{spy:+.2f}%"
        alpha_str = f"{alpha:+.2f}%"
        print(f"{year:<6} | {meta_str:<18} | {spy_str:<16} | {alpha_str:<10}")
    
    print("-" * 65)
    
    # Absolute Totals
    meta_total = ((results["Radar_Nav"].iloc[-1] / results["Radar_Nav"].iloc[0]) - 1.0) * 100
    spy_total = ((results["SPY_Price"].iloc[-1] / results["SPY_Price"].iloc[0]) - 1.0) * 100
    
    # Max DD calculations
    rolling_max_meta = results["Radar_Nav"].cummax()
    meta_dd = ((results["Radar_Nav"] / rolling_max_meta) - 1.0).min() * 100
    
    rolling_max_spy = results["SPY_Price"].cummax()
    spy_dd = ((results["SPY_Price"] / rolling_max_spy) - 1.0).min() * 100
    
    print(f"\nTOTAL SIMULATION RETURN ({penalty_weight}x Penalty):")
    print(f"XGBoost Black-Swan Radar    | Total Return: {meta_total:+.2f}% | Max DD: {meta_dd:,.2f}%")
    print(f"Baseline SPY (Buy & Hold)   | Total Return: {spy_total:+.2f}% | Max DD: {spy_dd:,.2f}%")
    print("=======================================================================\n")
    
    # --- PHASE 121: VISUAL FORENSICS RECONSTRUCTION ---
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['gray', 'lightblue', 'gold', 'lightgreen']
    labels = ['SPY Allocation', 'TLT Treasury Defense', 'GLD Gold Defense', '0% Yield Cash']
    
    ax.stackplot(results["Date"], results["SPY_Alloc"], results["TLT_Alloc"], results["GLD_Alloc"], results["Cash_Alloc"],
                 labels=labels, colors=colors, alpha=0.8)
    
    ax.set_title(f'Phase 125: Swarm Tri-Modal Defensive Allocation ({penalty_weight}x Alpha)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Percent Allocation (%)', fontsize=12)
    ax.set_xlabel('Date (Walk-Forward Execution Timeline)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xlim(results["Date"].iloc[0], results["Date"].iloc[-1])
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.xticks(rotation=45)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.tight_layout()
    plt.savefig('filled_allocation_map.png', dpi=150)
    print("[Visual Forensics] Successfully serialized 'filled_allocation_map.png'.")

if __name__ == "__main__":
    import sys
    weight = float(sys.argv[1]) if len(sys.argv) > 1 else 49.0
    backtest_meta_labeler(weight)

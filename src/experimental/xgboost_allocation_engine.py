import pandas as pd
import numpy as np
import sqlite3
import os
import xgboost as xgb
import yfinance as yf
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

from zscore_clustering_engine import calculate_rsi, calculate_stochastic, calculate_tsi

def get_ml_dataframe():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_data.db')
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM core_market_table', conn, index_col='Date')
    df.index = pd.to_datetime(df.index)
    
    # Re-calculate core engineered features dynamically
    try:
        vvix_data = yf.download(['^VVIX', '^VIX'], period='max', auto_adjust=False, progress=False)['Close'].dropna()
        df['VVIX'] = vvix_data['^VVIX']
        df['VIX_spot'] = vvix_data['^VIX']
        
        # Data Engineering Fix: Secure dynamically pulled VVIX/VIX history to prevent 2000-2006 NaN erasure
        df['VVIX'] = df['VVIX'].bfill()
        df['VIX_spot'] = df['VIX_spot'].bfill()
        
        df['VVIX_VIX_RATIO'] = df['VVIX'] / df['VIX_spot']
        for w in [5, 10, 20, 50, 100]:
            sma = df['VVIX'].rolling(w).mean()
            df[f'VVIX_PCT_SMA_{w}'] = (df['VVIX'] - sma) / sma
            df[f'VVIX_PCT_SMA_{w}'] = df[f'VVIX_PCT_SMA_{w}'].bfill()
    except Exception as e:
        pass

    for w in [10, 20, 50, 100, 200, 252]:
        sma = df['SPY_CLOSE'].rolling(w).mean()
        df[f'SPY_PCT_SMA_{w}'] = (df['SPY_CLOSE'] - sma) / sma
        
    for w in [10, 20, 50, 100, 200]:
        df[f'SPY_RSI_{w}'] = calculate_rsi(df['SPY_CLOSE'], w)
        
    for w in [5, 10, 20, 50, 100, 200, 252]:
        df[f'SPY_STOCH_{w}'] = calculate_stochastic(df['SPY_HIGH'], df['SPY_LOW'], df['SPY_CLOSE'], w)
        
    df['SPY_TSI_25_13'] = calculate_tsi(df['SPY_CLOSE'], 25, 13)
    df['SPY_TSI_SIGNAL_13'] = df['SPY_TSI_25_13'].ewm(span=13, adjust=False).mean()
    
    vix_tnx = df['VIX_CLOSE'] / df['TNX_CLOSE']
    df['VIX_TNX_TSI'] = calculate_tsi(vix_tnx, 25, 13)
    for w in [5, 10, 20, 50, 100, 200, 252]:
        vix_tnx_sma = vix_tnx.rolling(w).mean()
        df[f'VIX_TNX_PCT_SMA_{w}'] = (vix_tnx - vix_tnx_sma) / vix_tnx_sma
        
    df['SPY_VUSTX_DIFF_5D'] = df['SPY_CLOSE'].pct_change(5) - df['VUSTX_CLOSE'].pct_change(5)
    df['SPY_VUSTX_DIFF_10D'] = df['SPY_CLOSE'].pct_change(10) - df['VUSTX_CLOSE'].pct_change(10)
    
    df['SPY_Daily_Ret'] = df['SPY_CLOSE'].pct_change()
    df['Fwd_20D_Return'] = (df['SPY_CLOSE'].shift(-20) / df['SPY_CLOSE']) - 1
    
    # Calculate rolling 20-day forward minimum price to catch catastrophic intermediate V-Shapes
    df['Fwd_20D_Min_Price'] = df['SPY_CLOSE'].shift(-1).rolling(window=20).min().shift(-19)
    df['Fwd_20D_Max_Drawdown'] = (df['Fwd_20D_Min_Price'] / df['SPY_CLOSE']) - 1
    
    return df

def execute_xgboost_pipeline():
    print("Executing XGBoost Walk-Forward Protocol...")
    df = get_ml_dataframe()
    
    # Drop rows without targets and drop pure NaNs in predictors
    ml_df = df.dropna(subset=['Fwd_20D_Return']).copy()
    
    # Define our precise structural feature array
    # CRITICAL FIX: We must brutally strip out all absolute Dollar Prices, Raw Volumes, and Raw SMAs.
    # Feeding raw '$VUSTX' or 'VUSTX_SMA_252' into a 25-Year time-series model causes "Era Memorization" 
    # (The AI learns that higher dollar values just mean a more recent calendar year). 
    # We ONLY feed it unit-agnostic Stationary Vectors (PCT_SMA, Ratios, RSI, Spreads).
    
    # Nuke absolutely ALL raw Prices, OHLCV, and absolute Volume indicators
    excluded_cols = ['Fwd_20D_Return', 'Fwd_20D_Max_Drawdown', 'Fwd_20D_Min_Price', 'SPY_Daily_Ret']
    excluded_cols += [c for c in ml_df.columns if any(x in c for x in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'PRICE', 'VOLUME'])]
    excluded_cols += ['VVIX', 'VIX_spot', 'NYADV', 'NYDEC', 'NYUPV', 'NYDNV', 'NYADU', 'AD_LINE']
    
    # Nuke the absolute-number ALFRED/Central Bank timelines to force %-derivative logic
    excluded_cols += ['RECPROUSM156N', 'BOGMBASE', 'WALCL', 'TREAST', 'TSIFRGHT', 'JPNASSETS', 
                      'ECBASSETSW', 'DEXJPUS', 'DEXUSEU', 'World_CentralBank_BalSh', 
                      'MonetaryBase_50dMA', 'FederalReserveRecessionProbability_50dMA']
                      
    # Eliminate severe multicollinearity across the Central Bank metrics. 
    # We consolidate strictly upon 'World_CentralBank_BalSh_45d%Chg'.
    excluded_cols += ['FederalReserveTreasuryHoldings_45d%Chg', 'FederalReserveBalanceSheetSize_45d%Chg', 'FederalReserveBalanceSheetSize_20d%Chg']
    
    features = []
    for c in ml_df.columns:
        if c in excluded_cols: continue
        
        # Exclude raw simple moving averages that track underlying absolute dollar values
        if 'SPY_SMA' in c or 'VUSTX_SMA' in c or 'AD_LINE_SMA' in c: continue
        
        # Exclude Un-Scaled Nominal Differences (which cause massive Era Memorization as absolute Central Bank values inflate over decades).
        # We physically must retain only %Chg parameters.
        if 'Diff' in c and ('MonetaryBase' in c or 'TreasuryHoldings' in c or 'RecessionProbability' in c): continue
        if c in ['FederalReserveTreasuryHoldings_20dDiff', 'MonetaryBase_50dMA_20dDiff', 'MonetaryBase_50dMA_20dDiff_10dDiff', 'FederalReserveRecessionProbability_50dMA_5dDiff']: continue
        
        # Exclude non-stationary structural components of ratios
        if 'VIX_TNX_SMA' in c or 'VIX_TNX_BB' in c or 'VIX_TNX_STD' in c: continue
        
        features.append(c)
    
    ml_df = ml_df.dropna(subset=features)
    
    X = ml_df[features]
    # Classifier Requirement: The model will simply learn 1=UP, 0=DOWN, but will be punished if Risk hits > -5% intraday
    y = ((ml_df['Fwd_20D_Return'] > 0.0) & (ml_df['Fwd_20D_Max_Drawdown'] > -0.05)).astype(int)
    
    print(f"Matrix Dimension: {len(X)} Trading Days. Features: {len(features)}")
    
    train_size = 1260 # 5 Years of Trading Days
    step_size = 20    # Walk Forward 20 Days
    
    out_of_sample_preds = []
    out_of_sample_dates = []
    
    # XGBoost Hyperparameters structured securely for Classification Optimization
    xgb_params = {
        'n_estimators': 150,    # Board Recommendation: Increased trees
        'max_depth': 3,          # Board Recommendation: Strict cap to prevent overfitting
        'learning_rate': 0.05,
        'subsample': 0.8,        # Mathematically blind the AI to 20% of the days on each split
        'colsample_bytree': 0.8, # Mathematically blind the AI to 20% of indicators on each split
        'gamma': 0.5,
        'random_state': 42,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_jobs': -1
    }
    
    # Rolling Window Validation
    total_steps = (len(X) - train_size) // step_size
    print(f"Instantiating {total_steps} Independent XGBoost Overrides...")
    
    for i in range(0, len(X) - train_size, step_size):
        X_train = X.iloc[i : i + train_size]
        y_train = y.iloc[i : i + train_size]
        
        X_test = X.iloc[i + train_size : i + train_size + step_size]
        
        # Stop array bounds overflow
        if len(X_test) == 0:
            break
            
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        
        # We explicitly extract the Probability of being class 1 (UP)
        preds = model.predict_proba(X_test)[:, 1]
        
        out_of_sample_preds.extend(preds)
        out_of_sample_dates.extend(X_test.index)
        
    print("Walk-Forward Matrix Execution Fully Resolved.")
    
    # Construct Execution Analytics
    results_df = pd.DataFrame({
        'Predicted_Probability': out_of_sample_preds
    }, index=out_of_sample_dates)
    
    # Map back original Daily returns
    results_df = results_df.join(df['SPY_Daily_Ret']).dropna()
    
    # Logical Allocation Matrix
    # We demand the AI be mathematically confident (>55% Probability of an up-move).
    # If the probability is 54%, we entirely physically liquidate to CASH.
    results_df['Allocation'] = np.where(results_df['Predicted_Probability'] >= 0.55, 1.0, 0.0)
    
    results_df['XGBoost_Daily_Ret'] = results_df['SPY_Daily_Ret'] * results_df['Allocation']
    
    results_df['Cumulative_SPY'] = (1 + results_df['SPY_Daily_Ret']).cumprod()
    results_df['Cumulative_XGBoost'] = (1 + results_df['XGBoost_Daily_Ret']).cumprod()
    
    # Render the absolute Walk-Forward Performance Graphic
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.plot(results_df.index, results_df['Cumulative_SPY'], color='#555555', linewidth=2, label='SPY Baseline (Buy & Hold)')
    ax.plot(results_df.index, results_df['Cumulative_XGBoost'], color='#00ff00', linewidth=2.5, label='XGBoost Walk-Forward Physics Engine')
    
    ax.set_title('Out-of-Sample XGBoost Allocation Trajectory (5-Year Rolling Train, 20-Day Step)', fontsize=16, color='white', pad=20)
    ax.set_ylabel('Sequential Capital Multiple', fontsize=12, color='white')
    
    # Plot Allocation heat structure
    ax2 = ax.twinx()
    ax2.fill_between(results_df.index, 0, results_df['Allocation'], color='cyan', alpha=0.1, step='post')
    ax2.set_ylim(0, 5) # Hide it visually massive
    ax2.set_yticks([])
    
    ax.grid(color='#2A3459', linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=12, facecolor='#11152B', edgecolor='#435A94')
    
    out_path = '/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/walk_forward_curve_v7.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Equity Curve rendered natively to '{out_path}'.")

    # Generate SHAP values structurally across the final Master Model to decode systemic weights
    print("Decompiling final algorithmic state via SHAP...")
    global_model = xgb.XGBRegressor(**xgb_params)
    global_model.fit(X, y)
    
    explainer = shap.TreeExplainer(global_model)
    shap_values = explainer.shap_values(X)
    
    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title('XGBoost Macroeconomic Intelligence Decoder (Top 15 Absolute Features)', color='white')
    # Change axis label colors
    # Ensure dark mode compliance for the SHAP graph
    plt.savefig('/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/shap_decoder_v7.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("SHAP Matrix physically rendered.")

if __name__ == '__main__':
    execute_xgboost_pipeline()

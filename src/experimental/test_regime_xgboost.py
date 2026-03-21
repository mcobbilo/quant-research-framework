import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def get_regime(vix_val):
    if vix_val >= 20.0:
        return 1
    elif vix_val >= 15.0:
        return 0
    else:
        return -1

def run_multi_regime_backtest():
    print("[Regime AI] Downloading absolute dataset (2000-2025)...")
    spy_data = yf.download("SPY", start="2000-01-01", end="2025-01-01", progress=False, auto_adjust=True)
    vix_data = yf.download("^VIX", start="2000-01-01", end="2025-01-01", progress=False)

    df = pd.DataFrame(index=spy_data.index)
    if isinstance(spy_data.columns, pd.MultiIndex):
        df["SPY"] = spy_data[('Close', 'SPY')]
        df["VIX"] = vix_data[('Close', '^VIX')]
    else:
        df["SPY"] = spy_data['Close']
        df["VIX"] = vix_data['Close']
        
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    # Feature Engineering (The variables the AI will actually look at!)
    df['Returns_1d'] = df['SPY'].pct_change(1)
    df['Returns_3d'] = df['SPY'].pct_change(3)
    df['Returns_10d'] = df['SPY'].pct_change(10)
    df['VIX_Change_5d'] = df['VIX'].pct_change(5)
    
    # The Target: Did the market go UP exactly 10 days in the future?
    df['Target_10d_Ret'] = df['SPY'].shift(-10) / df['SPY'] - 1.0
    df['Target'] = (df['Target_10d_Ret'] > 0).astype(int)
    
    df.dropna(inplace=True)
    
    # Classify the Data strictly into Regimes
    df['Regime'] = df['VIX'].apply(get_regime)
    
    features = ['Returns_1d', 'Returns_3d', 'Returns_10d', 'VIX', 'VIX_Change_5d']
    
    # 3 Distinct AI Brains
    models = {
        1: xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42),
        0: xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42),
        -1: xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42)
    }
    
    # Walk-Forward parameters
    train_size = 1250 # Train on the first 5 years entirely
    retrain_freq = 120 # Re-train the models every 6 months to prevent infinite lag
    
    capital = 100000.0
    shares = 0.0
    margin_borrowed = 0.0
    previous_size = 0.0
    
    print(f"\n[Regime AI] System Initialized.")
    print(f"[Regime AI] Models 1, 0, and -1 loaded. Allocating walk-forward simulation...")
    
    for i in range(train_size, len(df) - 10):
        # Rolling Re-Training Sequence!
        if (i - train_size) % retrain_freq == 0:
            train_data = df.iloc[:i]
            
            # Sub-divide the training data and train the respective AI models contextually
            for reg in [-1, 0, 1]:
                reg_data = train_data[train_data['Regime'] == reg]
                if len(reg_data) > 30: 
                    X_train = reg_data[features]
                    y_train = reg_data['Target']
                    models[reg].fit(X_train, y_train)
        
        # Today's localized inference
        today = df.iloc[i]
        regime_today = int(today['Regime'])
        X_today = today[features].to_frame().T
        
        try:
            # Query the highly specialized model exclusively for TODAY'S VIX Regime!
            prob_up = models[regime_today].predict_proba(X_today)[0][1]
        except Exception:
            prob_up = 0.5 
            
        # Hardcoded Sizing Logic (Derived probabilistically)
        if prob_up > 0.58:
            size_target = 1.6 # Bullish Confidence
        elif prob_up < 0.45:
            size_target = 0.0 # Bearish Confidence -> Cash
        else:
            size_target = 1.0 # Uncertain -> SPY Baseline
            
        # Physical Friction Application
        price = today['SPY']
        friction = capital * abs(size_target - previous_size) * 0.001
        capital -= friction
        
        if margin_borrowed > 0:
            interest = margin_borrowed * (0.05 / 252)
            capital -= interest
            
        target_exposure = capital * size_target
        current_exposure = shares * price
        
        delta_exposure = target_exposure - current_exposure
        shares += delta_exposure / price
        
        if size_target > 1.0:
            margin_borrowed = (size_target - 1.0) * capital
        else:
            margin_borrowed = 0.0
            
        previous_size = size_target
        
    final_value = (shares * df.iloc[-11]['SPY']) - margin_borrowed
    total_ret = ((final_value - 100000.0) / 100000.0) * 100
    
    print("\n================= 20-YEAR SYSTEM YIELD =================")
    print(f"XGBoost Multi-Regime Ensemble | Total Return: {total_ret:,.2f}%")
    print("========================================================\n")

if __name__ == "__main__":
    run_multi_regime_backtest()

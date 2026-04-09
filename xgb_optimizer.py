import os
import pandas as pd
import numpy as np
import sqlite3
import sys
import xgboost as xgb
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def optimize_xgb():
    print(f"\n[PHASE 1] LOGISTIC REGRESSION L1 FEATURE EXTRACTION...")
    
    db_path = os.path.join("src", "data", "market_data.db")
    conn = sqlite3.connect(db_path)
    
    query = "SELECT * FROM core_market_table"
    df_core = pd.read_sql_query(query, conn)
    conn.close()
    
    df_core['ds'] = pd.to_datetime(df_core['Date']).dt.tz_localize(None)
    df_core.sort_values('ds', inplace=True)
    df_core.reset_index(drop=True, inplace=True)
    
    df_core = df_core.rename(columns={
        'SPY_CLOSE': 'spy_close', 
        'VUSTX_CLOSE': 'vustx_close', 
        'GLD_CLOSE': 'gld_close', 
        'VIX_CLOSE': 'vix_close'
    })
    
    df_returns = df_core[['ds', 'vustx_close', 'gld_close']].copy()
    df_returns['vustx_ret'] = df_returns['vustx_close'].pct_change()
    df_returns['gld_ret'] = df_returns['gld_close'].pct_change()
    df_returns['vustx_close_prev'] = df_returns['vustx_close'].shift(1)
    df_returns['gld_close_prev'] = df_returns['gld_close'].shift(1)
    df_returns['vustx_sma200_prev'] = df_returns['vustx_close_prev'].rolling(200).mean()
    df_returns['gld_sma200_prev'] = df_returns['gld_close_prev'].rolling(200).mean()
    
    HORIZON = 10
    
    # Binary classification labels for SPY
    # Drop the last HORIZON rows because we don't know the future yet!
    # They would return false (0) making the algorithm learn noise at the end of the dataset.
    df_core['y_spy'] = (df_core['spy_close'].shift(-HORIZON) > df_core['spy_close']).astype(int)
    df_core['y_vustx'] = (df_core['vustx_close'].shift(-HORIZON) > df_core['vustx_close']).astype(int)
    df_core['y_gld'] = (df_core['gld_close'].shift(-HORIZON) > df_core['gld_close']).astype(int)
    df_core['y_vix'] = (df_core['vix_close'].shift(-HORIZON) > df_core['vix_close']).astype(int)
    
    exclude_cols = ['ds', 'Date', 'spy_close', 'vustx_close', 'gld_close', 'vix_close', 'PRICEMAP', 'unique_id'] + [c for c in df_core.columns if c.startswith('y_')]
    
    all_features = [c for c in df_core.columns if c not in exclude_cols]
    all_features = df_core[all_features].select_dtypes(include=[np.number]).columns.tolist()
    
    df_core[all_features] = df_core[all_features].ffill().bfill()
    
    valid_train = df_core.iloc[:-HORIZON].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(valid_train[all_features])
    y_target = valid_train['y_spy'].values
    
    print(f"Running L1 Logistic Regression over {len(all_features)} numeric features...")
    lr = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42, max_iter=200)
    lr.fit(X_scaled, y_target)
    
    coef_abs = np.abs(lr.coef_[0])
    feature_importances = pd.DataFrame({'feature': all_features, 'importance': coef_abs})
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    
    top_60_features = feature_importances.head(60)['feature'].tolist()
    
    print("\n--- TOP 10 EXTRACTED FEATURES ---")
    for idx, f in enumerate(feature_importances.head(10)['feature'].tolist()):
        print(f"{idx+1}. {f}")
        
    print("\n[PHASE 2] XGBOOST WALK-FORWARD WITH TOP 60 FEATURES...")
    
    total_steps = 400
    step_size = 10
    end_idx = len(df_core) - HORIZON
    start_idx = end_idx - (total_steps * step_size)
    
    results = []
    
    for i in tqdm(range(total_steps)):
        current_cutoff_idx = start_idx + i * step_size
        
        train_end_idx = current_cutoff_idx - HORIZON
        df_train = df_core.iloc[:train_end_idx]
        X_train = df_train[top_60_features]
        
        df_test = df_core.iloc[[current_cutoff_idx]]
        X_test = df_test[top_60_features]
        test_ds = df_test['ds'].values[0]
        
        probs = {'signal_date': test_ds}
        
        for asset in ['spy', 'vustx', 'gld', 'vix']:
            y_train = df_train[f'y_{asset}']
            clf = xgb.XGBClassifier(
                n_estimators=100, 
                max_depth=4, 
                learning_rate=0.03, 
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=42, 
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            prob1 = clf.predict_proba(X_test)[0][1]
            probs[f'prob_up_{asset}'] = prob1
            
        results.append(probs)

    df_pred_pivot = pd.DataFrame(results)
    
    print("\n[PHASE 3] GRID SEARCH THRESHOLD EVALUATION...")
    df_cont = df_core[['ds', 'spy_close']].copy().dropna()
    df_cont['y_prev'] = df_cont['spy_close'].shift(1)
    df_cont = df_cont.dropna(subset=['y_prev']).copy()
    
    df_cont = pd.merge(df_cont, df_returns, on='ds', how='left')
    df_cont['asset_return'] = (df_cont['spy_close'] - df_cont['y_prev']) / df_cont['y_prev']
    
    df_cont = pd.merge(df_cont, df_pred_pivot, left_on='ds', right_on='signal_date', how='left')
    df_cont[['prob_up_spy', 'prob_up_vustx', 'prob_up_gld', 'prob_up_vix']] = df_cont[['prob_up_spy', 'prob_up_vustx', 'prob_up_gld', 'prob_up_vix']].ffill()
    df_cont = df_cont.dropna(subset=['prob_up_spy'])
    
    best_yield = 0
    best_params = {}
    
    spy_thresholds = np.linspace(0.45, 0.60, 16)
    vix_thresholds = np.linspace(0.50, 0.80, 16)
    
    grid_results = []
    
    for spy_t in spy_thresholds:
        for vix_t in vix_thresholds:
            alloc = []
            rets = []
            for idx, row in df_cont.iterrows():
                if row['prob_up_vix'] > vix_t:
                    alloc.append('CASH')
                    rets.append(0.0)
                elif row['prob_up_spy'] >= spy_t:
                    alloc.append('SPY')
                    rets.append(row['asset_return'])
                else:
                    if row['prob_up_vustx'] > row['prob_up_gld'] and row['prob_up_vustx'] > 0.50:
                        alloc.append('VUSTX')
                        rets.append(row['vustx_ret'])
                    elif row['prob_up_gld'] > row['prob_up_vustx'] and row['prob_up_gld'] > 0.50:
                        alloc.append('GLD')
                        rets.append(row['gld_ret'])
                    elif row['vustx_close_prev'] > row['vustx_sma200_prev']:
                        alloc.append('VUSTX')
                        rets.append(row['vustx_ret'])
                    elif row['gld_close_prev'] > row['gld_sma200_prev']:
                        alloc.append('GLD')
                        rets.append(row['gld_ret'])
                    else:
                        alloc.append('CASH')
                        rets.append(0.0)
                        
            strat_rets = pd.Series(rets)
            cum_strat = (1 + strat_rets).cumprod()
            final_y = cum_strat.iloc[-1] if not cum_strat.empty else 1.0
            
            peak = cum_strat.expanding(min_periods=1).max()
            mdd = ((cum_strat / peak) - 1).min() if not peak.empty else 0.0
            
            grid_results.append({
                'spy_t': round(spy_t, 2),
                'vix_t': round(vix_t, 2),
                'yield': round(final_y, 2),
                'mdd': round(mdd, 4)
            })

    df_grid = pd.DataFrame(grid_results)
    df_grid = df_grid.sort_values('yield', ascending=False).head(10)
    
    print("\n--- TOP THRESHOLD COORDINATES FOR XGBOOST ---")
    print(df_grid.to_string(index=False))

if __name__ == "__main__":
    optimize_xgb()

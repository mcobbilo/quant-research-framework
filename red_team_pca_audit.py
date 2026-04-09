import os
import pandas as pd
import numpy as np
import sqlite3
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.xlstm_wrapper import xLSTMForecast
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

def run_pca_audit(seed=42):
    print(f"\n[SEED {seed} PCA INJECTION] INIT...")
    
    db_path = os.path.join("src", "data", "market_data.db")
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT *
        FROM core_market_table
    """
    df_core = pd.read_sql_query(query, conn)
    conn.close()
    
    df_core['ds'] = pd.to_datetime(df_core['Date']).dt.tz_localize(None)
    df_core = df_core.rename(columns={'SPY_CLOSE': 'spy_close', 'VUSTX_CLOSE': 'vustx_close', 'GLD_CLOSE': 'gld_close', 'VIX_CLOSE': 'vix_close'})
    
    df_returns = df_core[['ds', 'vustx_close', 'gld_close']].copy()
    df_returns['vustx_ret'] = df_returns['vustx_close'].pct_change()
    df_returns['gld_ret'] = df_returns['gld_close'].pct_change()
    df_returns['vustx_close_prev'] = df_returns['vustx_close'].shift(1)
    df_returns['gld_close_prev'] = df_returns['gld_close'].shift(1)
    df_returns['vustx_sma200_prev'] = df_returns['vustx_close_prev'].rolling(200).mean()
    df_returns['gld_sma200_prev'] = df_returns['gld_close_prev'].rolling(200).mean()
    
    exclude_cols = ['ds', 'Date', 'spy_close', 'vustx_close', 'gld_close', 'vix_close', 'PRICEMAP', 'y', 'unique_id']
    hist_exog = [c for c in df_core.columns if c not in exclude_cols]
    hist_exog = df_core[hist_exog].select_dtypes(include=[np.number]).columns.tolist()
    
    df_core[hist_exog] = df_core[hist_exog].ffill().bfill()
    
    # ---------------------------------------------------------
    # PCA IMPLEMENTATION
    # ---------------------------------------------------------
    scaler = StandardScaler()
    scaled_exog = scaler.fit_transform(df_core[hist_exog])
    
    pca = PCA(n_components=15)
    pca_features = pca.fit_transform(scaled_exog)
    
    # Replace original exog columns with PCA components
    pca_cols = [f'pca_{i}' for i in range(15)]
    df_pca = pd.DataFrame(pca_features, columns=pca_cols)
    for col in pca_cols:
        df_core[col] = df_pca[col]
    
    # ---------------------------------------------------------
    
    panels = []
    assets_map = {'SPY': 'spy_close', 'VUSTX': 'vustx_close', 'GLD': 'gld_close', 'VIX': 'vix_close'}
    for uid, target_col in assets_map.items():
        df_sub = df_core.copy()
        df_sub['unique_id'] = uid
        df_sub['PRICEMAP'] = df_sub[target_col]
        df_sub['y'] = (df_sub['PRICEMAP'] > df_sub['PRICEMAP'].shift(10)).astype(int)
        panels.append(df_sub)
        
    df_panel = pd.concat(panels, ignore_index=True)
    
    df_model = df_panel[['unique_id', 'ds', 'y'] + pca_cols].dropna()
    
    WINDOWS = 400
    HORIZON = 10
    STEP_SIZE = 10
    
    model = xLSTMForecast(
        h=HORIZON, input_size=30, max_steps=10, 
        hist_exog_list=pca_cols, freq='B', random_seed=seed
    )
    
    cv_df = model.cross_validation(df_model, n_windows=WINDOWS, step_size=STEP_SIZE)
    
    cv_df['ds'] = pd.to_datetime(cv_df['ds'])
    cv_df['cutoff'] = pd.to_datetime(cv_df['cutoff'])
    
    df_pred = cv_df.groupby(['unique_id', 'cutoff']).last().reset_index()
    df_pred_pivot = df_pred.pivot(index='cutoff', columns='unique_id', values='xLSTM').reset_index()
    df_pred_pivot = df_pred_pivot.rename(columns={'cutoff':'signal_date','SPY':'prob_up_spy','VUSTX':'prob_up_vustx','GLD':'prob_up_gld','VIX':'prob_up_vix'})
    
    df_cont = df_core[['ds', 'spy_close']].copy().dropna()
    df_cont['y_prev'] = df_cont['spy_close'].shift(1)
    df_cont = df_cont.dropna(subset=['y_prev']).copy()
    
    df_cont = pd.merge(df_cont, df_returns, on='ds', how='left')
    df_cont['asset_return'] = (df_cont['spy_close'] - df_cont['y_prev']) / df_cont['y_prev']
    
    df_cont = pd.merge(df_cont, df_pred_pivot, left_on='ds', right_on='signal_date', how='left')
    df_cont[['prob_up_spy', 'prob_up_vustx', 'prob_up_gld', 'prob_up_vix']] = df_cont[['prob_up_spy', 'prob_up_vustx', 'prob_up_gld', 'prob_up_vix']].ffill()
    df_cont = df_cont.dropna(subset=['prob_up_spy'])
    
    alloc = []
    rets = []
    for idx, row in df_cont.iterrows():
        # WE WILL RELAX THE SPIKE THRESHOLD IN EXPECTATION OF PCA CHANGES
        if row['prob_up_vix'] > 0.60:
            alloc.append('CASH')
            rets.append(0.0)
        elif row['prob_up_spy'] >= 0.51:
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
                
    df_cont['strategy_allocation'] = alloc
    df_cont['strategy_return'] = rets
    
    cum_strat = (1 + df_cont['strategy_return']).cumprod()
    final_ret = cum_strat.iloc[-1]
    
    cum_asset = (1 + df_cont['asset_return']).cumprod()
    base_ret = cum_asset.iloc[-1]
    
    cum = (1 + df_cont['strategy_return']).cumprod()
    peak = cum.expanding(min_periods=1).max()
    mdd = ((cum / peak) - 1).min()
    
    print(f"[PCA {seed}] COMPLETE | Asset Yield: {base_ret:.2f}x | Model Yield: {final_ret:.2f}x | Max DD: {mdd:.2%}")
    return final_ret, mdd

if __name__ == "__main__":
    print("================================")
    print(" PCA STRUCTURAL AUDIT")
    print("================================")
    run_pca_audit(seed=42)
    run_pca_audit(seed=0)
    run_pca_audit(seed=1337)

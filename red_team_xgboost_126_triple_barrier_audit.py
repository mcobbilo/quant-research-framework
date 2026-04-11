import os
import pandas as pd
import numpy as np
import sqlite3
import sys
import xgboost as xgb
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def run_xgboost_audit(seed=0, noise_std=0.0, guillotine=False, test_name="BASELINE"):
    print(f"\n[{test_name}] INIT...")
    
    db_path = os.path.join("src", "data", "market_data.db")
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
    
    df_core['ds'] = pd.to_datetime(df_core['ds']).dt.tz_localize(None)
    df_core.sort_values('ds', inplace=True)
    df_core.reset_index(drop=True, inplace=True)
    
    if guillotine:
        # Time-travel data truncation
        df_core = df_core.iloc[:-1000].reset_index(drop=True)
        
    np.random.seed(seed)
        
    df_returns = df_core[['ds', 'vustx_close', 'gld_close', 'hg_close', 'vix_close', 'skew_zscore_252', 'vix_vxv_ratio_z', 'sent_0dte_repression_z']].copy()
    df_returns['vustx_ret'] = df_returns['vustx_close'].pct_change()
    df_returns['gld_ret'] = df_returns['gld_close'].pct_change()
    df_returns['vustx_close_prev'] = df_returns['vustx_close'].shift(1)
    df_returns['gld_close_prev'] = df_returns['gld_close'].shift(1)
    df_returns['vustx_sma200_prev'] = df_returns['vustx_close_prev'].rolling(200).mean()
    df_returns['gld_sma200_prev'] = df_returns['gld_close_prev'].rolling(200).mean()
    
    df_returns['hg_close_prev'] = df_returns['hg_close'].shift(1)
    df_returns['hg_sma200_prev'] = df_returns['hg_close_prev'].rolling(200).mean()
    df_returns['vix_close_prev'] = df_returns['vix_close'].shift(1)
    
    features = [
        'volume', 'high', 'low', 't10y2y', 'vix_bb_width', 'spy_vwap_20', 
        'nyhgh_nylow_10d', 'world_cb_liq', 'vix_tnx_pct_200', 'vix_tnx_tsi', 'vix_tsi', 'vix_vvix_ratio', 'spy_ppo_signal',
        'vix_vvix_ratio_z', 'sent_crossasset_vol_ratio', 'sent_0dte_repression_z', 'vix_vxv_ratio_z', 'spy_vol_ratio_21_252', 'spy_accel_mom',
        'spy_atr_14', 'spy_atr_pct', 'market_breadth_zscore_252d', 'vix_to_10y_mom21', 'corr_spy_vustx_63', 'nya200r'
    ]
    df_core[features] = df_core[features].ffill().bfill()

    if noise_std > 0.0:
        for f in features:
            if df_core[f].dtype in [np.float64, np.float32]:
                noise = np.random.normal(0, noise_std * df_core[f].std(), len(df_core))
                df_core[f] += noise

    HORIZON = 10
    
    print("Generating Triple-Barrier Meta-Labels...")
    n_len = len(df_core)
    spy_closes = df_core['spy_close'].values
    spy_highs = df_core['high'].values
    spy_lows = df_core['low'].values
    spy_atrs = df_core['spy_atr_pct'].values
    vustx_closes = df_core['vustx_close'].values
    gld_closes = df_core['gld_close'].values
    vix_closes = df_core['vix_close'].values

    y_spy_arr = np.zeros(n_len, dtype=int)
    y_vustx_arr = np.zeros(n_len, dtype=int)
    y_gld_arr = np.zeros(n_len, dtype=int)
    y_vix_arr = np.zeros(n_len, dtype=int)

    for i in range(n_len):
        end_idx = min(n_len, i + HORIZON + 1)
        
        # SPY Triple Barrier
        pt_spy = spy_closes[i] * (1.0 + (spy_atrs[i] * 1.5))
        sl_spy = spy_closes[i] * (1.0 - (spy_atrs[i] * 1.0))
        spy_lbl = 0
        for j in range(i+1, end_idx):
            if spy_highs[j] >= pt_spy:
                spy_lbl = 1
                break
            if spy_lows[j] <= sl_spy:
                spy_lbl = 0 # Lower Barrier takes precedence
                break
        else: # Time Expired without barrier touch
            if end_idx - 1 < n_len:
                spy_lbl = 1 if spy_closes[end_idx - 1] > spy_closes[i] else 0
        y_spy_arr[i] = spy_lbl

        # VUSTX Fixed Barrier (Upper +2.0%, Lower -1.5%)
        pt_vustx = vustx_closes[i] * 1.020
        sl_vustx = vustx_closes[i] * 0.985
        v_lbl = 0
        for j in range(i+1, end_idx):
            if vustx_closes[j] >= pt_vustx:
                v_lbl = 1
                break
            if vustx_closes[j] <= sl_vustx:
                v_lbl = 0
                break
        else:
            if end_idx - 1 < n_len:
                v_lbl = 1 if vustx_closes[end_idx-1] > vustx_closes[i] else 0
        y_vustx_arr[i] = v_lbl

        # GLD Fixed Barrier (Upper +2.5%, Lower -1.5%)
        pt_gld = gld_closes[i] * 1.025
        sl_gld = gld_closes[i] * 0.985
        g_lbl = 0
        for j in range(i+1, end_idx):
            if gld_closes[j] >= pt_gld:
                g_lbl = 1
                break
            if gld_closes[j] <= sl_gld:
                g_lbl = 0
                break
        else:
            if end_idx - 1 < n_len:
                g_lbl = 1 if gld_closes[end_idx-1] > gld_closes[i] else 0
        y_gld_arr[i] = g_lbl
        
        # VIX Fixed Barrier (Upper +10%, Lower -8%)
        pt_vix = vix_closes[i] * 1.10
        sl_vix = vix_closes[i] * 0.92
        vx_lbl = 0
        for j in range(i+1, end_idx):
            if vix_closes[j] >= pt_vix:
                vx_lbl = 1
                break
            if vix_closes[j] <= sl_vix:
                vx_lbl = 0
                break
        else:
            if end_idx - 1 < n_len:
                vx_lbl = 1 if vix_closes[end_idx-1] > vix_closes[i] else 0
        y_vix_arr[i] = vx_lbl

    df_core['y_spy'] = y_spy_arr
    df_core['y_vustx'] = y_vustx_arr
    df_core['y_gld'] = y_gld_arr
    df_core['y_vix'] = y_vix_arr

    total_steps = 252 if not guillotine else 150
    step_size = 10
    
    end_idx = len(df_core) - HORIZON
    start_idx = end_idx - (total_steps * step_size)
    
    results = []
    print("Training XGBoost...")
    for i in tqdm(range(total_steps)):
        current_cutoff_idx = start_idx + i * step_size
        train_end_idx = current_cutoff_idx - HORIZON
        df_train = df_core.iloc[:train_end_idx]
        X_train = df_train[features]
        df_test = df_core.iloc[[current_cutoff_idx]]
        X_test = df_test[features]
        test_ds = df_test['ds'].values[0]
        
        probs = {'signal_date': test_ds}
        
        for asset in ['spy', 'vustx', 'gld', 'vix']:
            target_col = f'y_{asset}'
            y_train = df_train[target_col]
            
            clf = xgb.XGBClassifier(
                n_estimators=150, 
                max_depth=4, 
                learning_rate=0.03, 
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=seed, 
                n_jobs=1
            )
            clf.fit(X_train, y_train)
            prob1 = clf.predict_proba(X_test)[0][1]
            probs[f'prob_up_{asset}'] = prob1
            
        results.append(probs)

    df_pred_pivot = pd.DataFrame(results)
    
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
        # Phase 126 Master Executions
        copper_strength = row['hg_close_prev'] / row['hg_sma200_prev'] if row['hg_sma200_prev'] > 0 else 1.0
        
        # Doctor Copper Defensive Block
        dr_copper_veto = (copper_strength < 1.04)
        
        # Gamma Exposure (GEX) Proxy Calculation
        gex_proxy = row['skew_zscore_252'] + row['vix_vxv_ratio_z'] + row['sent_0dte_repression_z']
        
        # VIX Panic Buy Override (VETOED IF SHORT GAMMA)
        vix_panic_override = (row['vix_close_prev'] > 20.0) and (gex_proxy <= 1.5)

        # Base prob logic
        prob_cash = row['prob_up_vix'] > 0.60
        prob_spy = row['prob_up_spy'] >= 0.51

        if dr_copper_veto:
            if row['prob_up_vustx'] > row['prob_up_gld'] and row['prob_up_vustx'] > 0.50:
                alloc.append('VUSTX')
                rets.append(row['vustx_ret'])
            elif row['prob_up_gld'] > row['prob_up_vustx'] and row['prob_up_gld'] > 0.50:
                alloc.append('GLD')
                rets.append(row['gld_ret'])
            else:
                alloc.append('CASH')
                rets.append(0.0)
        else:
            if vix_panic_override:
                alloc.append('SPY')
                rets.append(row['asset_return'])
            elif prob_cash:
                alloc.append('CASH')
                rets.append(0.0)
            elif prob_spy:
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
    
    print(f"[{test_name}] PHASE 126 XGBOOST COMPLETE | Asset Yield: {base_ret:.2f}x | Model Yield: {final_ret:.2f}x | Max DD: {mdd:.2%}")
    return final_ret, mdd

if __name__ == "__main__":
    import sys
    print("====================================")
    print(" PHASE 126 XGBOOST RED TEAM AUDIT ")
    print("====================================")
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "all"
        
    try:
        if mode == "seed0" or mode == "all":
            run_xgboost_audit(seed=0, noise_std=0.0, guillotine=False, test_name="SEED_0_PERTURBATION")
        if mode == "noise" or mode == "all":
            run_xgboost_audit(seed=42, noise_std=0.05, guillotine=False, test_name="5PCT_NOISE_INJECTION")
        if mode == "guillotine" or mode == "all":
            run_xgboost_audit(seed=42, noise_std=0.0, guillotine=True, test_name="DATA_GUILLOTINE_TIMETRAVEL")
    except Exception as e:
        print(f"Error during execution: {e}")

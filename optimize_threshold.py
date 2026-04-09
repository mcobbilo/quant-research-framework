import pandas as pd
import numpy as np

print("Loading previous CV predictions...")
cv_df = pd.read_csv("xLSTM_backtest_results.csv")
cv_df['ds'] = pd.to_datetime(cv_df['ds'])
cv_df['cutoff'] = pd.to_datetime(cv_df['cutoff'])

df_pred = cv_df.groupby('cutoff').last().reset_index()
df_pred = df_pred.rename(columns={'xLSTM': 'prob_up', 'cutoff': 'signal_date'})

# Re-pull safe haven returns
import sqlite3
import os
db_path = os.path.join("src", "data", "market_data.db")
conn = sqlite3.connect(db_path)
df_core = pd.read_sql_query("SELECT Date as ds, SPY_CLOSE as y, VUSTX_CLOSE as vustx_close, GLD_CLOSE as gld_close FROM core_market_table", conn)
conn.close()

df_core['ds'] = pd.to_datetime(df_core['ds']).dt.tz_localize(None)
df_core['SPY_PRICEMAP'] = df_core['y']
df_returns = df_core[['ds', 'vustx_close', 'gld_close']].copy()
df_returns['vustx_ret'] = df_returns['vustx_close'].pct_change()
df_returns['gld_ret'] = df_returns['gld_close'].pct_change()
df_returns['vustx_close_prev'] = df_returns['vustx_close'].shift(1)
df_returns['gld_close_prev'] = df_returns['gld_close'].shift(1)
df_returns['vustx_sma200_prev'] = df_returns['vustx_close_prev'].rolling(200).mean()
df_returns['gld_sma200_prev'] = df_returns['gld_close_prev'].rolling(200).mean()

thresholds = np.arange(0.50, 0.96, 0.01)
best_return = 0
best_thresh = 0

print("Threshold | Strat Return | Max DD | SPY Exposure | VUSTX Expo | CASH Expo")
print("-" * 75)

results = []
for t in thresholds:
    df_cont = df_core[['ds', 'SPY_PRICEMAP']].copy().dropna()
    df_cont['y_prev'] = df_cont['SPY_PRICEMAP'].shift(1)
    df_cont = df_cont.dropna(subset=['y_prev']).copy()
    
    df_cont = pd.merge(df_cont, df_returns, on='ds', how='left')
    df_cont['asset_return'] = (df_cont['SPY_PRICEMAP'] - df_cont['y_prev']) / df_cont['y_prev']
    
    df_cont = pd.merge(df_cont, df_pred[['signal_date', 'prob_up']], left_on='ds', right_on='signal_date', how='left')
    df_cont['prob_up'] = df_cont['prob_up'].ffill()
    df_cont = df_cont.dropna(subset=['prob_up'])
    
    alloc = []
    rets = []
    
    for idx, row in df_cont.iterrows():
        if row['prob_up'] >= t:
            alloc.append('SPY')
            rets.append(row['asset_return'])
        else:
            if row['vustx_close_prev'] > row['vustx_sma200_prev']:
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
    
    cum = (1 + df_cont['strategy_return']).cumprod()
    peak = cum.expanding(min_periods=1).max()
    dd = (cum / peak) - 1
    mdd = dd.min()
    
    alloc_counts = df_cont['strategy_allocation'].value_counts(normalize=True)
    spy_exp = alloc_counts.get('SPY', 0)
    vustx_exp = alloc_counts.get('VUSTX', 0)
    cash_exp = alloc_counts.get('CASH', 0)
    
    results.append({'thresh': t, 'ret': final_ret, 'mdd': mdd, 'spy': spy_exp, 'vustx': vustx_exp, 'cash': cash_exp})
    print(f"{t:.2f}      | {final_ret:>10.2f}x | {mdd:>6.2%} | {spy_exp:>12.1%} | {vustx_exp:>10.1%} | {cash_exp:>9.1%}")
    
results_df = pd.DataFrame(results)
best = results_df.loc[results_df['ret'].idxmax()]
print(f"\nOPTIMAL THRESHOLD: {best['thresh']:.2f} yielding {best['ret']:.2f}x")

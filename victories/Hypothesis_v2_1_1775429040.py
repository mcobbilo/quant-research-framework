import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('src/data/market_data.db')
df = pd.read_sql_query("SELECT * FROM core_market_table", conn)
conn.close()

if df.empty or len(df) < 500:
    print("RESULT_YIELD: 0.0000")
    print("RESULT_SHARPE: 0.0000")
else:
    df.columns = [c.upper() for c in df.columns]
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.set_index('DATE')
    df = df.sort_index()
    df = df.ffill()

    df['SPY_CLOSE'] = pd.to_numeric(df['SPY_CLOSE'], errors='coerce')
    df['RET'] = np.log(df['SPY_CLOSE'] / df['SPY_CLOSE'].shift(1))
    df = df.dropna(subset=['RET'])

    returns = df['RET'].values
    n = len(df)
    W = 60
    K = 7
    gamma = 2.0
    H_list = [np.nan] * n
    ks = np.arange(1, K) / float(K)

    for i in range(n):
        if i < W + 252:
            continue
        past_rets = returns[:i]
        if len(past_rets) == 0:
            continue
        qs = np.nanquantile(past_rets, ks)
        window_rets = returns[i - W:i]
        if len(window_rets) < W:
            continue
        digitized = np.digitize(window_rets, qs)
        counts = np.bincount(digitized, minlength=K)
        p_k = counts.astype(float) / W
        mask = p_k > 1e-8
        if np.sum(mask) == 0:
            h_norm = 0.0
        else:
            h = -np.sum(p_k[mask] * np.log2(p_k[mask]))
            h_norm = h / np.log2(K)
        H_list[i] = h_norm

    df['H_NORM'] = H_list
    df['H_NORM'] = df['H_NORM'].ffill().fillna(1.0)
    df['H_STAR'] = df['H_NORM'].ewm(span=252, adjust=False).mean().fillna(1.0)

    info_surplus = (df['H_STAR'] - df['H_NORM']) / df['H_STAR'].replace(0, np.nan)
    info_surplus = info_surplus.fillna(0.0)
    info_factor = np.maximum(info_surplus, 0.0)
    M = W
    df['MOM_SUM'] = df['RET'].rolling(window=M).sum()
    df['SIGN'] = np.sign(df['MOM_SUM']).fillna(0.0)
    df['EXPOSURE'] = (info_factor ** gamma) * df['SIGN']

    df['VOL_21'] = df['RET'].rolling(21).std() * np.sqrt(252)
    vol_target = 0.12
    df['LEVERAGE'] = vol_target / df['VOL_21'].replace(0, np.nan)
    df['LEVERAGE'] = df['LEVERAGE'].fillna(0.0).clip(upper=5.0, lower=0.0)
    df['FINAL_POS'] = df['EXPOSURE'] * df['LEVERAGE']
    df['STRAT_RET'] = df['FINAL_POS'].shift(1) * df['RET']

    valid_rets = df['STRAT_RET'].dropna()
    if len(valid_rets) > 100:
        ann_yield = valid_rets.mean() * 252
        ann_vol = valid_rets.std() * np.sqrt(252)
        sharpe = ann_yield / ann_vol if ann_vol > 0 else 0.0
    else:
        ann_yield = 0.0
        sharpe = 0.0

    print("RESULT_YIELD: {:.4f}".format(ann_yield))
    print("RESULT_SHARPE: {:.4f}".format(sharpe))
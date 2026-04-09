import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import sqlite3

# ====================== DATA LOADING ======================
conn = sqlite3.connect('src/data/market_data.db')
df = pd.read_sql_query("SELECT * FROM core_market_table", conn)
conn.close()

df.columns = [c.upper() for c in df.columns]

required_cols = ['DATE', 'SPY_CLOSE']
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

burnin = 800
if len(df) <= burnin:
    print("RESULT_YIELD: 0.0000")
    print("RESULT_SHARPE: 0.0000")
    # Satisfy framework that expects WEALTH column
    df['WEALTH'] = 1.0
    df['STRATEGY_RET'] = 0.0
    exit(0)

df = df.sort_values('DATE').reset_index(drop=True)

# Convert all columns except DATE to numeric
for col in df.columns:
    if col != 'DATE':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.ffill().bfill()

# ====================== FEATURE ENGINEERING ======================
df['RET'] = df['SPY_CLOSE'].pct_change().fillna(0.0)
ret_arr = df['RET'].values.copy()
sq_ret = ret_arr ** 2

df['RV1'] = pd.Series(sq_ret).rolling(1, min_periods=1).sum()
df['RV5'] = pd.Series(sq_ret).rolling(5, min_periods=1).sum()
df['RV22'] = pd.Series(sq_ret).rolling(22, min_periods=1).sum()
df['RV_RATIO'] = df['RV5'] / (df['RV22'] + 1e-8)
df['DELTA_RV5'] = df['RV5'].diff().fillna(0.0)
df['RV_OF_RV'] = df['RV5'].rolling(22, min_periods=1).std().fillna(0.0)

features = ['RV1', 'RV5', 'RV22', 'RV_RATIO', 'DELTA_RV5', 'RV_OF_RV']
df[features] = df[features].apply(pd.to_numeric, errors='coerce').fillna(0.0)

# ====================== MODEL PARAMETERS ======================
n = len(df)
K = 3
gmm_window = 500
lamb = 8.0
min_var = 1e-5

current_mu = np.full(K, 0.0004)
current_var = np.full(K, 0.00015)
regime_count = np.ones(K)

strategy_returns = np.zeros(n)

# ====================== MAIN LOOP (No Lookahead) ======================
for t in range(burnin, n - 1):
    start = max(t - gmm_window, 0)
    X = df.iloc[start:t][features].values.copy()
    
    if len(X) < 100:
        strategy_returns[t] = 0.0
        continue

    gmm = GaussianMixture(
        n_components=K,
        random_state=42,
        covariance_type='full',
        max_iter=100,
        n_init=1
    )
    gmm.fit(X)

    v_t = df.iloc[t:t+1][features].values.copy()
    gamma = gmm.predict_proba(v_t)[0]

    w = 0.0
    for k in range(K):
        mu_k = current_mu[k]
        sig2_k = max(current_var[k], min_var)
        alloc = mu_k / (lamb * sig2_k)
        w += gamma[k] * max(alloc, 0.0)

    ret_next = df['RET'].iloc[t + 1]
    strategy_returns[t] = w * ret_next

    # ====================== ONLINE UPDATE (Fixed) ======================
    for k in range(K):
        gk = gamma[k]
        if gk > 1e-8:
            old_mu = current_mu[k]
            # Update mean
            current_mu[k] = (current_mu[k] * regime_count[k] + gk * ret_next) / (regime_count[k] + gk)
            # Use old mean for error (correct statistical update)
            err = ret_next - old_mu
            # Update variance
            current_var[k] = (current_var[k] * regime_count[k] + gk * err**2) / (regime_count[k] + gk)
            regime_count[k] += gk

# ====================== PERFORMANCE ======================
df['STRATEGY_RET'] = strategy_returns
wealth = np.cumprod(1.0 + strategy_returns)
wealth = np.maximum(wealth, 0.0)
df['WEALTH'] = wealth

# Use only post-burnin returns (excluding final zero if present)
valid_rets = strategy_returns[burnin:n-1]

if len(valid_rets) > 20:
    ann_ret = np.mean(valid_rets) * 252
    ann_std = np.std(valid_rets) * np.sqrt(252)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0.0
    total_yield = np.prod(1.0 + valid_rets) - 1.0
else:
    total_yield = 0.0
    sharpe = 0.0

print("RESULT_YIELD: {:.4f}".format(total_yield))
print("RESULT_SHARPE: {:.4f}".format(sharpe))
import pandas as pd
import numpy as np
import sqlite3

# ====================== DATA LOADING ======================
conn = sqlite3.connect('src/data/market_data.db')
df = pd.read_sql_query("SELECT * FROM core_market_table", conn)
conn.close()

df.columns = [c.upper() for c in df.columns]

# Robust target selection (prefer CLOSE or PRICE over first column)
price_cols = [col for col in df.columns if col in ('CLOSE', 'PRICE', 'ADJ_CLOSE', 'VALUE')]
target = price_cols[0] if price_cols else df.columns[1] if len(df.columns) > 1 else df.columns[0]

# ====================== FEATURE ENGINEERING (NO LOOKAHEAD) ======================
df = df.apply(pd.to_numeric, errors='coerce')

# Forward/backward fill (safe after numeric conversion)
df = df.ffill().bfill()

# Log returns
df['RET'] = np.log(df[target] / df[target].shift(1))

# IMPORTANT: All predictive features must use only information available at t-1
df['RET_L1'] = df['RET'].shift(1)
df['RET_L2'] = df['RET'].shift(2)

df['SIGMA21'] = df['RET'].rolling(window=21, min_periods=10).std().shift(1)
df['SKEW10']  = df['RET'].rolling(window=10,  min_periods=5).skew().shift(1)
df['DELTA_VOL'] = df['SIGMA21'].diff()

# Fill NaNs after shifting (burn-in period)
df['RET'] = df['RET'].fillna(0.0)
for col in ['RET_L1', 'RET_L2', 'SIGMA21', 'SKEW10', 'DELTA_VOL']:
    df[col] = df[col].fillna(0.0)

# ====================== MODEL PARAMETERS ======================
feature_cols = ['RET_L1', 'RET_L2', 'SIGMA21', 'SKEW10', 'DELTA_VOL']
X = df[feature_cols].values
rets = df['RET'].values
sigmas = df['SIGMA21'].values

n = len(df)
K = 4
lam = 0.96          # decay factor
burnin = 500

if n <= burnin + 10:
    res_yield = 0.0
    res_sharpe = 0.0
else:
    # Initialize GMM-style parameters
    pi = np.full(K, 1.0 / K)
    mu = np.zeros((K, len(feature_cols)))
    Sigma = np.array([np.eye(len(feature_cols)) * 0.01 for _ in range(K)])
    
    nk = np.full(K, 10.0)
    m_k = np.zeros(K)      # mean return per cluster
    v_k = np.ones(K) * 0.0001

    pos = np.zeros(n)

    for t in range(burnin, n - 1):
        x_t = X[t].reshape(1, -1)
        
        # E-step: compute responsibilities
        gamma = np.zeros(K)
        for k in range(K):
            diff = x_t - mu[k]
            sig_k = Sigma[k] + 1e-6 * np.eye(len(feature_cols))
            inv = np.linalg.inv(sig_k)
            mah = np.dot(diff, np.dot(inv, diff.T))[0, 0]
            det = np.linalg.det(sig_k)
            pdf = np.exp(-0.5 * mah) / np.sqrt((2 * np.pi)**len(feature_cols) * det)
            gamma[k] = pi[k] * pdf
        
        gamma /= (gamma.sum() + 1e-12)

        # Predictive moments across clusters
        m_bar = np.sum(gamma * m_k)
        v_bar = np.sum(gamma * (v_k + (m_k - m_bar)**2))
        
        ic = m_bar / (v_bar + 1e-8)
        scale = 1.0 / np.sqrt(v_bar + 1e-8)
        w = np.tanh(8.0 * ic * scale)

        # Risk control
        if gamma.max() < 0.35:
            w *= 0.3
            
        vol_t = max(sigmas[t], 0.01)
        w *= 0.12 / vol_t                     # volatility target 12%
        pos[t] = np.clip(w, -2.0, 2.0)

        r_next = rets[t + 1]

        # Online parameter update
        old_nk = nk.copy()
        for k in range(K):
            nk[k] = lam * old_nk[k] + (1 - lam) * gamma[k]
            
            # Update feature distribution
            mu[k] = (lam * old_nk[k] * mu[k] + (1 - lam) * gamma[k] * x_t[0]) / nk[k]
            diff = x_t[0] - mu[k]
            Sigma[k] = (lam * (old_nk[k] / nk[k]) * Sigma[k] +
                       (1 - lam) * (gamma[k] / nk[k]) * np.outer(diff, diff) +
                       1e-6 * np.eye(len(feature_cols)))
            
            pi[k] = lam * pi[k] + (1 - lam) * gamma[k]
            
            # Update predictive return distribution
            m_k[k] = (lam * old_nk[k] * m_k[k] + (1 - lam) * gamma[k] * r_next) / nk[k]
            v_k[k] = (lam * old_nk[k] * v_k[k] +
                     (1 - lam) * gamma[k] * (r_next - m_k[k])**2) / nk[k]

        pi /= pi.sum()

    # ====================== PERFORMANCE ======================
    strategy_rets = pos[burnin:n-1] * rets[burnin+1:n]
    
    if len(strategy_rets) > 0:
        cum_yield = np.prod(1 + strategy_rets) - 1
        mean_ret = np.mean(strategy_rets) * 252
        std_ret = np.std(strategy_rets) * np.sqrt(252)
        sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
    else:
        cum_yield = 0.0
        sharpe = 0.0

    res_yield = cum_yield
    res_sharpe = sharpe

print("RESULT_YIELD: {:.4f}".format(res_yield))
print("RESULT_SHARPE: {:.4f}".format(res_sharpe))
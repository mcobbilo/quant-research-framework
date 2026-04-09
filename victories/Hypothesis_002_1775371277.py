import sqlite3
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

conn = sqlite3.connect('src/data/market_data.db')
df = pd.read_sql_query("SELECT * FROM core_market_table", conn)
conn.close()

df.columns = [c.upper() for c in df.columns]
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

df['LOGRET'] = np.log(df['SPY_CLOSE'] / df['SPY_CLOSE'].shift(1))
df['RET'] = df['SPY_CLOSE'].pct_change()
df['RV'] = df['LOGRET'] ** 2
df['RV'] = df['RV'].fillna(0)
df['LOG_RV'] = np.log(df['RV'] + 1e-8)
df['DLOG_RV'] = df['LOG_RV'].diff()
df['RV_AVG20'] = df['RV'].rolling(window=20, min_periods=10).mean()
df['RV_RATIO'] = df['RV'] / df['RV_AVG20']
df['DRV'] = df['RV'].diff()
df['ABS_DRV_OVER_RV'] = np.abs(df['DRV']) / (df['RV'] + 1e-8)
df['SKEW_5'] = df['LOGRET'].rolling(window=5, min_periods=3).skew()

feature_cols = ['RV', 'DLOG_RV', 'RV_RATIO', 'ABS_DRV_OVER_RV', 'SKEW_5']
df = df.dropna(subset=feature_cols).reset_index(drop=True)

n = len(df)
train_size = int(n * 0.6)
X_train = df.iloc[:train_size][feature_cols].values

gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='full', max_iter=200)
gmm.fit(X_train)

X_all = df[feature_cols].values
gammas = gmm.predict_proba(X_all)

means = gmm.means_
rv_means = means[:, 0]
idx = np.argsort(rv_means)
low_idx = idx[0]
norm_idx = idx[1]
cris_idx = idx[2]

gamma_low = gammas[:, low_idx]
gamma_norm = gammas[:, norm_idx]
gamma_cris = gammas[:, cris_idx]

mu_rv = means[:, 0]
mu_rv_low = mu_rv[low_idx]
mu_rv_norm = mu_rv[norm_idx]
mu_rv_cris = mu_rv[cris_idx]

df['GAMMA_LOW'] = gamma_low
df['GAMMA_NORM'] = gamma_norm
df['GAMMA_CRIS'] = gamma_cris

hl = 4
df['GAMMA_LOW_S'] = df['GAMMA_LOW'].ewm(halflife=hl).mean()
df['GAMMA_NORM_S'] = df['GAMMA_NORM'].ewm(halflife=hl).mean()
df['GAMMA_CRIS_S'] = df['GAMMA_CRIS'].ewm(halflife=hl).mean()

s_t = 1.8 * df['GAMMA_LOW_S'] + 0.6 * df['GAMMA_NORM_S'] - 0.4 * df['GAMMA_CRIS_S']
s_bar = s_t.iloc[:train_size].mean()
lambda_ = 2.5
df['M'] = 1.0 + np.tanh(lambda_ * (s_t - s_bar))

df['SIGMA_HAT'] = (df['GAMMA_LOW_S'] * mu_rv_low + 
                   df['GAMMA_NORM_S'] * mu_rv_norm + 
                   df['GAMMA_CRIS_S'] * mu_rv_cris)

sigma_target = 1e-6
df['SQRT_RV'] = np.sqrt(df['RV'] + 1e-8)
df['PI'] = df['M'] * (sigma_target / (df['SIGMA_HAT'] + 1e-8)) * (1.0 / df['SQRT_RV'])
df['PI'] = df['PI'].clip(-1.5, 2.5)

df['STRAT_RET'] = df['PI'].shift(1) * df['RET']
df = df.dropna(subset=['STRAT_RET']).reset_index(drop=True)

cum_ret = (1 + df['STRAT_RET']).cumprod()
final_yield = cum_ret.iloc[-1] - 1.0

strat_rets = df['STRAT_RET']
mean_ret = strat_rets.mean()
std_ret = strat_rets.std()
sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

print("RESULT_YIELD: " + str(final_yield))
print("RESULT_SHARPE: " + str(sharpe))
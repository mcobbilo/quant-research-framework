import sqlite3
import pandas as pd
import numpy as np
from collections import deque

# =============================================================================
# Load and prepare data
# =============================================================================
conn = sqlite3.connect('src/data/market_data.db')
df = pd.read_sql_query("SELECT * FROM core_market_table", conn)
conn.close()

df.columns = [c.upper() for c in df.columns]
df = df.ffill()
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_index().set_index('DATE')          # ensure chronological order

# Use log-price and compute returns
price = df['SPY_CLOSE'].values
X = np.log(price)
n = len(X)

df['RET'] = np.diff(X, prepend=X[0])
df['ABS_RET'] = np.abs(df['RET'])

# Causal EWMA volatility (span=15 → λ≈0.935)
df['SIGMA_HAT'] = (
    df['ABS_RET'].ewm(span=15, adjust=False, ignore_na=True).mean()
    * np.sqrt(252)
)

sigma_arr = df['SIGMA_HAT'].values.copy()

# =============================================================================
# Strategy parameters
# =============================================================================
burnin = 600
gamma = 0.85
lambda_base = 0.965
lambda_adj = 0.018
long_run_lambda = 0.999
precision_exp_alpha = 0.65
precision_exp_w = 0.85
target_vol = 0.10

# State arrays
v = np.zeros(n)
m = np.zeros(n)
alpha_p = np.zeros(n)
beta_p = np.zeros(n)
pi = np.zeros(n)
bar_pi = np.zeros(n)
kappa_arr = np.full(n, 1.5)
w = np.zeros(n)

# Initial conditions
m[0] = X[0]
v[0] = 5.0
alpha_p[0] = 3.0
beta_p[0] = 0.5
pi[0] = (v[0] * alpha_p[0]) / (beta_p[0] + 1e-12)
bar_pi[0] = pi[0]

# Rolling history for percentile (last 252 days only)
pi_history = deque(maxlen=252)

# =============================================================================
# Main recursive loop
# =============================================================================
for t in range(1, n):
    if t < burnin:
        v[t] = v[t-1]
        m[t] = m[t-1]
        alpha_p[t] = alpha_p[t-1]
        beta_p[t] = beta_p[t-1]
        pi[t] = pi[t-1]
        bar_pi[t] = bar_pi[t-1]
        w[t] = 0.0
        pi_history.append(pi[t])
        continue

    kappa_tm1 = kappa_arr[t-1]
    kappa_t = kappa_arr[t]

    # ------------------------------------------------------------------
    # Adaptive kappa from autocorrelation (purely backward-looking)
    # ------------------------------------------------------------------
    if t > 252:
        past_rets = np.diff(X[t-252:t])                     # returns up to t-1
        if len(past_rets) > 5:
            rho = np.corrcoef(past_rets[:-1], past_rets[1:])[0, 1]
            rho = np.clip(rho, 0.05, 0.95)
            k_t = -np.log(rho)
            kappa_arr[t] = np.clip(k_t, 0.75, 2.4)
            kappa_t = kappa_arr[t]

    # ------------------------------------------------------------------
    # Volatility regime detection (all data ≤ t-1)
    # ------------------------------------------------------------------
    vol20 = np.std(df['RET'].iloc[t-20:t].values) * np.sqrt(252) if t > 20 else 0.15

    if t > 252:
        roll_vols = (df['RET'].iloc[t-252:t]
                     .rolling(20, min_periods=5)
                     .std(ddof=0).values * np.sqrt(252))
        vol_med = np.nanmedian(roll_vols)
        high_vol = vol20 > 1.4 * vol_med if not np.isnan(vol_med) else False
    else:
        high_vol = False

    lambda_t = lambda_base - lambda_adj * (1.0 if high_vol else 0.0)

    # ------------------------------------------------------------------
    # Bayesian update (mean + precision)
    # ------------------------------------------------------------------
    eps = (X[t] - X[t-1]) - kappa_tm1 * (m[t-1] - X[t-1])   # prediction error

    v[t] = lambda_t * v[t-1] + 1.0
    m[t] = (lambda_t * v[t-1] * m[t-1] + (eps / kappa_tm1)) / v[t]

    alpha_p[t] = lambda_t * alpha_p[t-1] + 0.5

    # err_term should be the same prediction error (original code had a bug)
    err_term = eps
    beta_p[t] = (lambda_t * beta_p[t-1] +
                 (lambda_t * v[t-1] / (2.0 * v[t])) * (err_term ** 2))

    pi[t] = (v[t] * alpha_p[t]) / (beta_p[t] + 1e-12)
    bar_pi[t] = long_run_lambda * bar_pi[t-1] + (1.0 - long_run_lambda) * pi[t]

    pi_history.append(pi[t])

    # 25th percentile of recent precision values
    pi_25 = np.percentile(pi_history, 25)

    # ------------------------------------------------------------------
    # Signal construction with precision scaling
    # ------------------------------------------------------------------
    alpha_t = kappa_t * (m[t] - X[t])

    if pi[t] > pi_25 + 1e-8:
        alpha_t *= (pi[t] / (bar_pi[t] + 1e-8)) ** precision_exp_alpha
    else:
        alpha_t = 0.0

    sig_t = max(sigma_arr[t], 0.01)
    precision_factor = (pi[t] / (bar_pi[t] + 1e-8)) ** precision_exp_w if bar_pi[t] > 0 else 0.0

    w_t = (alpha_t / (gamma * sig_t ** 2)) * precision_factor

    # Volatility targeting
    port_vol_approx = abs(w_t) * sig_t
    if port_vol_approx > 1e-8:
        w_t *= target_vol / port_vol_approx

    # Minimum signal strength filter
    min_alpha_hurdle = 0.0008 * sig_t
    if abs(alpha_t) < min_alpha_hurdle or pi[t] < pi_25:
        w_t = 0.0

    w[t] = np.clip(w_t, -10.0, 10.0)

# =============================================================================
# Performance (returns realized AFTER signal is formed → no lookahead)
# =============================================================================
dX = np.diff(X)
strat_returns = w[:-1] * dX                     # w[t] multiplies return from t → t+1

# Drop burn-in period
if len(strat_returns) > burnin:
    strat_returns = strat_returns[burnin:]

mean_ret = np.mean(strat_returns)
std_ret = np.std(strat_returns)
total_yield = np.sum(strat_returns)
sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-8 else 0.0

# Reasonable clipping for reporting
total_yield = np.clip(total_yield, -1.0, None)

print("RESULT_YIELD: {:.4f}".format(total_yield))
print("RESULT_SHARPE: {:.4f}".format(sharpe))
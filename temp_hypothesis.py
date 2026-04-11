import sqlite3
import pandas as pd
import numpy as np

# =============================================================================
# Load and prepare data
# =============================================================================
conn = sqlite3.connect("src/data/market_data.db")
df = pd.read_sql_query("SELECT * FROM core_market_table", conn)
conn.close()

df.columns = [c.upper() for c in df.columns]
df.set_index("DATE", inplace=True)
df.sort_index(inplace=True)
df = df.ffill()

schema = ["GLD_CLOSE", "VUSTX_CLOSE"]
y_col = schema[0]
x_col = schema[1]

df = df.dropna(subset=[y_col, x_col])

y = np.log(df[y_col].values)
x = np.log(df[x_col].values)
n = len(y)

# =============================================================================
# Kalman Filter (random-walk state transition)
# =============================================================================
Q = np.diag([1e-5, 1e-5])  # process noise
R = 1e-3  # observation noise
P = np.eye(2) * 1.0  # initial covariance

beta = np.zeros(n)
alpha = np.zeros(n)
e = np.zeros(n)  # innovation (prediction error)
S = np.zeros(n)  # innovation variance

beta[0] = 0.5
alpha[0] = 0.0

for t in range(1, n):
    H = np.array([x[t], 1.0])
    x_pred = np.array([beta[t - 1], alpha[t - 1]])
    P_pred = P + Q

    e[t] = y[t] - np.dot(H, x_pred)
    S[t] = np.dot(H, np.dot(P_pred, H)) + R
    K = np.dot(P_pred, H) / S[t]

    x_upd = x_pred + K * e[t]
    beta[t] = x_upd[0]
    alpha[t] = x_upd[1]

    # Joseph stabilized update (numerically more stable)
    I_KH = np.eye(2) - np.outer(K, H)
    P = np.dot(I_KH, np.dot(P_pred, I_KH.T)) + np.outer(K, K) * R

# =============================================================================
# Regime detection (CAUSAL - no lookahead)
# =============================================================================
lambda_r = 0.97
R_t = pd.Series(e**2).ewm(alpha=(1 - lambda_r), adjust=False).mean().values

# Rolling median must only use past data → shift(1)
median_R252 = (
    pd.Series(R_t).rolling(window=252, min_periods=100).median().shift(1).values
)

# regime[t] = True if volatility is in the low-volatility regime
regime = R_t < (0.65 * median_R252)

# Z-score of innovation
Z = e / np.sqrt(S)

# =============================================================================
# Trading logic (strictly causal)
# =============================================================================
position = np.zeros(n)
in_position = False
curr_sign = 0
hold_days = 0

for t in range(252, n):  # start after burn-in
    prev_z = Z[t - 1]
    prev_regime = regime[t - 1]

    if in_position:
        hold_days += 1
        # Exit conditions use current bar's information (realistic)
        crossed = Z[t] * curr_sign > 0  # mean-reversion completed
        if crossed or not regime[t] or hold_days > 15:
            in_position = False
            curr_sign = 0
            hold_days = 0
        else:
            position[t] = curr_sign
    else:
        # Entry only on previous bar's regime and z-score (no lookahead)
        if prev_regime:
            if prev_z < -1.75:
                curr_sign = 1
                in_position = True
                hold_days = 1
                position[t] = curr_sign
            elif prev_z > 1.75:
                curr_sign = -1
                in_position = True
                hold_days = 1
                position[t] = curr_sign

# =============================================================================
# Performance (volatility-scaled PNL)
# =============================================================================
pnl = np.zeros(n)
for t in range(1, n):
    if position[t - 1] != 0:
        # Volatility targeting: larger notional when predicted variance is low
        notional = 0.0075 / np.sqrt(max(S[t - 1], 1e-8))
        pnl[t] = position[t - 1] * notional * e[t]

returns = pnl
cum_returns = np.cumsum(returns)

result_yield = cum_returns[-1]
result_sharpe = (
    (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
)

print("RESULT_YIELD:  {:.4f}".format(result_yield))
print("RESULT_SHARPE: {:.4f}".format(result_sharpe))

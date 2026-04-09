import sqlite3
import pandas as pd
import numpy as np

# =============================================================================
# Load and prepare data
# =============================================================================
conn = sqlite3.connect('src/data/market_data.db')
df = pd.read_sql_query("SELECT * FROM core_market_table", conn)
conn.close()

df.columns = [c.upper() for c in df.columns]
df = df.ffill().sort_values('DATE').reset_index(drop=True)

if len(df) <= 100:
    print("RESULT_YIELD: 0.0")
    print("RESULT_SHARPE: 0.0")
    # Exit early to avoid running the rest of the script
    # (important for pipeline compatibility)
else:
    # ---------------------------------------------------------------------
    # Target selection (SPY_CLOSE preferred)
    # ---------------------------------------------------------------------
    if 'SPY_CLOSE' in df.columns:
        target = 'SPY_CLOSE'
    else:
        target = df.columns[1]  # fallback

    df['RETURN'] = df[target].pct_change().fillna(0.0)
    r = df['RETURN'].values
    n = len(r)

    # ---------------------------------------------------------------------
    # Kalman + Adaptive Strategy Parameters
    # ---------------------------------------------------------------------
    lambda_adapt = 0.05
    Q = 1e-5
    R = 1e-4
    P = 1e-4
    mu_hat = 0.0

    burnin = 100
    drawdown_control = 0.20

    position_ts = np.zeros(n)
    strategy_returns = np.zeros(n)
    equity_curve = np.ones(n)
    peak = 1.0

    # ---------------------------------------------------------------------
    # Main loop - NO LOOKAHEAD
    # ---------------------------------------------------------------------
    for t in range(1, n):
        # 1. Kalman filter update using information available at time t
        P_pred = P + Q
        mu_pred = mu_hat
        nu = r[t] - mu_pred
        S = P_pred + R
        K = P_pred / S if S > 0 else 0.0

        mu_hat = mu_pred + K * nu
        P = (1 - K) * P_pred

        # Adaptive noise covariance updates
        R = (1 - lambda_adapt) * R + lambda_adapt * max(0.0, nu**2 - P_pred)
        Q = (1 - lambda_adapt) * Q + lambda_adapt * (K**2 * nu**2)

        # 2. Compute signal (alpha) using information up to t
        alpha = mu_hat / np.sqrt(P + R) if (P + R) > 0 else 0.0

        # 3. Realize return from PREVIOUS position (this is critical - no lookahead)
        strategy_returns[t] = position_ts[t - 1] * r[t]
        equity_curve[t] = equity_curve[t - 1] * (1.0 + strategy_returns[t])

        # Update running peak after realizing the return
        if equity_curve[t] > peak:
            peak = equity_curve[t]

        # 4. Decide position to be held into NEXT period (t+1)
        if t >= burnin and t < n - 1:
            current_dd = max(0.0, 1.0 - equity_curve[t] / peak)
            tau = min(1.0, drawdown_control / current_dd) if current_dd > 0 else 1.0

            sigma2 = r[t]**2 + 1e-8
            w = (alpha / sigma2) * tau
            w = np.clip(w, -1.0, 1.0)
            position_ts[t] = w

    # ---------------------------------------------------------------------
    # Performance metrics (only on post-burnin period)
    # ---------------------------------------------------------------------
    valid_returns = strategy_returns[burnin:]

    if len(valid_returns) > 20 and np.std(valid_returns) > 1e-8:
        mean_ret = np.mean(valid_returns)
        std_ret = np.std(valid_returns)
        sharpe = mean_ret / std_ret * np.sqrt(252)
        total_yield = equity_curve[-1] - 1.0
    else:
        sharpe = 0.0
        total_yield = 0.0

    print(f"RESULT_YIELD: {total_yield:.4f}")
    print(f"RESULT_SHARPE: {sharpe:.4f}")
import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm
from typing import Tuple

# =============================================================================
# Configuration
# =============================================================================
SPAN = 252
WINDOW = 120
BURNIN = 500
DT = 1.0 / 252.0
NU = 40.0
LAMBDA_RISK = 8.0
MIN_WINDOW_FOR_FIT = 30
PRIOR_KAPPA = 0.15
PRIOR_MU = 0.0
PRIOR_SIGMA = 0.015


def main() -> None:
    # Load data
    conn = sqlite3.connect('src/data/market_data.db')
    try:
        df = pd.read_sql_query("SELECT * FROM core_market_table", conn)
    finally:
        conn.close()

    df.columns = [c.upper() for c in df.columns]
    df = df.sort_values('DATE').reset_index(drop=True)

    # Identify target column
    close_cols = [col for col in df.columns 
                  if col.startswith('SPY') and col.endswith('CLOSE')]
    asset = close_cols[0] if close_cols else 'SPY_CLOSE'

    if asset not in df.columns:
        print("RESULT_YIELD: 0.0000")
        print("RESULT_SHARPE: 0.0000")
        return

    # Prepare price series (only forward-fill the price column)
    df[asset] = pd.to_numeric(df[asset], errors='coerce')
    df[asset] = df[asset].ffill()

    prices = df[asset].values
    if len(prices) <= BURNIN + 30:
        print("RESULT_YIELD: 0.0000")
        print("RESULT_SHARPE: 0.0000")
        return

    log_prices = np.log(prices)
    
    # Trend filter - causal (adjust=False + ewm is recursive)
    f_t = (pd.Series(log_prices)
           .ewm(span=SPAN, adjust=False)
           .mean()
           .values)
    
    X = log_prices - f_t

    n = len(df)
    kappa_list = np.full(n, PRIOR_KAPPA, dtype=float)
    mu_list = np.full(n, PRIOR_MU, dtype=float)
    sigma_list = np.full(n, PRIOR_SIGMA, dtype=float)

    for t in range(BURNIN, n):
        start = max(t - WINDOW, 0)
        x_win = X[start:t + 1]
        
        if len(x_win) < MIN_WINDOW_FOR_FIT:
            continue

        y = x_win[1:]
        x_lag = x_win[:-1]
        X_mat = sm.add_constant(x_lag, has_constant='add')

        try:
            model = sm.OLS(y, X_mat).fit(disp=0)
            c = model.params[0]
            phi = model.params[1]

            # Prevent invalid parameters
            if phi >= 0.99 or phi <= -0.99 or np.isnan(phi):
                continue

            kappa = -np.log(max(phi, 0.01)) / DT
            mu = c / (1.0 - phi) if abs(phi) < 0.99 else 0.0
            resid_std = np.std(model.resid, ddof=1)
            sigma = resid_std / np.sqrt(DT)

            # Shrinkage toward priors
            kappa_s = 0.6 * kappa + 0.4 * PRIOR_KAPPA
            mu_s = 0.6 * mu + 0.4 * PRIOR_MU
            sigma_s = 0.6 * sigma + 0.4 * PRIOR_SIGMA

            kappa_list[t] = kappa_s
            mu_list[t] = mu_s
            sigma_list[t] = sigma_s

        except:
            # Keep priors on any numerical failure
            continue

    # Forward-fill parameters (safe because we only use past values)
    kappa_s = pd.Series(kappa_list).ffill().values
    mu_s = pd.Series(mu_list).ffill().values
    sigma_s = pd.Series(sigma_list).ffill().values

    # Mean-reversion signal
    alpha = kappa_s * (mu_s - X) * DT
    credibility = NU / (NU + 1.0)
    alpha = alpha * credibility

    # Position sizing
    v_t = (sigma_s ** 2) * DT
    w_t = np.zeros(n)
    w_t[BURNIN:] = alpha[BURNIN:] / (LAMBDA_RISK * np.maximum(v_t[BURNIN:], 1e-8))
    w_t = np.clip(w_t, -1.5, 1.5)

    # Strategy returns - position at t is known at t, applied to next return
    rets = np.diff(log_prices)
    pos = w_t[:len(rets)]                    # w_t[t] → return from t to t+1
    strat_rets = pos * rets

    # Performance from burn-in period
    valid_rets = strat_rets[BURNIN:]
    if len(valid_rets) < 30:
        print("RESULT_YIELD: 0.0000")
        print("RESULT_SHARPE: 0.0000")
        return

    cum_log = np.cumsum(valid_rets)
    terminal_wealth = np.exp(cum_log[-1])
    yield_val = max(float(terminal_wealth - 1.0), 0.0)

    mean_r = np.mean(valid_rets)
    std_r = np.std(valid_rets, ddof=1)
    sharpe_val = (mean_r / std_r * np.sqrt(252)) if std_r > 1e-8 else 0.0

    print(f"RESULT_YIELD: {yield_val:.4f}")
    print(f"RESULT_SHARPE: {sharpe_val:.4f}")


if __name__ == "__main__":
    main()
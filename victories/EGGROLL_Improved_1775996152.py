import numpy as np
import pandas as pd

EGGROLL_PARAMS = {'SEED': 48, 'NUM_PERIODS': 2040, 'INITIAL_PRICE': np.float64(78.97399308078978), 'RETURN_MU': np.float64(0.0007461168281682206), 'RETURN_SIGMA': np.float64(0.010970140761874048), 'BURNIN': 715, 'LOOKBACK': 266, 'MIN_HISTORY': 59, 'LAMBDA_EW': np.float64(0.9327746517416557), 'BETA_MP': np.float64(0.13576167793590282), 'MU_MP': np.float64(0.9303538910213311), 'Q_DIVISOR': 71, 'ALPHA_SMOOTH': np.float64(0.07462953926903132), 'DEFAULT_ALPHA': np.float64(2.0178396393225144), 'MIN_K_POWERLAW': 2, 'MIN_K': 1, 'THRESH_BURN_OFFSET': 104, 'PERCENTILE': np.float64(91.29605835077767), 'VOL_EPS': np.float64(9.24688266937942e-09), 'STABILITY_EPS': np.float64(8.877085322552429e-13), 'ENTROPY_EPS': np.float64(1.2503171790905711e-08), 'PNL_CLIP': np.float64(0.09516015660399448), 'ANNUALIZATION': np.float64(269.62015345453403), 'MIN_STD': np.float64(8.392358689476804e-09), 'SHARPE_CLIP': np.float64(4.860193640047662)}

# =============================================================================
# Synthetic data generation (for reproducibility / standalone execution)
# =============================================================================
np.random.seed(EGGROLL_PARAMS["SEED"])
dates = pd.date_range('2010-01-01', periods=EGGROLL_PARAMS["NUM_PERIODS"], freq='B')
schema = ['ASSET_A', 'ASSET_B', 'ASSET_C', 'ASSET_D', 'ASSET_E']

df = pd.DataFrame(index=range(EGGROLL_PARAMS["NUM_PERIODS"]))
for col in schema:
    prices = EGGROLL_PARAMS["INITIAL_PRICE"] * np.cumprod(1.0 + np.random.normal(EGGROLL_PARAMS["RETURN_MU"], EGGROLL_PARAMS["RETURN_SIGMA"], EGGROLL_PARAMS["NUM_PERIODS"]))
    df[col] = prices

df['DATE'] = dates
df = df.ffill().bfill()
df[schema] = df[schema].apply(pd.to_numeric, errors='coerce')
df.columns = [c.upper() for c in df.columns]
schema = [c.upper() for c in schema]


def run_strategy(df: pd.DataFrame, schema: list) -> tuple:
    df = df.copy()
    burnin = EGGROLL_PARAMS["BURNIN"]
    n = len(df)
    
    if n <= burnin:
        return 0.0, 0.0

    # ---------------------------------------------------------------------
    # Identify close columns
    # ---------------------------------------------------------------------
    close_cols = [col for col in schema if col in df.columns]
    if len(close_cols) < 2:
        close_cols = [col for col in df.columns 
                     if col.endswith('_CLOSE') or 'CLOSE' in col][:5]
    if len(close_cols) < 2:
        raise ValueError("Need at least 2 assets")

    # ---------------------------------------------------------------------
    # Compute log returns - NO lookahead (shift(1) uses only past price)
    # ---------------------------------------------------------------------
    prices = df[close_cols].values
    rets_raw = np.log(prices[1:] / prices[:-1])
    rets_raw = np.nan_to_num(rets_raw, nan=0.0, posinf=0.0, neginf=0.0)
    
    returns = pd.DataFrame(rets_raw, index=df.index[1:], columns=close_cols)
    n_rets = len(returns)                    # = n - 1

    positions = np.zeros((n_rets, len(close_cols)))
    alpha_ewma = 0.0
    s_bar = np.log(float(len(close_cols)))
    lambda_ew = EGGROLL_PARAMS["LAMBDA_EW"]
    zs = np.zeros(n_rets)

    for i in range(burnin, n_rets):
        # Use only data available up to time i (strictly past returns)
        start = max(0, i - EGGROLL_PARAMS["LOOKBACK"])
        ret_lag = returns.iloc[start:i].copy()
        
        if len(ret_lag) < EGGROLL_PARAMS["MIN_HISTORY"]:
            continue

        ret_arr = ret_lag.values
        C = np.cov(ret_arr.T, ddof=0)
        stds = np.sqrt(np.diag(C))
        stds = np.maximum(stds, EGGROLL_PARAMS["VOL_EPS"])
        R = C / np.outer(stds, stds)
        R = np.clip(R, -1.0, 1.0)

        eigvals, eigvecs = np.linalg.eigh(R)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        trace = np.sum(eigvals)
        eigvals = eigvals * len(close_cols) / trace

        # Marchenko-Pastur cleaning
        beta = EGGROLL_PARAMS["BETA_MP"]
        mu_mp = EGGROLL_PARAMS["MU_MP"]
        p = (1.0 - beta) * eigvals + beta * mu_mp
        p = p / np.sum(p)

        q = float(len(close_cols)) / EGGROLL_PARAMS["Q_DIVISOR"]
        lambda_plus = (1.0 + np.sqrt(q)) ** 2
        k = int(np.sum(eigvals > lambda_plus))
        k = max(k, EGGROLL_PARAMS["MIN_K"])

        p_clean = p[:k].copy()
        p_clean = p_clean / np.sum(p_clean)
        S = -np.sum(p_clean * np.log(p_clean + EGGROLL_PARAMS["ENTROPY_EPS"]))

        s_bar = lambda_ew * s_bar + (1.0 - lambda_ew) * S

        # Fit power-law to get alpha
        if k > EGGROLL_PARAMS["MIN_K_POWERLAW"]:
            ranks = np.arange(1, k + 1, dtype=float)
            log_r = np.log(ranks)
            log_e = np.log(eigvals[:k])
            w = eigvals[:k] / np.sum(eigvals[:k])
            A = np.vstack([log_r, np.ones(len(log_r))]).T
            coeffs = np.linalg.lstsq(A * np.sqrt(w[:, np.newaxis]),
                                   log_e * np.sqrt(w), rcond=None)[0]
            alpha = -coeffs[0]
        else:
            alpha = EGGROLL_PARAMS["DEFAULT_ALPHA"]

        delta_alpha = alpha - alpha_ewma
        alpha_ewma = (1.0 - EGGROLL_PARAMS["ALPHA_SMOOTH"]) * alpha_ewma + EGGROLL_PARAMS["ALPHA_SMOOTH"] * alpha
        delta_s = s_bar - S
        bayes_f = float(max(k, 1))
        Z = delta_alpha * delta_s * bayes_f
        zs[i] = Z

        # Adaptive threshold (only on previous realized signals)
        thresh = 0.0
        if i > burnin + EGGROLL_PARAMS["THRESH_BURN_OFFSET"]:
            past_z = zs[burnin:i][zs[burnin:i] != 0]
            if len(past_z) > 0:
                thresh = np.percentile(past_z, EGGROLL_PARAMS["PERCENTILE"])

        if Z > thresh:
            v1 = eigvecs[:, 0].copy()
            w = v1 / (np.sum(np.abs(v1)) + EGGROLL_PARAMS["STABILITY_EPS"])

            # Volatility scaling using most recent return (still within past window)
            latest_ret = ret_lag.iloc[-1].values.copy()
            vol_scale = np.sqrt(np.maximum(latest_ret**2, EGGROLL_PARAMS["VOL_EPS"]))
            w = w / (vol_scale + EGGROLL_PARAMS["STABILITY_EPS"])
            w = w / (np.sum(np.abs(w)) + EGGROLL_PARAMS["STABILITY_EPS"])

            positions[i, :] = w

    # ---------------------------------------------------------------------
    # Out-of-sample PnL (position taken at i is multiplied by return i → i+1)
    # ---------------------------------------------------------------------
    strat_pnl = np.sum(positions * returns.values, axis=1)
    strat_pnl = np.nan_to_num(strat_pnl, nan=0.0)
    strat_pnl = np.clip(strat_pnl, -EGGROLL_PARAMS["PNL_CLIP"], EGGROLL_PARAMS["PNL_CLIP"])

    burn_pnl = strat_pnl[burnin:]
    cum_pnl = np.cumsum(burn_pnl)
    
    total_yield = cum_pnl[-1] if len(cum_pnl) > 0 else 0.0
    mean_pnl = np.mean(burn_pnl)
    std_pnl = np.std(burn_pnl, ddof=0)
    sharpe = (mean_pnl / std_pnl * np.sqrt(EGGROLL_PARAMS["ANNUALIZATION"])) if std_pnl > EGGROLL_PARAMS["MIN_STD"] else 0.0
    sharpe = np.clip(sharpe, -EGGROLL_PARAMS["SHARPE_CLIP"], EGGROLL_PARAMS["SHARPE_CLIP"])

    return total_yield, sharpe


# =============================================================================
# Execution
# =============================================================================
yield_val, sharpe_val = run_strategy(df, schema)

print("RESULT_YIELD: " + str(round(yield_val, 4)))
print("RESULT_SHARPE: " + str(round(sharpe_val, 4)))
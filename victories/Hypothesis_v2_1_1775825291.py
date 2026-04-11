import sqlite3
import pandas as pd
import numpy as np


def run_boula_rap() -> None:
    # Load data
    with sqlite3.connect("src/data/market_data.db") as conn:
        df = pd.read_sql_query("SELECT * FROM core_market_table", conn)

    df.columns = [c.upper() for c in df.columns]
    burnin = 504

    if len(df) <= burnin:
        print("RESULT_YIELD: 0.0")
        print("RESULT_SHARPE: 0.0")
        return

    df = df.sort_values("DATE").reset_index(drop=True)

    # === Target selection ===
    price_cols = [c for c in df.columns if c.endswith("_CLOSE")]
    if not price_cols:
        print("RESULT_YIELD: 0.0")
        print("RESULT_SHARPE: 0.0")
        return
    target = price_cols[0]

    df[target] = pd.to_numeric(df[target], errors="coerce").ffill()
    df["RET"] = df[target].pct_change().fillna(0.0)

    # === Regime detection (NO LOOKAHEAD) ===
    vix_cols = [c for c in df.columns if "VIX_CLOSE" in c]
    if not vix_cols:
        print("RESULT_YIELD: 0.0")
        print("RESULT_SHARPE: 0.0")
        return

    vix_col = vix_cols[0]
    # Use shifted rolling mean → strictly uses only information available at t-1
    vix_rolling_mean = df[vix_col].rolling(window=252, min_periods=1).mean()
    df["REGIME"] = (
        df[vix_col] > vix_rolling_mean.shift(1).fillna(vix_rolling_mean)
    ).astype(int)

    # Extract arrays for speed
    rets = df["RET"].values
    regimes = df["REGIME"].values
    n = len(df)

    # Filter state
    m = np.zeros(n)
    v = np.zeros(n)
    alpha = np.zeros(n)
    precision = np.zeros(n)
    positions = np.zeros(n)

    # Initial conditions
    m[0] = 0.0
    v[0] = 0.125
    precision[0] = 1.0 / v[0]
    alpha[0] = 0.0

    # Regime-dependent parameters
    kappa_base = {0: 22.4 / 252.0, 1: 9.8 / 252.0}
    sigma_base = {0: 0.092 / np.sqrt(252.0), 1: 0.141 / np.sqrt(252.0)}
    tau_base = {0: 0.011, 1: 0.028}

    for t in range(1, n):
        R = int(regimes[t])
        k = kappa_base[R]
        s = sigma_base[R]
        ta = tau_base[R]
        theta = np.exp(-k)

        m_tm1 = m[t - 1]
        v_tm1 = v[t - 1]
        alpha_tm1 = alpha[t - 1]

        # Predictive step
        alpha_hat = m_tm1 + (alpha_tm1 - m_tm1) * theta
        var_pred = v_tm1 * (1 - theta) ** 2 + (s**2 / (2 * k)) * (1 - theta**2)
        var_pred = max(var_pred, 1e-12)

        z_t = alpha_hat / np.sqrt(var_pred)
        z_min = np.sqrt(max(2 * v_tm1 / (s**2 + 1e-12), 1e-8))

        # Position decision using only information up to t-1 (no lookahead)
        if abs(z_t) < z_min or v_tm1 > 0.45 * (s**2):
            positions[t] = 0.0
        else:
            # Realized volatility over past 20 days (strictly before t)
            past_rets = rets[max(0, t - 20) : t]
            vol = max(np.std(past_rets) * np.sqrt(252), 0.01)
            pos = (alpha_hat / (0.12 * vol)) * 0.6
            positions[t] = np.clip(pos, -1.5, 1.5)

        # Bayesian update with observation at t
        prec_new = precision[t - 1] + 1.0 / (ta**2)
        m[t] = (precision[t - 1] * m_tm1 + rets[t] / (ta**2)) / prec_new
        v[t] = 1.0 / prec_new
        precision[t] = prec_new
        alpha[t] = m[t]

    # === Out-of-sample strategy returns (position at t used for return t+1) ===
    strat_rets = np.zeros(n)
    for t in range(burnin, n - 1):
        strat_rets[t + 1] = positions[t] * rets[t + 1]

    sample = strat_rets[burnin:]
    if np.std(sample) < 1e-8:
        print("RESULT_YIELD: 0.0")
        print("RESULT_SHARPE: 0.0")
        return

    cum_wealth = np.cumprod(1.0 + sample)
    cum_wealth = np.maximum(cum_wealth, 0.0)

    result_yield = float(cum_wealth[-1] - 1.0)
    mean_ret = float(np.mean(sample))
    std_ret = float(np.std(sample))
    result_sharpe = float(mean_ret / std_ret * np.sqrt(252.0)) if std_ret > 0 else 0.0

    print("RESULT_YIELD: {:.4f}".format(result_yield))
    print("RESULT_SHARPE: {:.4f}".format(result_sharpe))


if __name__ == "__main__":
    run_boula_rap()

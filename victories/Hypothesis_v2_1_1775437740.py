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
df = df.sort_values("DATE").reset_index(drop=True)

# Select SPY close column
schema = [c for c in df.columns if c.endswith("_CLOSE") and "SPY" in c]
asset = schema[0] if schema else "SPY_CLOSE"

# Ensure numeric and forward-fill
df[asset] = pd.to_numeric(df[asset], errors="coerce")
df = df.ffill()

# =============================================================================
# Parameters
# =============================================================================
burnin = 504
window = 22
kappa, alpha, theta, gamma, beta_p = 0.5, 5.0, 0.8, 1.0, 5.0

n = len(df)
if n <= burnin:
    print("RESULT_YIELD: 0.0000")
    print("RESULT_SHARPE: 0.0000")
else:
    # Work with log prices
    df["LOG_PRICE"] = np.log(df[asset])
    y = df["LOG_PRICE"].values

    # Log returns (return from t-1 to t)
    df["LOGRET"] = df["LOG_PRICE"].diff().fillna(0.0)
    df["SQ_RET"] = df["LOGRET"] ** 2

    # =====================================================================
    # Features - ALL constructed to avoid lookahead bias
    # =====================================================================
    # MU: rolling mean of log price using data up to t-1
    df["MU"] = df["LOG_PRICE"].rolling(window=window, min_periods=1).mean().shift(1)

    # BETA: linear slope of log price over last window days (up to t-1)
    def roll_slope(x: np.ndarray) -> float:
        if len(x) < window:
            return 0.0
        return np.polyfit(np.arange(len(x)), x, 1)[0]

    df["BETA"] = (
        df["LOG_PRICE"]
        .rolling(window=window, min_periods=1)
        .apply(roll_slope, raw=True)
        .shift(1)
    )

    # Rolling statistics of BETA (also lagged)
    df["BETA_STD"] = (
        df["BETA"].rolling(window=window, min_periods=1).std(ddof=0).shift(1)
    )

    df["TAU"] = df["BETA"] / (df["BETA_STD"] + 1e-8)

    # Volatility estimate using past returns only
    f_t = (
        df["LOGRET"]
        .rolling(window=window, min_periods=1)
        .std(ddof=0)
        .shift(1)
        .fillna(1e-4)
    )

    # DELTA uses only information available at t-1
    df["DELTA"] = (df["LOG_PRICE"] - df["MU"]) / f_t

    # =====================================================================
    # Strategy signal
    # =====================================================================
    g1 = np.exp(-kappa * df["DELTA"] ** 2)
    g2 = 1 - 1 / (1 + np.exp(alpha * (np.abs(df["TAU"]) - theta)))
    g3 = 1 / (1 + np.exp(beta_p * (np.abs(df["TAU"]) - theta)))

    df["PHI"] = df["TAU"] * g1 * g2 - gamma * df["DELTA"] * g3

    # Volatility scaling using previous day's squared return
    sigma = np.sqrt(df["SQ_RET"].shift(1).fillna(1e-4))

    df["W"] = df["PHI"] / (sigma + 1e-8)
    df["W"] = df["W"].clip(-1.5, 1.5)

    # Position for day t is decided at close of day t-1
    df["POS"] = df["W"].shift(1).fillna(0.0)

    # Strategy returns
    df["STRAT"] = df["POS"] * df["LOGRET"]

    # =====================================================================
    # Performance (only after burn-in period)
    # =====================================================================
    strat_rets = df["STRAT"].iloc[burnin:].values
    cumw = (1 + strat_rets).cumprod()
    total_return = cumw[-1] - 1.0

    # Annualized yield
    n_periods = len(strat_rets)
    ann_yield = (cumw[-1] ** (252.0 / n_periods)) - 1.0 if n_periods > 0 else 0.0

    # Annualized Sharpe
    if strat_rets.std(ddof=0) > 1e-8:
        ann_sharpe = (strat_rets.mean() / strat_rets.std(ddof=0)) * np.sqrt(252)
    else:
        ann_sharpe = 0.0

    print("RESULT_YIELD: {:.4f}".format(ann_yield))
    print("RESULT_SHARPE: {:.4f}".format(ann_sharpe))

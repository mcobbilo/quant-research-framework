import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("src/data/market_data.db")
df = pd.read_sql_query("SELECT * FROM core_market_table", conn)
conn.close()
df.columns = [c.upper() for c in df.columns]
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").set_index("DATE")

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.ffill().bfill()

univ_cols = [
    "SPY_CLOSE",
    "IWM_CLOSE",
    "GLD_CLOSE",
    "RSP_CLOSE",
    "VUSTX_CLOSE",
    "CL_CLOSE",
    "HG_CLOSE",
]
univ = [c for c in univ_cols if c in df.columns]
prices = df[univ].copy()

if len(prices.columns) < 3 or len(prices) < 200:
    print("RESULT_YIELD: 0.0000")
    print("RESULT_SHARPE: 0.0000")
else:
    returns = prices.pct_change()

    W = 126
    K = 5
    L = 504
    kappa = 1.0
    sigma_target = 0.01

    dates = df.index
    n = len(dates)
    beta_series = pd.Series(np.nan, index=dates, dtype=float)
    lambda1_series = pd.Series(np.nan, index=dates, dtype=float)

    for i in range(W, n):
        window_ret = returns.iloc[i - W : i].copy()
        if window_ret.shape[1] < 3 or window_ret.shape[0] < W:
            continue
        R = (
            window_ret.apply(lambda x: (x - x.mean()) / x.std(ddof=0), axis=0)
            .fillna(0)
            .copy()
        )
        corr_mat = R.corr().values.copy()
        eigvals = np.sort(np.linalg.eigvalsh(corr_mat))[::-1]
        eigvals = np.maximum(eigvals, 1e-8)
        lambda1_series.iloc[i] = eigvals[0]
        ks = np.arange(1, min(K, len(eigvals)) + 1)
        log_lam = np.log(eigvals[: len(ks)])
        if len(ks) >= 2:
            slope, _ = np.polyfit(ks, log_lam, 1)
            beta_series.iloc[i] = -slope

    lambda1_series = lambda1_series.ffill()
    beta = beta_series.ffill()
    ewma_fast = beta.ewm(alpha=0.12, adjust=False).mean()
    ewma_slow = beta.ewm(alpha=0.03, adjust=False).mean()
    delta_beta = ewma_fast - ewma_slow
    Z = (delta_beta - delta_beta.rolling(L).mean()) / delta_beta.rolling(L).std()

    spy_close = df["SPY_CLOSE"].copy()
    spy_ret = spy_close.pct_change()
    spy_vol = spy_ret.rolling(21).std()

    pos = -np.tanh(kappa * Z)
    vol_scale = sigma_target / spy_vol.replace(0, np.nan)
    w = (pos * vol_scale).fillna(0)

    beta_10 = beta.rolling(252).quantile(0.1)
    filt = ((lambda1_series > 1.8) & (lambda1_series < 5.5) & (beta > beta_10)).astype(
        float
    )
    filt = filt.ffill().fillna(0)
    w = w * filt

    strat_ret = w.shift(1) * spy_ret
    strat_ret = strat_ret.dropna()

    if len(strat_ret) == 0:
        print("RESULT_YIELD: 0.0000")
        print("RESULT_SHARPE: 0.0000")
    else:
        total_yield = (1 + strat_ret).cumprod().iloc[-1] - 1
        ann_factor = 252.0
        mean_ret = strat_ret.mean() * ann_factor
        std_ret = strat_ret.std() * np.sqrt(ann_factor)
        sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

        print("RESULT_YIELD: {:.4f}".format(total_yield))
        print("RESULT_SHARPE: {:.4f}".format(sharpe))

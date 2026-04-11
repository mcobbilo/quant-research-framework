import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import sqlite3


def get_frac_diff_weights(d, num_terms):
    w = np.zeros(num_terms)
    w[0] = 1.0
    for k in range(1, num_terms):
        w[k] = w[k - 1] * (d - k + 1) / k * (-1.0)
    return w


def find_min_d(log_prices, num_terms=252, critical_value=-2.86):
    if len(log_prices) < num_terms * 2:
        return 0.4
    d_candidates = np.arange(0.05, 1.01, 0.05)
    min_d = 1.0
    n = len(log_prices)
    for d in d_candidates:
        w = get_frac_diff_weights(d, num_terms)
        y = np.full(n, np.nan)
        for i in range(num_terms, n):
            start = i - num_terms + 1
            segment = log_prices[start : i + 1][::-1]
            y[i] = np.dot(w[: len(segment)], segment)
        y_valid = y[num_terms:]
        if len(y_valid) < 50:
            continue
        adf_stat = adfuller(y_valid, maxlag=1, regression="c", autolag=None)[0]
        if adf_stat < critical_value:
            min_d = d
            break
    return min_d


conn = sqlite3.connect("src/data/market_data.db")
df = pd.read_sql_query("SELECT * FROM core_market_table", conn)
conn.close()

if df.empty or len(df) < 300:
    print("RESULT_YIELD: 0.0000")
    print("RESULT_SHARPE: 0.0000")
else:
    df.columns = [c.upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").set_index("DATE")
    df["LOGP"] = np.log(df["SPY_CLOSE"])
    logp_vals = df["LOGP"].values
    n = len(df)

    d_star = find_min_d(logp_vals)
    num_terms = 252
    w = get_frac_diff_weights(d_star, num_terms)

    Y = np.full(n, np.nan)
    for i in range(num_terms, n):
        start = i - num_terms + 1
        segment = logp_vals[start : i + 1][::-1]
        Y[i] = np.dot(w[: len(segment)], segment)
    df["Y"] = Y

    roll_window = 60
    df["Y_MEAN"] = df["Y"].rolling(roll_window, min_periods=10).mean()
    df["Y_STD"] = df["Y"].rolling(roll_window, min_periods=10).std()
    df["Z"] = (df["Y"] - df["Y_MEAN"]) / df["Y_STD"]

    df["POS_RAW"] = -df["Z"].shift(1)
    df["POS"] = df["POS_RAW"].clip(-1.5, 1.5)
    df["RET"] = df["LOGP"].diff()
    df["STRAT_RET"] = df["POS"] * df["RET"]

    df["ROLL_STD"] = df["STRAT_RET"].rolling(roll_window, min_periods=10).std()
    df["ANN_VOL"] = df["ROLL_STD"] * np.sqrt(252)
    target_vol = 0.12
    df["LEV"] = target_vol / df["ANN_VOL"].replace(0, np.nan)
    df["LEV"] = df["LEV"].fillna(1.0).clip(0.2, 3.0)
    df["FINAL_RET"] = df["STRAT_RET"] * df["LEV"].shift(1)

    perf_df = df.iloc[500:].copy()
    perf_df = perf_df[perf_df["FINAL_RET"].notna()]

    if len(perf_df) == 0:
        print("RESULT_YIELD: 0.0000")
        print("RESULT_SHARPE: 0.0000")
    else:
        total_log = perf_df["FINAL_RET"].sum()
        result_yield = np.exp(total_log) - 1
        mean_ret = perf_df["FINAL_RET"].mean()
        std_ret = perf_df["FINAL_RET"].std()
        if std_ret > 0:
            sharpe = mean_ret / std_ret * np.sqrt(252)
        else:
            sharpe = 0.0
        print("RESULT_YIELD: {:.4f}".format(result_yield))
        print("RESULT_SHARPE: {:.4f}".format(sharpe))

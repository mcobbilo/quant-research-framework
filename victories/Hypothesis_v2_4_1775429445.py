import sqlite3
import pandas as pd
import numpy as np


def run_structural_alpha_backtest():
    try:
        conn = sqlite3.connect("src/data/market_data.db")
        df = pd.read_sql_query("SELECT * FROM core_market_table", conn)
        conn.close()

        df.columns = [c.upper() for c in df.columns]
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.set_index("DATE")

        asset_cols = [
            "SPY_CLOSE",
            "IWM_CLOSE",
            "GLD_CLOSE",
            "CL_CLOSE",
            "VUSTX_CLOSE",
            "HG_CLOSE",
        ]
        prices = df[asset_cols].apply(pd.to_numeric, errors="coerce")

        if prices.empty or len(prices) < 200:
            print("RESULT_YIELD: 0.0000")
            print("RESULT_SHARPE: 0.0000")
            return

        prices = prices.ffill().dropna()
        if len(prices) < 200:
            print("RESULT_YIELD: 0.0000")
            print("RESULT_SHARPE: 0.0000")
            return

        returns = np.log(prices / prices.shift(1)).dropna().copy()

        if len(returns) < 150:
            print("RESULT_YIELD: 0.0000")
            print("RESULT_SHARPE: 0.0000")
            return

        W = 126
        tau = 21
        lam = np.exp(-np.log(2) / tau)
        M = 3
        alpha = 0.15
        gamma = 0.75
        theta = 0.4
        phi = 0.3

        n_t = len(returns)
        deltas = np.zeros(n_t)
        vs = np.zeros(n_t)
        signals = np.zeros(n_t)
        tilde = 0.0
        v_list = []

        ret_values = returns.values
        spy_rets = ret_values[:, 0]

        for i in range(W, n_t):
            window = ret_values[i - W : i]
            k = np.arange(W - 1, -1, -1)
            weights = lam**k
            weights = weights / weights.sum()
            mu = np.average(window, axis=0, weights=weights)
            demeaned = window - mu
            cov_mat = np.dot(demeaned.T, demeaned * weights[:, np.newaxis])

            eigvals = np.linalg.eigvalsh(cov_mat)
            eigvals = np.sort(eigvals)[::-1]
            eigvals = eigvals[eigvals > 1e-8]

            if len(eigvals) < M:
                continue

            x = np.arange(M)
            y = np.log(eigvals[:M])
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            delta = -slope
            deltas[i] = delta

            tilde_new = alpha * delta + (1 - alpha) * tilde
            v = tilde_new - tilde
            vs[i] = v
            tilde = tilde_new
            v_list.append(v)

            if len(v_list) > 20:
                sigma_v = np.std(v_list[-20:])
            else:
                sigma_v = np.std(v_list) if len(v_list) > 0 else 1.0

            if sigma_v == 0 or np.isnan(sigma_v):
                sigma_v = 1.0

            ev_ratio = eigvals[0] / np.sum(eigvals)
            f_val = min(1.0, max(0.2, (ev_ratio - theta) / phi))
            v_norm = v / sigma_v
            s = -np.sign(v) * (np.abs(v_norm) ** gamma) * f_val
            signals[i] = s

        valid_signals = signals[W:]
        valid_rets = spy_rets[W:]

        if len(valid_signals) == 0:
            print("RESULT_YIELD: 0.0000")
            print("RESULT_SHARPE: 0.0000")
            return

        strat_rets = valid_signals * valid_rets
        cum_yield = np.sum(strat_rets)
        n_years = len(strat_rets) / 252.0
        result_yield = cum_yield / n_years if n_years > 0 else 0.0
        std = np.std(strat_rets)
        mean_ret = np.mean(strat_rets)
        sharpe = (mean_ret / std * np.sqrt(252)) if std > 1e-8 else 0.0

        print("RESULT_YIELD: {:.4f}".format(result_yield))
        print("RESULT_SHARPE: {:.4f}".format(sharpe))

    except Exception:
        print("RESULT_YIELD: 0.0000")
        print("RESULT_SHARPE: 0.0000")


if __name__ == "__main__":
    run_structural_alpha_backtest()

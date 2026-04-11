import pandas as pd
import numpy as np
import sqlite3
import xgboost as xgb
import sys


def custom_asymmetric_objective(y_true, y_pred, gamma=10.0, tau=5.0):
    y_true = np.asarray(y_true).astype(np.float64)
    y_pred = np.asarray(y_pred).astype(np.float64)
    grad = np.zeros_like(y_true, dtype=np.float64)
    hess = np.zeros_like(y_true, dtype=np.float64)
    pos = y_true >= 0
    neg = y_true < 0
    grad[pos] = -2.0 * (y_true[pos] - y_pred[pos])
    hess[pos] = 2.0
    grad[neg] = -2.0 * gamma * (y_true[neg] - y_pred[neg])
    hess[neg] = 2.0 * gamma
    if tau > 0 and np.any(neg):
        penalty_mask = (y_pred[neg] > 0).astype(np.float64)
        penalty = tau * np.abs(y_true[neg]) * penalty_mask
        grad[neg] += penalty
    return grad, hess


# Security & Isolation: only allowed DB access
db_path = "src/data/market_data.db"
if any(
    forbidden in db_path.lower() for forbidden in ["..", "/etc", "/var", "root", "sys"]
) or db_path.startswith("/"):
    print("ERROR: Path traversal detected")
    sys.exit(1)

conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM core_market_table", conn)
conn.close()

df.columns = [c.upper() for c in df.columns]
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").set_index("DATE").copy()

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.ffill().bfill()

df["RETURN"] = df["SPY_CLOSE"].pct_change()
df["TARGET"] = df["RETURN"].shift(-1)
df = df.dropna(subset=["TARGET"]).copy()

if len(df) < 20:
    print("RESULT_YIELD: 0.0000")
    print("RESULT_SHARPE: 0.0000")
    sys.exit(0)

feature_list = [
    "VIX_CLOSE",
    "MOVE_CLOSE",
    "VIX_TNX_RATIO",
    "SPY_PCT_FROM_200",
    "SPY_PPO_HIST",
    "VIX_TERM_STRUCTURE_3M",
    "SKEW_CLOSE",
    "SPY_ATR_PCT",
    "SPY_VOL_RATIO_21_252",
    "CORR_SPY_VUSTX_63_ZSCORE",
    "VIX_VVIX_RATIO_Z",
    "SPY_ACCEL_MOM",
    "VIX_SMA20",
    "VIX_BB_WIDTH",
    "SPY_PCT_FROM_DONCHIAN_UPPER",
    "T10Y2Y",
    "TEDRATE",
    "NFCI",
]
feature_cols = [f for f in feature_list if f in df.columns]
if len(feature_cols) < 5:
    feature_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if c not in ["RETURN", "TARGET"]
    ][:15]

n = len(df)
train_size = int(n * 0.65)
if train_size < 10:
    train_size = 10
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

gammas = [1.0, 3.0, 6.0, 10.0, 15.0, 25.0, 40.0]
models = []
for i, gamma in enumerate(gammas):
    tau = 4.0 + (i % 3)

    def make_obj(g=gamma, t=tau):
        def obj(preds, dtrain):
            labels = dtrain.get_label()
            return custom_asymmetric_objective(labels, preds, g, t)

        return obj

    dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df["TARGET"])
    model = xgb.train(
        {
            "max_depth": 4,
            "eta": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        },
        dtrain,
        num_boost_round=60,
        obj=make_obj(),
        verbose_eval=False,
    )
    models.append(model)

test_df = test_df.copy()
preds = np.zeros(len(test_df))
for model in models:
    dtest = xgb.DMatrix(test_df[feature_cols])
    preds += model.predict(dtest)
preds /= len(models)
test_df["PRED"] = preds

test_df["VOL"] = test_df["RETURN"].rolling(20, min_periods=5).std().shift(1)
test_df["VOL"] = test_df["VOL"].fillna(test_df["VOL"].mean()).replace(0, 0.01)

lambda_param = 2.0
test_df["SIG"] = np.tanh(lambda_param * test_df["PRED"] / (test_df["VOL"]))

cum_ret = (1 + test_df["RETURN"]).cumprod()
running_peak = cum_ret.cummax().shift(1).fillna(1.0)
dd = ((cum_ret / running_peak) - 1.0).shift(1).fillna(0.0)
phi = np.maximum(1.0 - 0.6 * (-dd), 0.3)
test_df["POSITION"] = test_df["SIG"] * phi

test_df["STRAT_RETURN"] = test_df["POSITION"] * test_df["RETURN"]
test_df["STRAT_RETURN"] = test_df["STRAT_RETURN"].fillna(0)

total_yield = (1 + test_df["STRAT_RETURN"]).prod() - 1
std_ret = test_df["STRAT_RETURN"].std()
sharpe = (
    (test_df["STRAT_RETURN"].mean() / std_ret * np.sqrt(252)) if std_ret > 1e-8 else 0.0
)

print("RESULT_YIELD: {:.4f}".format(total_yield))
print("RESULT_SHARPE: {:.4f}".format(sharpe))

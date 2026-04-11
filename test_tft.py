import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import warnings

warnings.filterwarnings("ignore")

df = pd.read_parquet("/Users/milocobb/tft_model/clean_aligned_features_27yr.parquet")
df.columns = [c.replace(".", "_") for c in df.columns]
if "Kronos_S1" in df.columns:
    df["Kronos_S1"] = df["Kronos_S1"].astype(str)
    df["Kronos_S2"] = df["Kronos_S2"].astype(str)
if df.index.name == "Date":
    df = df.reset_index()

df["SPY_Log_Return"] = np.log(df["SPY"] / df["SPY"].shift(1)).fillna(0)
df["Daily_Return"] = df["SPY"].pct_change().fillna(0)
df = df.fillna(0)
df["time_idx"] = np.arange(len(df))
df["group_id"] = "portfolio"

exclude_f = [
    "time_idx",
    "group_id",
    "day_of_year_sin",
    "month",
    "Date",
    "Daily_Return",
    "SPY_Log_Return",
    "Kronos_S1",
    "Kronos_S2",
]

reals = [c for c in df.columns if c not in exclude_f and not c.startswith("target")]
for c in reals:
    if not pd.api.types.is_numeric_dtype(df[c]):
        print(f"Non-numeric real column: {c}")

try:
    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="SPY_Log_Return",
        group_ids=["group_id"],
        max_encoder_length=252,
        max_prediction_length=21,
        time_varying_unknown_reals=reals,
        time_varying_known_reals=["day_of_year_sin", "month"]
        if "day_of_year_sin" in df.columns
        else [],
        categorical_encoders={
            "Kronos_S1": NaNLabelEncoder(add_nan=True),
            "Kronos_S2": NaNLabelEncoder(add_nan=True),
        }
        if "Kronos_S1" in df.columns
        else {},
        time_varying_unknown_categoricals=[
            c for c in ["Kronos_S1", "Kronos_S2"] if c in df.columns
        ],
        add_relative_time_idx=True,
    )
    print("Success training setup")
    oos_start_idx = df[df["Date"] >= "2025-12-30"]["time_idx"].min() - 10
    dataset = TimeSeriesDataSet.from_dataset(
        training, df, min_prediction_idx=oos_start_idx
    )
    print(f"Success dataset from_dataset, size={len(dataset)}")
except Exception:
    import traceback

    traceback.print_exc()

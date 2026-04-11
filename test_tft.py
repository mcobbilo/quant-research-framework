import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet

df = pd.read_parquet("/Users/milocobb/tft_model/clean_aligned_features_27yr.parquet")
df.columns = [c.replace(".", "_") for c in df.columns]  
df = df.fillna(0)  

df["time_idx"] = np.arange(len(df))  
df["group_id"] = "portfolio"         

max_prediction_length = 21
training_cutoff = df["time_idx"].max() - max_prediction_length - 126 

training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="target_SPY_fwd21",          
    group_ids=["group_id"],
    max_encoder_length=252,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=[c for c in df.columns if c not in ["time_idx","group_id", "day_of_year_sin", "month"] and not c.startswith("target")],
    time_varying_known_reals=["day_of_year_sin", "month"] if "day_of_year_sin" in df.columns else [],
    add_relative_time_idx=True,
)

validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training_cutoff + 1)

print("Training size:", len(training))
print("Validation size:", len(validation))

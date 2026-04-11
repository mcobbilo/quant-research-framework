import pandas as pd
import numpy as np
import torch
import warnings
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

warnings.filterwarnings("ignore")

def extract_tft_feature_importance():
    print("Loading pristine Market Data...")
    df = pd.read_parquet("/Users/milocobb/tft_model/clean_aligned_features_27yr.parquet")
    df.columns = [c.replace(".", "_") for c in df.columns]
    df = df.fillna(0)
    
    if df.index.name == 'Date':
        df = df.reset_index()

    if 'SPY' in df.columns:
        df['SPY_Log_Return'] = np.log(df['SPY'] / df['SPY'].shift(1)).fillna(0)
        df['Daily_Return'] = df['SPY'].pct_change().fillna(0)

    if 'directional_impact' in df.columns:
        df['fomc_regime'] = df['directional_impact'].ewm(span=21, adjust=False).mean()

    df["time_idx"] = np.arange(len(df))
    df["group_id"] = "portfolio"
    
    # Re-instantiate exactly as trained
    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="SPY_Log_Return",          
        group_ids=["group_id"],
        max_encoder_length=252,
        max_prediction_length=21,
        time_varying_unknown_reals=[c for c in df.columns if c not in ["time_idx","group_id", "day_of_year_sin", "month", "Date", "Daily_Return"] and not c.startswith("target") and c != "SPY_Log_Return"],
        time_varying_known_reals=["day_of_year_sin", "month"] if "day_of_year_sin" in df.columns else [],
        add_relative_time_idx=True,
    )

    print("Loading internal weights from Epoch 12...")
    ckpt_path = "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/lightning_logs/version_1162/checkpoints/epoch=12-step=3055.ckpt"
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)

    print("Extracting Neural Feature Importance via Variable Selection Networks...")
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    # We pull a highly comprehensive validation subset representing different regimes
    # Specifically the last 500 batches to get modern macro importance
    dataset = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=len(df)-128)
    dataloader = dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    # Accumulate interpretation
    all_encoder_importances = []
    
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            # Interpret the raw tensor output
            interpretation = model.interpret_output(out, reduction="sum")
            
            # Encoder variables
            enc_vars = interpretation["encoder_variables"].numpy()
            all_encoder_importances.append(enc_vars)
            
    # Aggregate across the subset
    agg_encoder = np.sum(all_encoder_importances, axis=0)
    
    # Normalize to 100%
    agg_encoder = (agg_encoder / agg_encoder.sum()) * 100
    
    # Map to variable names (PyTorch Forecasting tracks exact indices)
    encoder_features = model.encoder_variables
    
    importance_df = pd.DataFrame({
        "Feature": encoder_features,
        "Importance_Score": agg_encoder
    }).sort_values(by="Importance_Score", ascending=False).reset_index(drop=True)
    
    print("\n=============================================")
    print(" TFT NATIVE VARIABLE IMPORTANCE (TOP 20) ")
    print("=============================================")
    for idx, row in importance_df.head(20).iterrows():
        print(f"{idx+1:2d}. {row['Importance_Score']:5.2f}%  |  {row['Feature']}")
        
    print("\n(Note: Similar to SHAP, this maps the explicit mathematical attention ")
    print("assigned by the Variable Selection Network.)")

if __name__ == '__main__':
    extract_tft_feature_importance()

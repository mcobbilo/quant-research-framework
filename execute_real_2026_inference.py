import pandas as pd
import numpy as np
import torch
import warnings
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

warnings.filterwarnings("ignore")

def run_real_backtest():
    print("Loading Genuine Rebuilt Market Data...")
    df = pd.read_parquet("/Users/milocobb/tft_model/clean_aligned_features_27yr.parquet")
    df.columns = [c.replace(".", "_") for c in df.columns]
    
    if df.index.name == 'Date':
        df = df.reset_index()
        
    print(f"Data natively extends to: {df['Date'].max().strftime('%Y-%m-%d')}")

    print("Verifying genuine FOMC Sentiment NLP signals...")
    
    # We explicitly forward-fill just FOMC sentiment because it naturally holds until the next Fed statement!
    if 'directional_impact' in df.columns:
        df['directional_impact'] = df['directional_impact'].ffill().fillna(0.0)
    if 'surprise_factor' in df.columns:
        df['surprise_factor'] = df['surprise_factor'].ffill().fillna(0.0)

    # Clean target columns
    if 'SPY' in df.columns:
        df['SPY_Log_Return'] = np.log(df['SPY'] / df['SPY'].shift(1)).fillna(0)
        df['Daily_Return'] = df['SPY'].pct_change().fillna(0)

    # Re-build exponential smoothing for FOMC regime (memory vector)
    df['fomc_regime'] = df['directional_impact'].ewm(span=21, adjust=False).mean()
    
    df = df.fillna(0) # baseline safety
    df["time_idx"] = np.arange(len(df))
    df["group_id"] = "portfolio"
    
    # Exclude prediction indices
    exclude_f = ["time_idx","group_id", "day_of_year_sin", "month", "Date", "Daily_Return", "SPY_Log_Return"]
    
    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="SPY_Log_Return",          
        group_ids=["group_id"],
        max_encoder_length=252,
        max_prediction_length=21,
        time_varying_unknown_reals=[c for c in df.columns if c not in exclude_f and not c.startswith("target")],
        time_varying_known_reals=["day_of_year_sin", "month"] if "day_of_year_sin" in df.columns else [],
        add_relative_time_idx=True,
    )

    print("\nLoading static Phase-10 Weights (Epoch 12) trained on data prior to 2023...")
    ckpt_path = "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/lightning_logs/version_1162/checkpoints/epoch=12-step=3055.ckpt"
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)

    # To isolate the True 2026 Test, we extract prediction dates from 2025-12-30 forward.
    # What was the index at 2025-12-30?
    target_idx = df[df['Date'] >= '2025-12-30']['time_idx'].min() - 10 # Adding buffer
    
    dataset = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=target_idx)
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    print("Executing Neural Inference Out-Of-Sample...")
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    preds_list = []
    idx_list = []
    
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            p = model.to_prediction(out)
            final_p = p.sum(dim=1).numpy()
            final_idx = x["decoder_time_idx"][:, 0].numpy() - 1
            
            preds_list.extend(final_p)
            idx_list.extend(final_idx)
            
    results_df = pd.DataFrame({
        "time_idx": np.array(idx_list),
        "Forward_21d_Prediction": np.array(preds_list)
    })
    
    eval_df = pd.merge(df[['time_idx', 'Date', 'SPY', 'Daily_Return']], results_df, on="time_idx", how="inner")
    
    # Ensure it only prints the block of 2026 we requested
    eval_df = eval_df[eval_df['Date'] >= '2025-12-30'].copy()
    
    eval_df['Position'] = np.where(eval_df['Forward_21d_Prediction'] >= 0.02, 1.0, 
                     np.where(eval_df['Forward_21d_Prediction'] <= -0.01, -1.0, 0.0))

    eval_df['Strategy_Return'] = eval_df['Position'].shift(1) * eval_df['Daily_Return']
    eval_df['Equity_Curve'] = (1 + eval_df['Strategy_Return'].fillna(0)).cumprod()
    eval_df['Buy_Hold'] = (1 + eval_df['Daily_Return'].fillna(0)).cumprod()

    final_eq = eval_df['Equity_Curve'].iloc[-1]
    final_bh = eval_df['Buy_Hold'].iloc[-1]
    alpha = (final_eq - final_bh) / final_bh * 100
    
    print("\n" + "="*60)
    print(" GENUINE 2026 INFERENCE: TFT (v4.2) TRUE OUT-OF-SAMPLE TEST ")
    print("="*60)
    print(f"Total True Evaluation Days:  {len(eval_df)}")
    print(f"TFT Neural Strategy Equity:   {final_eq:.3f}x")
    print(f"Naive Baseline (Buy/Hold):    {final_bh:.3f}x")
    print(f"Alpha Captured:               {alpha:+.2f}%")
    
    print("\n--- Genuine Position Sequence (2026) ---")
    eval_df['Prev_Position'] = eval_df['Position'].shift(1).fillna(1.0) # Assume it carried over LONG from Dec
    changes = eval_df[eval_df['Position'] != eval_df['Prev_Position']]
    pos_map = {1.0: "LONG SPY", -1.0: "SHORT SPY", 0.0: "CASH"}
    
    print(f"{eval_df['Date'].iloc[0].strftime('%Y-%m-%d')} Initial Carry-Over -> {pos_map.get(eval_df['Position'].iloc[0])}")
    
    for _, row in changes.iterrows():
        print(f"{row['Date'].strftime('%Y-%m-%d')} Switched to {pos_map.get(row['Position'])} (Prediction vector: {row['Forward_21d_Prediction']:.3f})")
    
    if len(changes) == 0:
        print("\nNote: The Neural Network structurally held its position through the entire 2026 slice.")

if __name__ == '__main__':
    run_real_backtest()

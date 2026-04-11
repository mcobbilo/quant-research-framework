import pandas as pd
import numpy as np
import yfinance as yf
import torch
import warnings
import tqdm
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

warnings.filterwarnings("ignore")

def run_extended_backtest():
    print("Loading pristine Market Data...")
    df = pd.read_parquet("/Users/milocobb/tft_model/clean_aligned_features_27yr.parquet")
    df.columns = [c.replace(".", "_") for c in df.columns]
    
    if df.index.name == 'Date':
        df = df.reset_index()
    
    max_date = df['Date'].max()
    print(f"Original Max Date in Parquet: {max_date}")

    # Fetch fresh SPY extending from max_date to today
    print("Fetching SPY through today's closing price (2026-04-10)...")
    spy_fresh = yf.download("SPY", start=max_date.strftime('%Y-%m-%d'), end="2026-04-11")['Close']
    
    spy_df = pd.DataFrame({'Date': spy_fresh.index, 'SPY': spy_fresh.values.flatten()})
    
    # Filter only new dates not in df
    new_dates = spy_df[spy_df['Date'] > max_date].copy()
    print(f"Extracted {len(new_dates)} new trading days to simulate.")

    # Concat the new template
    df = pd.concat([df, new_dates], ignore_index=True)
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # Forward fill ALL missing macro variables to simulate "holding" the macro view
    df = df.ffill()
    df = df.fillna(0)
    
    # Re-calculate needed log returns!
    if 'SPY' in df.columns:
        df['SPY_Log_Return'] = np.log(df['SPY'] / df['SPY'].shift(1)).fillna(0)
        df['Daily_Return'] = df['SPY'].pct_change().fillna(0)

    # Recreate the exact FOMC smoothing state the model trained on
    if 'directional_impact' in df.columns:
        df['fomc_regime'] = df['directional_impact'].ewm(span=21, adjust=False).mean()

    # Re-index
    df["time_idx"] = np.arange(len(df))
    df["group_id"] = "portfolio"
    
    max_prediction_length = 21

    # Re-instantiate the EXACT data boundaries used in training to map scalers properly
    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="SPY_Log_Return",          
        group_ids=["group_id"],
        max_encoder_length=252,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=[c for c in df.columns if c not in ["time_idx","group_id", "day_of_year_sin", "month", "Date", "Daily_Return"] and not c.startswith("target") and c != "SPY_Log_Return"],
        time_varying_known_reals=["day_of_year_sin", "month"] if "day_of_year_sin" in df.columns else [],
        add_relative_time_idx=True,
    )

    print("Loading internal weights from Epoch 12...")
    ckpt_path = "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/lightning_logs/version_1162/checkpoints/epoch=12-step=3055.ckpt"
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)

    # Only evaluate the newly attached segment + standard trailing context
    # Min prediction index captures the exact moment the original dataset stopped
    min_idx = len(df) - len(new_dates) - 10 
    dataset = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=min_idx)
    dataloader = dataset.to_dataloader(train=False, batch_size=8, num_workers=0)

    print("Executing Neural Inference Out-Of-Sample (Through Today's Close)...")
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
            
    preds = np.array(preds_list)
    pred_idx = np.array(idx_list)

    results_df = pd.DataFrame({
        "time_idx": pred_idx,
        "Forward_21d_Prediction": preds
    })
    
    eval_df = pd.merge(new_dates[['Date']], df[['time_idx', 'Date', 'SPY', 'Daily_Return']], on='Date', how='left')
    eval_df = pd.merge(eval_df, results_df, on="time_idx", how="inner")
    
    # We execute position based on the sequential log return threshold
    eval_df['Position'] = np.where(eval_df['Forward_21d_Prediction'] >= 0.02, 1.0, 
                     np.where(eval_df['Forward_21d_Prediction'] <= -0.01, -1.0, 0.0))

    eval_df['Strategy_Return'] = eval_df['Position'].shift(1) * eval_df['Daily_Return']
    eval_df['Equity_Curve'] = (1 + eval_df['Strategy_Return'].fillna(0)).cumprod()
    eval_df['Buy_Hold'] = (1 + eval_df['Daily_Return'].fillna(0)).cumprod()

    final_eq = eval_df['Equity_Curve'].iloc[-1]
    final_bh = eval_df['Buy_Hold'].iloc[-1]
    
    print("\n" + "="*60)
    print(" 2026 CONTINUATION: TFT (v4.2) RECENT DRAWDOWN EXTENSION ")
    print("="*60)
    print(f"Total Extension Days Scored: {len(eval_df)}")
    print(f"TFT Neural Strategy Ext Equity:   {final_eq:.3f}x")
    print(f"Naive Baseline (Buy/Hold) Equity: {final_bh:.3f}x")
    
    print("\n--- Position Sequence ---")
    eval_df['Prev_Position'] = eval_df['Position'].shift(1)
    changes = eval_df[eval_df['Position'] != eval_df['Prev_Position']].dropna(subset=['Prev_Position'])
    pos_map = {1.0: "LONG SPY", -1.0: "SHORT SPY", 0.0: "CASH"}
    print(f"{eval_df['Date'].iloc[0].strftime('%Y-%m-%d')}: Initial -> {pos_map.get(eval_df['Position'].iloc[0])}")
    for _, row in changes.iterrows():
        print(f"{row['Date'].strftime('%Y-%m-%d')}: Switched to {pos_map.get(row['Position'])}")

if __name__ == '__main__':
    run_extended_backtest()

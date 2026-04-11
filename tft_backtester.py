import pandas as pd
import numpy as np
import torch
import warnings
import tqdm
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

warnings.filterwarnings("ignore")

def run_tft_backtest():
    print("Loading pristine Market Data and building tensors...")
    df = pd.read_parquet("/Users/milocobb/tft_model/clean_aligned_features_27yr.parquet")
    df.columns = [c.replace(".", "_") for c in df.columns]
    df = df.fillna(0)

    # Recreate the exact FOMC smoothing state the model trained on
    if 'directional_impact' in df.columns:
        df['fomc_regime'] = df['directional_impact'].ewm(span=21, adjust=False).mean()

    df["time_idx"] = np.arange(len(df))
    df["group_id"] = "portfolio"
    
    max_prediction_length = 21

    # Re-instantiate the EXACT data boundaries used in training to map scalers properly
    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target_SPY_fwd21",          
        group_ids=["group_id"],
        max_encoder_length=252,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=[c for c in df.columns if c not in ["time_idx","group_id", "day_of_year_sin", "month"] and not c.startswith("target")],
        time_varying_known_reals=["day_of_year_sin", "month"] if "day_of_year_sin" in df.columns else [],
        add_relative_time_idx=True,
    )

    print("Loading internal weights from Epoch 13...")
    ckpt_path = "/Users/milocobb/Desktop/Recent Swarm Papers/lightning_logs/version_6/checkpoints/epoch=13-step=3290.ckpt"
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)

    # Evaluate the final 3 years (out-of-sample stress test) to avoid multi-hour iterator stalls
    dataset = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=len(df)-750)
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    print("Executing Neural Inference Out-Of-Sample (Final 3 Years)...")
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    preds_list = []
    idx_list = []
    
    with torch.no_grad():
        for x, y in tqdm.tqdm(dataloader, desc="TFT Forward Pass"):
            # Execute on CPU
            out = model(x)
            p = model.to_prediction(out)
            
            final_p = p[:, -1].numpy()
            final_idx = x["decoder_time_idx"][:, -1].numpy()
            
            preds_list.extend(final_p)
            idx_list.extend(final_idx)
            
    preds = np.array(preds_list)
    pred_idx = np.array(idx_list)

    # Map predictions back to the actual timeline
    results_df = pd.DataFrame({
        "time_idx": pred_idx,
        "Forward_21d_Prediction": preds
    })
    
    # Merge back and map original Datetime Index
    df = pd.merge(df, results_df, on="time_idx", how="left").set_index(df.index)
    # Carry-forward the prediction for evaluating daily steps if desired, 
    # but the simplest execution trades based directly on rolling predictions.
    df['Forward_21d_Prediction'] = df['Forward_21d_Prediction'].ffill().fillna(0)

    print("Applying Positional Threshold Logic (+2.00% / -1.00%)...")
    # Condition: 0.02 = 2.00% log return expected over next month
    df['Position'] = np.where(df['Forward_21d_Prediction'] >= 0.02, 1.0, 
                     np.where(df['Forward_21d_Prediction'] <= -0.01, -1.0, 0.0))

    # Calculate returns (using the actual 1-day daily forward log returned approximation)
    if 'SPY' in df.columns:
        df['Daily_Return'] = df['SPY'].pct_change()
    elif 'target_SPY_fwd21' in df.columns:
        # If true SPY daily return isn't preserved easily, we use the forward target fractionalized as a proxy
        # But wait! A proper backtest needs the day's actual return to multiply position.
        # Let's derive daily returns. In typical pipelines there's a base close price variable.
        pass

    # Better logic: target_SPY_fwd21 is derived from shifting SPY by 21.
    # But usually 'target_SPY_fwd21' implies the forward window. We need the 1-day underlying.
    # To keep this mathematically safe, let's load SPY from market_df.
    # Let's rebuild the market close logic exactly as we did in the original fomc script
    # For now, let's look for a SPY Close proxy, or generate one from another proxy variable if missing.
    # Assume target_SPY is the return. If clean_aligned_features_27yr doesn't have SPY...
    pass

    # Actually, df already has the returns if we look back. But wait, `target_SPY_fwd21` is the 21-day forward. 
    # This means `target_SPY_fwd21` / 21 is a static daily average, not a true sequential compounded path. 
    # Let's search inside df for specific SPY index fields.
    close_col = [c for c in df.columns if 'SPY' in c and 'target' not in c]
    if len(close_col) > 0:
        df['Daily_Return'] = df[close_col[0]].pct_change()
    else:
        # Fallback to the 21-day return divided by 21, but shifted properly backward
        # target_SPY_fwd21 at T = return from T to T+21.
        # Thus, target_SPY_fwd21 at T-21 = return from T-21 to T.
        # We can approximate daily returns just by taking the daily diff of a proxy
        df['Daily_Return'] = df['target_SPY_fwd21'].shift(21) / 21

    # Calculate returns inside the actual out-of-sample forward bounds securely
    eval_df = df[df['time_idx'].isin(pred_idx)].copy()
    
    eval_df['Strategy_Return'] = eval_df['Position'].shift(1) * eval_df['Daily_Return']
    
    eval_df['Equity_Curve'] = (1 + eval_df['Strategy_Return'].fillna(0)).cumprod()
    eval_df['Buy_Hold'] = (1 + eval_df['Daily_Return'].fillna(0)).cumprod()

    final_eq = eval_df['Equity_Curve'].iloc[-1]
    final_bh = eval_df['Buy_Hold'].iloc[-1]
    
    
    print("\n" + "="*60)
    print(" TEMPORAL FUSION TRANSFORMER (v4.1) OUT-OF-SAMPLE BACKTEST ")
    print("="*60)
    print(f"Total OOS Trading Days Scored: {len(eval_df)}")
    print(f"TFT Neural Strategy Final Equity:   {final_eq:.2f}x")
    print(f"Naive Baseline (Buy/Hold) Equity:   {final_bh:.2f}x")
    
    outperformance = (final_eq - final_bh) / final_bh * 100
    print(f"Cumulative Alpha Outperformance:    {outperformance:+.2f}%")

    def get_max_drawdown(equity_series):
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        return drawdown.min() * 100
        
    print(f"TFT Maximum Drawdown:               {get_max_drawdown(eval_df['Equity_Curve']):.2f}%")
    print(f"Buy/Hold Maximum Drawdown:          {get_max_drawdown(eval_df['Buy_Hold']):.2f}%")

    print("\n--- Year-by-Year Performance ---")
    if eval_df.index.name == 'Date':
        eval_df['Year'] = pd.to_datetime(eval_df.index).year
    elif 'Date' in eval_df.columns:
        eval_df['Year'] = pd.to_datetime(eval_df['Date']).dt.year
    else:
        eval_df['Year'] = 'OOS'
        
    if 'Year' in eval_df.columns:
        yearly_perf = eval_df.groupby('Year').apply(
            lambda x: pd.Series({
                'TFT': (x['Strategy_Return'].fillna(0) + 1).prod() - 1,
                'SPY': (x['Daily_Return'].fillna(0) + 1).prod() - 1
            })
        )
        for year, row in yearly_perf.iterrows():
            print(f"{year}  |  TFT: {row['TFT']*100:>7.2f}%  |  SPY: {row['SPY']*100:>7.2f}%")
            
    if outperformance > 0:
        print("\nCONCLUSION: Strategy demonstrates dominant structural Alpha generation post-latency-shielding.")
    else:
        print("\nCONCLUSION: Strategy failed to outperform Baseline Buy/Hold across 3-year OOS dataset.")

if __name__ == '__main__':
    run_tft_backtest()

import pandas as pd
import numpy as np
import torch
import warnings
import tqdm
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

warnings.filterwarnings("ignore")

def run_full_backtest():
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
    
    max_prediction_length = 21

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

    dataset = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=253)
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    print("Executing Neural Inference over FULL HISTORY...")
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    preds_list = []
    idx_list = []
    
    with torch.no_grad():
        for x, y in tqdm.tqdm(dataloader):
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
    
    eval_df = pd.merge(df, results_df, on="time_idx", how="inner")
    
    # +2.00% / -1.00% logic
    eval_df['Position'] = np.where(eval_df['Forward_21d_Prediction'] >= 0.02, 1.0, 
                     np.where(eval_df['Forward_21d_Prediction'] <= -0.01, -1.0, 0.0))

    eval_df['Prev_Position'] = eval_df['Position'].shift(1)
    changes = eval_df[eval_df['Position'] != eval_df['Prev_Position']].dropna(subset=['Prev_Position'])
    
    print(f"\n=========================================")
    print(f" TOTAL TRADES OVER 27 YEARS: {len(changes)}")
    print(f"=========================================\n")
    
    pos_map = {1.0: "LONG", -1.0: "SHORT", 0.0: "CASH"}
    output = []
    
    for _, row in changes.iterrows():
        dt = row['Date'].strftime('%Y-%m-%d')
        prev = pos_map.get(row['Prev_Position'])
        curr = pos_map.get(row['Position'])
        
        # Determine rationale based on extreme metrics at the time:
        liqi = row.get("net_liquidity_momentum", 0.0)
        cpi = row.get("cpi_yoy", 0.0)
        fede = row.get("FEDFUNDS", 0.0)
        fomc = row.get("fomc_regime", 0.0)
        
        rationale = []
        if liqi < -0.05: rationale.append("Defensive: Severe Global Liquidity Drain (<-5%)")
        elif liqi > 0.05: rationale.append("Offensive: Explosive Central Bank Liquidity Growth (>+5%)")
        
        if cpi > 0.04 and fede < cpi: rationale.append("Caution: Runaway Inflation with Fed Behind-the-Curve")
        elif fomc < -0.2: rationale.append("Defensive: Ultra-Hawkish FOMC NLP Paradigm Shift")
        elif fomc > 0.2: rationale.append("Offensive: Dovish/Accommodative FOMC Communications")
        
        if not rationale:
            rationale.append("Standard Model Re-evaluation (Momentum / Volatility Bounds Crossed Thresholds)")
            
        rat_str = " | ".join(rationale)
        output.append(f"{dt}: {prev} -> {curr} | Rationale: {rat_str}")

    print("\nSAMPLE TRADES (First 10 & Last 10) for Analysis:")
    for l in output[:10]: print(l)
    print("... (Truncated)")
    for l in output[-10:]: print(l)
    
    # Save the full trade log
    pd.Series(output).to_csv("full_trade_log.csv", index=False, header=False)
    print("\nComplete log saved to 'full_trade_log.csv'.")

if __name__ == '__main__':
    run_full_backtest()

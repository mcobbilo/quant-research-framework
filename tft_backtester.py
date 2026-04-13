import pandas as pd
import numpy as np
import torch
import warnings
import tqdm
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

warnings.filterwarnings("ignore")


def run_tft_backtest():
    print("Loading pristine Market Data and building tensors...")
    df = pd.read_parquet(
        "/Users/milocobb/tft_model/clean_aligned_features_27yr.parquet"
    )
    df.columns = [c.replace(".", "_") for c in df.columns]
    df = df.fillna(0)

    if "SPY" in df.columns:
        df["SPY_Log_Return"] = np.log(df["SPY"] / df["SPY"].shift(1)).fillna(0)

    # Recreate the exact FOMC smoothing state the model trained on
    if "directional_impact" in df.columns:
        df["fomc_regime"] = df["directional_impact"].ewm(span=21, adjust=False).mean()

    df["time_idx"] = np.arange(len(df))
    df["group_id"] = "portfolio"

    categoricals = ["Kronos_S1", "Kronos_S2"]
    for c in categoricals:
        if c in df.columns:
            # Handle NaN to string mapping logic natively to match training state and remove float artifacts
            df[c] = df[c].apply(lambda x: np.nan if pd.isna(x) else str(int(float(x))))

    max_prediction_length = 21

    print("Loading internal weights from Epoch 10...")
    ckpt_path = "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/lightning_logs/version_1166/checkpoints/epoch=10-step=2596.ckpt"
    model = TemporalFusionTransformer.load_from_checkpoint(
        ckpt_path, map_location="cpu", weights_only=False
    )

    # Evaluate the final 3 years (out-of-sample stress test) to avoid multi-hour iterator stalls
    # Construct seamlessly from the model's saved parameters to prevent metric corruption
    dataset = TimeSeriesDataSet.from_parameters(
        model.dataset_parameters, df, min_prediction_idx=len(df) - 750
    )
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

            # The model predicts max_prediction_length consecutive log returns. Sum them for the total forward expected scalar.
            final_p = p.sum(dim=1).numpy()
            # Map index exactly flush backward to the end of the encoder step! (0th decoder step - 1)
            final_idx = x["decoder_time_idx"][:, 0].numpy() - 1

            preds_list.extend(final_p)
            idx_list.extend(final_idx)

    preds = np.array(preds_list)
    pred_idx = np.array(idx_list)

    # Map predictions back to the actual timeline
    results_df = pd.DataFrame({"time_idx": pred_idx, "Forward_21d_Prediction": preds})

    # Merge back and map original Datetime Index
    df = pd.merge(df, results_df, on="time_idx", how="left").set_index(df.index)
    # Carry-forward the prediction for evaluating daily steps if desired,
    # but the simplest execution trades based directly on rolling predictions.
    df["Forward_21d_Prediction"] = df["Forward_21d_Prediction"].ffill().fillna(0)

    print("Applying Positional Threshold Logic (+2.00% / -1.00%)...")
    # Condition: 0.02 = 2.00% log return expected over next month
    df["Position"] = np.where(
        df["Forward_21d_Prediction"] >= 0.02,
        1.0,
        np.where(df["Forward_21d_Prediction"] <= -0.01, -1.0, 0.0),
    )

    # Calculate returns structurally, completely abolishing any latency fallbacks to target variables
    if "SPY" in df.columns:
        df["Daily_Return"] = df["SPY"].pct_change()
    else:
        close_col = [c for c in df.columns if "SPY" in c and "target" not in c]
        if len(close_col) > 0:
            df["Daily_Return"] = df[close_col[0]].pct_change()
        else:
            raise ValueError("No viable proxy for sequence daily execution found. 'target_SPY_fwd21' fallback is abolished.")

    # Calculate returns inside the actual out-of-sample forward bounds securely
    eval_df = df[df["time_idx"].isin(pred_idx)].copy()

    eval_df["Strategy_Return"] = eval_df["Position"].shift(1) * eval_df["Daily_Return"]

    eval_df["Equity_Curve"] = (1 + eval_df["Strategy_Return"].fillna(0)).cumprod()
    eval_df["Buy_Hold"] = (1 + eval_df["Daily_Return"].fillna(0)).cumprod()

    final_eq = eval_df["Equity_Curve"].iloc[-1]
    final_bh = eval_df["Buy_Hold"].iloc[-1]

    print("\n" + "=" * 60)
    print(" TEMPORAL FUSION TRANSFORMER (v4.1) OUT-OF-SAMPLE BACKTEST ")
    print("=" * 60)
    print(f"Total OOS Trading Days Scored: {len(eval_df)}")
    print(f"TFT Neural Strategy Final Equity:   {final_eq:.2f}x")
    print(f"Naive Baseline (Buy/Hold) Equity:   {final_bh:.2f}x")

    outperformance = (final_eq - final_bh) / final_bh * 100
    print(f"Cumulative Alpha Outperformance:    {outperformance:+.2f}%")

    def get_max_drawdown(equity_series):
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        return drawdown.min() * 100

    print(
        f"TFT Maximum Drawdown:               {get_max_drawdown(eval_df['Equity_Curve']):.2f}%"
    )
    print(
        f"Buy/Hold Maximum Drawdown:          {get_max_drawdown(eval_df['Buy_Hold']):.2f}%"
    )

    print("\n--- Year-by-Year Performance ---")
    if eval_df.index.name == "Date":
        eval_df["Year"] = pd.to_datetime(eval_df.index).year
    elif "Date" in eval_df.columns:
        eval_df["Year"] = pd.to_datetime(eval_df["Date"]).dt.year
    else:
        eval_df["Year"] = "OOS"

    if "Year" in eval_df.columns:
        yearly_perf = eval_df.groupby("Year").apply(
            lambda x: pd.Series(
                {
                    "TFT": (x["Strategy_Return"].fillna(0) + 1).prod() - 1,
                    "SPY": (x["Daily_Return"].fillna(0) + 1).prod() - 1,
                }
            )
        )
        for year, row in yearly_perf.iterrows():
            print(
                f"{year}  |  TFT: {row['TFT'] * 100:>7.2f}%  |  SPY: {row['SPY'] * 100:>7.2f}%"
            )

    if outperformance > 0:
        print(
            "\nCONCLUSION: Strategy demonstrates dominant structural Alpha generation post-latency-shielding."
        )
    else:
        print(
            "\nCONCLUSION: Strategy failed to outperform Baseline Buy/Hold across 3-year OOS dataset."
        )

    print("\n--- Holding Sequence Transitions ---")
    eval_df["Prev_Position"] = eval_df["Position"].shift(1)
    changes = eval_df[eval_df["Position"] != eval_df["Prev_Position"]].dropna(
        subset=["Prev_Position"]
    )

    pos_map = {1.0: "LONG SPY", -1.0: "SHORT SPY", 0.0: "CASH"}

    # Print the initial position
    initial_date = (
        eval_df.index[0].strftime("%Y-%m-%d")
        if hasattr(eval_df.index[0], "strftime")
        else eval_df.index[0]
    )
    initial_pos = pos_map.get(eval_df["Position"].iloc[0], "UNKNOWN")
    print(f"{initial_date}: Initial Allocation -> {initial_pos}")

    for idx, row in changes.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else idx
        prev = pos_map.get(row["Prev_Position"], "UNKNOWN")
        curr = pos_map.get(row["Position"], "UNKNOWN")
        print(f"{date_str}: Switched from {prev} to {curr}")


if __name__ == "__main__":
    run_tft_backtest()

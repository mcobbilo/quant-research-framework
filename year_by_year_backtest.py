import pandas as pd
import numpy as np
import torch
import os
import glob
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import warnings

warnings.filterwarnings("ignore")


def run_historical_backtest():
    print("Loading Genuine Rebuilt Market Data...")
    df = pd.read_parquet(
        "/Users/milocobb/tft_model/clean_aligned_features_27yr.parquet"
    )
    df.columns = [c.replace(".", "_") for c in df.columns]

    if df.index.name == "Date":
        df = df.reset_index()

    if "directional_impact" in df.columns:
        df["directional_impact"] = df["directional_impact"].ffill().fillna(0.0)
        df["fomc_regime"] = df["directional_impact"].ewm(span=21, adjust=False).mean()
    if "surprise_factor" in df.columns:
        df["surprise_factor"] = df["surprise_factor"].ffill().fillna(0.0)

    if "SPY" in df.columns:
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
    ]

    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="SPY_Log_Return",
        group_ids=["group_id"],
        max_encoder_length=252,
        max_prediction_length=21,
        time_varying_unknown_reals=[
            c for c in df.columns if c not in exclude_f and not c.startswith("target")
        ],
        time_varying_known_reals=["day_of_year_sin", "month"]
        if "day_of_year_sin" in df.columns
        else [],
        add_relative_time_idx=True,
    )

    print("\nLocating newest PyTorch Lightning Checkpoint...")
    # Check both directories for the newest checkpoint
    ckpts = glob.glob(
        "/Users/milocobb/tft_model/lightning_logs/version_*/checkpoints/*.ckpt"
    )
    ckpts += glob.glob(
        "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/lightning_logs/version_*/checkpoints/*.ckpt"
    )

    if not ckpts:
        print("No checkpoints found. Training must finish first.")
        return

    latest_ckpt = max(ckpts, key=os.path.getctime)
    print(f"Loading checkpoint: {latest_ckpt}")

    model = TemporalFusionTransformer.load_from_checkpoint(
        latest_ckpt, map_location="cpu", weights_only=False
    )

    dataset = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=0)
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    print(
        "Executing Neural Inference Over Entire History (This computes the 27yr evaluation)..."
    )
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

    results_df = pd.DataFrame(
        {"time_idx": np.array(idx_list), "Forward_21d_Prediction": np.array(preds_list)}
    )

    eval_df = pd.merge(
        df[["time_idx", "Date", "SPY", "Daily_Return"]],
        results_df,
        on="time_idx",
        how="inner",
    )

    eval_df["Position"] = np.where(
        eval_df["Forward_21d_Prediction"] >= 0.02,
        1.0,
        np.where(eval_df["Forward_21d_Prediction"] <= -0.01, -1.0, 0.0),
    )

    eval_df["Strategy_Return"] = eval_df["Position"].shift(1) * eval_df["Daily_Return"]
    eval_df["Year"] = pd.to_datetime(eval_df["Date"]).dt.year

    print("\n" + "=" * 80)
    print(" GENUINE YEAR-BY-YEAR IN-SAMPLE & OUT-OF-SAMPLE TEST ")
    print("=" * 80)

    years = eval_df["Year"].unique()

    def max_drawdown(return_series):
        comp_ret = (1 + return_series.fillna(0)).cumprod()
        peak = comp_ret.cummax()
        drawdown = (comp_ret - peak) / peak
        return drawdown.min() * 100

    print(
        f"{'Year':<6} | {'Strategy':<10} | {'SPY':<10} | {'Strat DD':<10} | {'SPY DD':<10} | {'Status'}"
    )
    print("-" * 80)
    for y in years:
        sub = eval_df[eval_df["Year"] == y]
        strat_eq = (1 + sub["Strategy_Return"].fillna(0)).cumprod().iloc[-1] - 1
        spy_eq = (1 + sub["Daily_Return"].fillna(0)).cumprod().iloc[-1] - 1
        strat_dd = max_drawdown(sub["Strategy_Return"])
        spy_dd = max_drawdown(sub["Daily_Return"])

        status = "OUT OF SAMPLE" if y >= 2026 else "IN-SAMPLE (TRAINING)"

        print(
            f"{y:<6} | {strat_eq * 100:>8.2f}% | {spy_eq * 100:>8.2f}% | {strat_dd:>8.2f}% | {spy_dd:>8.2f}% | {status}"
        )

    print("\n" + "=" * 80)
    print(" CUMULATIVE 27-YEAR RETURNS ")
    print("=" * 80)
    total_strat = (1 + eval_df["Strategy_Return"].fillna(0)).cumprod().iloc[-1] - 1
    total_spy = (1 + eval_df["Daily_Return"].fillna(0)).cumprod().iloc[-1] - 1
    print(f"Total Cumulative Strategy Return: {total_strat * 100:.2f}%")
    print(f"Total Cumulative SPY (Benchmark): {total_spy * 100:.2f}%")


if __name__ == "__main__":
    run_historical_backtest()

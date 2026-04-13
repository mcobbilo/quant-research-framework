import pandas as pd
import numpy as np
import warnings
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src", "models"))
from timesfm_wrapper import TimesFMWrapper

warnings.filterwarnings("ignore")

def run_timesfm_backtest():
    print("Loading pristine Market Data...")
    df = pd.read_parquet(
        "/Users/milocobb/tft_model/clean_aligned_features_27yr.parquet"
    )
    df.columns = [c.replace(".", "_") for c in df.columns]
    df = df.fillna(0)

    if "SPY" in df.columns:
        df["SPY_Log_Return"] = np.log(df["SPY"] / df["SPY"].shift(1)).fillna(0)
    else:
        raise ValueError("Missing 'SPY' column for performance baseline.")

    horizon = 21
    context_size = 512
    n_windows = 35 # 21 * 35 = 735 trading days (approx 3 years OOS)
    
    print("Formatting DataFrame to Nixtla Spec (ds, unique_id, y)...")
    if "Date" not in df.columns and hasattr(df.index, 'name') and df.index.name != "Date":
        # Maybe it's a generic index?
        pass
    
    # Safely extract Date from index if it's there
    date_col = df.index if "Date" not in df.columns else df["Date"]
        
    nf_df = pd.DataFrame({
        "unique_id": "SPY",
        "ds": pd.to_datetime(date_col),
        "y": df["SPY_Log_Return"]
    }).reset_index(drop=True)
    
    # In order to align our predictions easily, retain the original index/date mappings
    nf_df["original_index"] = df.index
    
    # Initialize the Foundation Model
    print(f"\nInitializing TimesFM Wrapper (horizon={horizon}, context={context_size})...")
    tfm = TimesFMWrapper(h=horizon, input_size=context_size)

    # Perform the cross validation loop which internally strides backwards
    print("\nExecuting Out-Of-Sample Cross Validation (Final 3 Years)...")
    cv_results = tfm.cross_validation(nf_df, n_windows=n_windows, step_size=horizon)

    if cv_results.empty:
        raise ValueError("Cross validation loop yielded no results. Please check window sizings.")

    # Sort evaluation logic chronologically
    cv_results = cv_results.sort_values(by="ds").reset_index(drop=True)
    
    # Extract prediction 
    cv_results["Forward_21d_Prediction"] = cv_results["xLSTM-median"] # output target format

    # Since it predicts a sequence of 21 days going forward, in log returns, the cumulative
    # expected return over that sequence is the sum of the log return predictions.
    # However, each row in cv_results already represents a single step mapped within the horizon.
    # We should calculate the *total sum of the horizon* mapped to the beginning of the evaluation period
    # OR trade dynamically daily based on the step-wise forward projection. 
    # Since TimesFM outputs direct values per slice, we'll treat them directly.

    print("Applying Positional Threshold Logic (LONG if forecast > 0.0, else CASH)...")
    # Using symmetrical zero-threshold for long/cash regime against SPY 
    cv_results["Position"] = np.where(cv_results["Forward_21d_Prediction"] > 0.0, 1.0, 0.0)

    # Calculate Daily_Return before merge
    if "SPY" in df.columns:
        df["Daily_Return"] = df["SPY"].pct_change()
        
    # Merge results back into full dataframe
    # We kept the original_index so we can align it
    eval_df = pd.merge(
        df[["SPY", "Daily_Return"]].reset_index(names=["Date"] if "Date" not in df.columns else None), 
        cv_results[["ds", "Position", "Forward_21d_Prediction"]], 
        left_on="Date" if "Date" in df.columns or df.index.name == "Date" else "index", 
        right_on="ds", 
        how="inner"
    )
    
    eval_df["Daily_Return"] = eval_df["SPY"].pct_change()
    
    # Strategy captures the *next* day's return based on current position shift (Shift 1)
    eval_df["Strategy_Return"] = eval_df["Position"].shift(1) * eval_df["Daily_Return"]
    
    eval_df["Equity_Curve"] = (1 + eval_df["Strategy_Return"].fillna(0)).cumprod()
    eval_df["Buy_Hold"] = (1 + eval_df["Daily_Return"].fillna(0)).cumprod()
    
    final_eq = eval_df["Equity_Curve"].iloc[-1]
    final_bh = eval_df["Buy_Hold"].iloc[-1]

    print("\n" + "=" * 60)
    print(" TIMESFM 2.5 (200M) OUT-OF-SAMPLE BACKTEST ")
    print("=" * 60)
    print(f"Total OOS Trading Days Scored: {len(eval_df)}")
    print(f"TimesFM Zero-Shot Final Equity:     {final_eq:.2f}x")
    print(f"Naive Baseline (Buy/Hold) Equity:   {final_bh:.2f}x")

    outperformance = (final_eq - final_bh) / final_bh * 100
    print(f"Cumulative Alpha Outperformance:    {outperformance:+.2f}%")

    def get_max_drawdown(equity_series):
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        return drawdown.min() * 100

    print(
        f"TimesFM Maximum Drawdown:           {get_max_drawdown(eval_df['Equity_Curve']):.2f}%"
    )
    print(
        f"Buy/Hold Maximum Drawdown:          {get_max_drawdown(eval_df['Buy_Hold']):.2f}%"
    )

    print("\n--- Year-by-Year Performance ---")
    try:
        date_col_eval = eval_df.index if "Date" not in eval_df.columns else eval_df["Date"]
        eval_df["Year"] = pd.to_datetime(date_col_eval).dt.year
    except:
        eval_df["Year"] = "OOS"
    
    yearly_perf = eval_df.groupby("Year").apply(
        lambda x: pd.Series(
            {
                "TimesFM": (x["Strategy_Return"].fillna(0) + 1).prod() - 1,
                "SPY": (x["Daily_Return"].fillna(0) + 1).prod() - 1,
            }
        )
    )

    for year, row in yearly_perf.iterrows():
        print(f"{year}  |  TimesFM: {row['TimesFM']*100:>7.2f}%  |  SPY: {row['SPY']*100:>7.2f}%")

if __name__ == "__main__":
    run_timesfm_backtest()

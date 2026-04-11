import yfinance as yf
import pandas as pd
import numpy as np

# 1. Data Acquisition
ticker = "SPY"
try:
    # Download all available history for SPY
    df = yf.download(ticker, start="1990-01-01", progress=False)

    # Ensure to extract the 'Close' price column correctly.
    # For a single ticker download, yfinance usually returns a flat DataFrame.
    if "Close" in df.columns:
        df = df[["Close"]]
    else:
        # This block handles potential MultiIndex structure if it were present,
        # but for a single ticker download, 'Close' is typically a top-level column.
        # If the columns were a MultiIndex (e.g., ('Close', 'SPY')), this would extract it.
        # For a single ticker, `df['Close']` is usually sufficient.
        if isinstance(
            df.columns, pd.MultiIndex
        ) and "Close" in df.columns.get_level_values("Price"):
            df = df.xs("Close", level="Price", axis=1)
        else:
            raise ValueError(
                f"Could not find 'Close' column in downloaded data for {ticker}"
            )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

except Exception as e:
    print(f"Error downloading data for {ticker}: {e}")
    # Exit or handle the error appropriately if data cannot be obtained
    exit()

# Ensure 'Close' is numeric and handle potential NaNs from data download
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df.dropna(subset=["Close"], inplace=True)

# 2. Calculate SMAs
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_200"] = df["Close"].rolling(window=200).mean()

# 3. Define "Crashes" and "Deepest Crashes"
# Calculate the deviation of Close from SMA as a percentage
df["SMA_20_deviation"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
df["SMA_200_deviation"] = (df["Close"] - df["SMA_200"]) / df["SMA_200"]

# Define "deepest crashes" as being in the bottom Xth percentile of historical deviations.
# Using the 5th percentile as a measure of "deepest" or "extreme".
percentile_threshold = 0.05  # Bottom 5%

# Calculate the deviation thresholds for both SMAs
threshold_20 = df["SMA_20_deviation"].quantile(percentile_threshold)
threshold_200 = df["SMA_200_deviation"].quantile(percentile_threshold)

# Identify signal days where the price is significantly below the respective SMA
signal_20_crash_days = df[df["SMA_20_deviation"] < threshold_20].index
signal_200_crash_days = df[df["SMA_200_deviation"] < threshold_200].index

# 4. Calculate 60-Day Forward Returns
# Calculate the percentage change from current close price to close price 60 days later.
# .shift(-60) ensures there is no lookahead bias.
df["Fwd_60D_Return"] = df["Close"].pct_change(60).shift(-60)

# 5. Calculate Average Returns for Signals
# For SMA_20 crashes
# Select forward returns only on the signal days and drop any NaNs (e.g., at end of data)
returns_20_crash = df.loc[signal_20_crash_days, "Fwd_60D_Return"].dropna()
avg_return_20_crash = returns_20_crash.mean() if not returns_20_crash.empty else np.nan

# For SMA_200 crashes
returns_200_crash = df.loc[signal_200_crash_days, "Fwd_60D_Return"].dropna()
avg_return_200_crash = (
    returns_200_crash.mean() if not returns_200_crash.empty else np.nan
)

# Convert average returns to percentage for clear comparison
avg_return_20_crash_pct = (
    avg_return_20_crash * 100 if not np.isnan(avg_return_20_crash) else np.nan
)
avg_return_200_crash_pct = (
    avg_return_200_crash * 100 if not np.isnan(avg_return_200_crash) else np.nan
)

# 6. Validate Hypothesis and print result
fact_string = f"Fact: For SPY data (1990-present), buying the deepest 20-Day short-term SMA crashes (defined as Close being in the bottom {percentile_threshold * 100:.0f}th percentile relative to SMA_20) yielded an average 60-day forward return of {avg_return_20_crash_pct:.2f}%. "
fact_string += f"Buying extreme 200-Day macro SMA crashes (defined as Close being in the bottom {percentile_threshold * 100:.0f}th percentile relative to SMA_200) yielded an average 60-day forward return of {avg_return_200_crash_pct:.2f}%. "

if not np.isnan(avg_return_20_crash_pct) and not np.isnan(avg_return_200_crash_pct):
    if avg_return_20_crash_pct > avg_return_200_crash_pct:
        fact_string += "Mathematically, the average 60-day return for short-term 20-Day SMA crashes ({avg_return_20_crash_pct:.2f}%) was indeed higher than for long-term 200-Day SMA crashes ({avg_return_200_crash_pct:.2f}%), supporting the hypothesis that buying short-term cyclical drawdowns empirically outperforms attempting to buy extreme long-term structural momentum loss."
    else:
        fact_string += "Mathematically, the average 60-day return for short-term 20-Day SMA crashes ({avg_return_20_crash_pct:.2f}%) was not higher than for long-term 200-Day SMA crashes ({avg_return_200_crash_pct:.2f}%), which contradicts the hypothesis that buying short-term cyclical drawdowns empirically outperforms attempting to buy extreme long-term structural momentum loss."
else:
    fact_string += "Insufficient data points or calculations resulted in NaN values, preventing a conclusive comparison."

print(fact_string)

import os
import pandas as pd
import yfinance as yf
import numpy as np

def run_anomaly_test():
    """
    Tests the 5-day CPCE Moving Average > 1.2 anomaly.
    Prioritizes expected value and asymmetric tails over win rate.
    """
    # Construct path robustly to quant_framework root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.normpath(os.path.join(base_dir, '..', '..'))
    cpce_path = os.path.join(ROOT_DIR, '_cpce.csv')
    
    if not os.path.exists(cpce_path):
        return f"Error: _cpce.csv not found precisely at: {cpce_path}"
        
    # 1. Load the CPCE data
    cpce_df = pd.read_csv(cpce_path, skiprows=1)
    cpce_df.columns = [c.strip() for c in cpce_df.columns]
    
    if 'Date' not in cpce_df.columns or 'Close' not in cpce_df.columns:
        return "Error: Unexpected columns in _cpce.csv"
        
    cpce_df['Date'] = pd.to_datetime(cpce_df['Date'])
    cpce_df = cpce_df.sort_values('Date').set_index('Date')
    
    # Calculate 5-day MA of CPCE Close
    cpce_df['CPCE_5MA'] = cpce_df['Close'].rolling(window=5).mean()
    
    # 2. Fetch SPY Market returns from yfinance
    start_date = cpce_df.index.min().strftime('%Y-%m-%d')
    end_date = cpce_df.index.max().strftime('%Y-%m-%d')
    
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    # yfinance multi-index columns adjustment in newer versions
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
        
    spy = spy[['Close']].rename(columns={'Close': 'SPY_Close'})
    
    # Calculate 48-hour (2 trading days) forward returns
    spy['Fwd_2D_Return'] = spy['SPY_Close'].pct_change(2).shift(-2)
    
    # 3. Align data
    df = cpce_df.join(spy, how='inner')
    df = df.dropna(subset=['CPCE_5MA', 'Fwd_2D_Return'])
    
    # 4. Filter logic based on SOUL.md directives (Expected Value analysis)
    normal_conditions = df[df['CPCE_5MA'] <= 1.2]
    fat_tail_conditions = df[df['CPCE_5MA'] > 1.2]
    
    if len(normal_conditions) == 0 or len(fat_tail_conditions) == 0:
        return "Insufficient events to calculate asymmetry."

    # Normal Metrics
    norm_fwd = normal_conditions['Fwd_2D_Return']
    norm_win_rate = (norm_fwd > 0).mean() * 100
    norm_avg_return = norm_fwd.mean() * 100
    
    # Fat-Tail Anomaly Metrics
    anom_fwd = fat_tail_conditions['Fwd_2D_Return']
    anom_win_rate = (anom_fwd > 0).mean() * 100
    anom_avg_return = anom_fwd.mean() * 100
    
    win_rate_diff = norm_win_rate - anom_win_rate
    
    # Format the explicit output requested
    return (
        f"Fact: 5-Day PUT/CALL > 1.2 reduces win rate by {win_rate_diff:.1f}% "
        f"(from {norm_win_rate:.1f}% to {anom_win_rate:.1f}%) but pushes expected "
        f"48-hour upside return to {anom_avg_return:.2f}% (vs {norm_avg_return:.2f}% "
        f"normally). Action: Maintain position sizing, but widen take-profit targets "
        f"to capture standard deviation expansion."
    )

if __name__ == "__main__":
    print(run_anomaly_test())

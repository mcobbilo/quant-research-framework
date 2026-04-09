import pandas as pd
import yfinance as yf
import os

def analyze_nya200r():
    # Load NYA200R
    filepath = "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/_NYA200R.csv"
    print(f"Loading {filepath}...")
    df_nya = pd.read_csv(filepath, header=1, skipinitialspace=True)
    df_nya['Date'] = pd.to_datetime(df_nya['Date'])
    df_nya = df_nya[['Date', 'Close']].rename(columns={'Close': 'NYA200R'})
    df_nya.set_index('Date', inplace=True)
    
    # Load SPY
    print("Downloading historical SPY data for forward return calculations...")
    spy = yf.download('SPY', start="1990-01-01", end=None, progress=False)
    
    # Handle yfinance multi-index if present
    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy['Close']['SPY']
    else:
        spy_close = spy['Close']
        
    df_spy = pd.DataFrame({'SPY_Close': spy_close})
    
    # Merge datasets
    df = df_spy.join(df_nya, how='inner')
    
    # Calculate Forward Returns for 5, 10, 20, 60 days
    df['Fwd_5D'] = (df['SPY_Close'].shift(-5) / df['SPY_Close'] - 1) * 100
    df['Fwd_10D'] = (df['SPY_Close'].shift(-10) / df['SPY_Close'] - 1) * 100
    df['Fwd_20D'] = (df['SPY_Close'].shift(-20) / df['SPY_Close'] - 1) * 100
    df['Fwd_60D'] = (df['SPY_Close'].shift(-60) / df['SPY_Close'] - 1) * 100
    
    # Clean up NaNs from the end of the dataset
    df_clean = df.dropna(subset=['NYA200R'])
    
    # Get 50 Highest and 50 Lowest Readings
    highest_50 = df_clean.nlargest(50, 'NYA200R')
    lowest_50 = df_clean.nsmallest(50, 'NYA200R')
    
    # Averages
    def summarize(subset, name):
        print(f"\n--- {name} (N={len(subset)}) ---")
        print(f"Average NYA200R Value: {subset['NYA200R'].mean():.2f}%")
        print(f"Average 5-Day Forward Return:  {subset['Fwd_5D'].mean():.2f}%")
        print(f"Average 10-Day Forward Return: {subset['Fwd_10D'].mean():.2f}%")
        print(f"Average 20-Day Forward Return: {subset['Fwd_20D'].mean():.2f}%")
        print(f"Average 60-Day Forward Return: {subset['Fwd_60D'].mean():.2f}%")
        
        # Win Rates
        win_5d = (subset['Fwd_5D'] > 0).mean() * 100
        win_10d = (subset['Fwd_10D'] > 0).mean() * 100
        win_20d = (subset['Fwd_20D'] > 0).mean() * 100
        win_60d = (subset['Fwd_60D'] > 0).mean() * 100
        print(f"\nWin Rates (>0% return):")
        print(f"5-Day:  {win_5d:.1f}%")
        print(f"10-Day: {win_10d:.1f}%")
        print(f"20-Day: {win_20d:.1f}%")
        print(f"60-Day: {win_60d:.1f}%")

    summarize(highest_50, "50 HIGHEST NYA200R READINGS")
    summarize(lowest_50, "50 LOWEST NYA200R READINGS")
    
if __name__ == "__main__":
    analyze_nya200r()

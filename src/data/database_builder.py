import pandas as pd
import yfinance as yf
import sqlite3
import os
import warnings

warnings.filterwarnings('ignore')

def calc_tsi(series, r=25, s=13):
    """
    True Strength Indicator (TSI)
    TSI = (EMA(EMA(Momentum, r), s) / EMA(EMA(Absolute Momentum, r), s)) * 100
    """
    m = series.diff()
    am = abs(m)
    
    ema_m = m.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
    ema_am = am.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
    
    return (ema_m / ema_am) * 100

def build_database():
    print("[DB Architect] Establishing local SQLite connection...")
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'market_data.db')
    conn = sqlite3.connect(db_path)
    
    print("[DB Architect] Fetching 25-Year Core Market Data (+TLT, +TNX)...")
    tickers = ["SPY", "^VIX", "GC=F", "HG=F", "TLT", "^TNX"]
    
    # Download the core data array simultaneously
    data = yf.download(tickers, start="2000-01-01", end="2025-01-01", group_by='ticker', auto_adjust=True, progress=False)
    
    df = pd.DataFrame(index=data.index)
    
    # Map the OHLC arrays into a flattened pandas schema
    print("[DB Architect] Flattening OHLC structures...")
    for ticker in tickers:
        prefix = ticker.replace('^', '').replace('=F', '')
        
        try:
            if isinstance(data.columns, pd.MultiIndex):
                tk_data = data[ticker]
            else:
                tk_data = data
                
            if 'Close' in tk_data.columns:
                df[f'{prefix}_CLOSE'] = tk_data['Close']
            if 'Open' in tk_data.columns:
                df[f'{prefix}_OPEN'] = tk_data['Open']
            if 'High' in tk_data.columns:
                df[f'{prefix}_HIGH'] = tk_data['High']
            if 'Low' in tk_data.columns:
                df[f'{prefix}_LOW'] = tk_data['Low']
        except Exception as e:
            print(f"Error parsing {ticker}: {e}")
            
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    print("[DB Architect] Calculating True Strength Indicators (TSI)...")
    df['SPY_TSI'] = calc_tsi(df['SPY_CLOSE'])
    df['VIX_TSI'] = calc_tsi(df['VIX_CLOSE'])
    df['TLT_TSI'] = calc_tsi(df['TLT_CLOSE'])
    
    print("[DB Architect] Synthesizing VIX/TNX Cross-Asset Derivatives...")
    df['VIX_TNX_RATIO'] = df['VIX_CLOSE'] / df['TNX_CLOSE']
    
    # 1. 7-Day Fast PPO (Fast=1, Slow=7)
    ema_fast = df['VIX_TNX_RATIO'].ewm(span=1, adjust=False).mean()
    ema_slow = df['VIX_TNX_RATIO'].ewm(span=7, adjust=False).mean()
    df['VIX_TNX_PPO_7'] = ((ema_fast - ema_slow) / ema_slow) * 100
    
    # 2. 20-Day Bollinger Bands
    df['VIX_TNX_SMA_20'] = df['VIX_TNX_RATIO'].rolling(20).mean()
    df['VIX_TNX_STD_20'] = df['VIX_TNX_RATIO'].rolling(20).std()
    df['VIX_TNX_BB_UPPER'] = df['VIX_TNX_SMA_20'] + (df['VIX_TNX_STD_20'] * 2)
    df['VIX_TNX_BB_LOWER'] = df['VIX_TNX_SMA_20'] - (df['VIX_TNX_STD_20'] * 2)
    
    # 3. TSI Integration
    df['VIX_TNX_TSI'] = calc_tsi(df['VIX_TNX_RATIO'])
    
    print("[DB Architect] Mapping 200-Day Geometries and Percentage Displacements...")
    # SPY 200-Day
    df['SPY_SMA_200'] = df['SPY_CLOSE'].rolling(200).mean()
    df['SPY_PCT_FROM_200'] = ((df['SPY_CLOSE'] - df['SPY_SMA_200']) / df['SPY_SMA_200']) * 100
    
    # TLT 200-Day
    df['TLT_SMA_200'] = df['TLT_CLOSE'].rolling(200).mean()
    df['TLT_PCT_FROM_200'] = ((df['TLT_CLOSE'] - df['TLT_SMA_200']) / df['TLT_SMA_200']) * 100
    
    # VIX/TNX 200-Day
    df['VIX_TNX_SMA_200'] = df['VIX_TNX_RATIO'].rolling(200).mean()
    df['VIX_TNX_PCT_FROM_200'] = ((df['VIX_TNX_RATIO'] - df['VIX_TNX_SMA_200']) / df['VIX_TNX_SMA_200']) * 100
    
    print("[DB Architect] Compiling SPY vs TLT Relative Performance \u0026 Z-Scores...")
    for period in [3, 5, 7]:
        spy_ret = df['SPY_CLOSE'].pct_change(periods=period)
        tlt_ret = df['TLT_CLOSE'].pct_change(periods=period)
        diff = spy_ret - tlt_ret
        df[f'SPY_TLT_DIFF_{period}D'] = diff
        
        # Calculate 252-day (1-Year) rolling Z-Score for the performance offset
        r_mean = diff.rolling(window=252).mean()
        r_std = diff.rolling(window=252).std()
        df[f'SPY_TLT_DIFF_{period}D_ZSCORE'] = (diff - r_mean) / r_std
    
    # Secure the Synthetic McClellan matrix into the DB architecture
    mcc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mcclellan_sp500.csv')
    try:
        mcc = pd.read_csv(mcc_path, index_col=0, parse_dates=True)
        if mcc.index.tz is not None:
             mcc.index = mcc.index.tz_localize(None)
        mcc.to_sql('breadth_table', conn, if_exists='replace')
        print("[DB Architect] Migrated breadth_table natively into SQLite.")
    except Exception as e:
        print(f"[DB Architect] Warning: mcclellan_sp500.csv could not be merged. {e}")
        
    # Drop rows that contain NaNs due to TSI EMA spin-ups
    df.dropna(inplace=True)
    
    # Strip internal timezones for SQL compliance
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    print("[DB Architect] Committing core_market_table ledger...")
    df.to_sql('core_market_table', conn, if_exists='replace')
    
    print(f"[DB Architect] Successfully serialized {len(df)} core market instances.")
    print(f"[DB Architect] Database firmly locked at: {db_path}")
    conn.close()

if __name__ == "__main__":
    build_database()

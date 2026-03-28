import pandas as pd
import yfinance as yf
import sqlite3
import os
import requests
import warnings

warnings.filterwarnings('ignore')

FRED_API_KEY = 'ecef78f092efcef79959d0d700c594f1'

def fetch_alfred_vintage(ticker):
    import time
    print(f"[DB Architect] Connecting ALFRED Secure API -> {ticker}")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={ticker}&api_key={FRED_API_KEY}&file_type=json&realtime_start=1999-01-01"
    
    for attempt in range(3):
        r = requests.get(url)
        if r.status_code == 200:
            break
        print(f"[DB Architect] WARNING: ALFRED Connection Failed for {ticker} (Status: {r.status_code}). Retrying {attempt+1}/3...")
        time.sleep(2)
        
    if r.status_code != 200:
        return pd.Series(dtype=float, name=ticker)
        
    data = r.json()
    observations = data.get('observations', [])
    df = pd.DataFrame(observations)
    if df.empty: return pd.Series(dtype=float, name=ticker)
    df = df[df['value'] != '.']
    df['value'] = df['value'].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    df['realtime_start'] = pd.to_datetime(df['realtime_start'])
    df = df.sort_values(by=['date', 'realtime_start'])
    first_vintages = df.drop_duplicates(subset=['date'], keep='first').copy()
    first_vintages = first_vintages.set_index('date')['value']
    first_vintages.name = ticker
    return first_vintages


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

def calc_ppo(series, fast=12, slow=26, signal=9):
    """
    Percentage Price Oscillator (PPO) 12-26-9
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    ppo_line = ((ema_fast - ema_slow) / ema_slow) * 100
    ppo_signal = ppo_line.ewm(span=signal, adjust=False).mean()
    ppo_hist = ppo_line - ppo_signal
    return ppo_line, ppo_signal, ppo_hist

def calc_cmf(df, prefix, period=20):
    """
    Chaikin Money Flow (CMF)
    """
    high = df[f'{prefix}_HIGH']
    low = df[f'{prefix}_LOW']
    close = df[f'{prefix}_CLOSE']
    vol = df[f'{prefix}_VOLUME']
    
    # Avoid division by zero
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, 0.00001)
    mfv = mfm * vol
    
    cmf = mfv.rolling(window=period).sum() / vol.rolling(window=period).sum()
    return cmf
def calc_trix(series, period=15):
    """
    Triple Exponential Average (TRIX) Rate of Change.
    """
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    return ema3.pct_change() * 100

def calc_donchian(high_series, low_series, period=20):
    """
    Donchian Channel Upper and Lower Bounds
    """
    upper = high_series.rolling(window=period).max()
    lower = low_series.rolling(window=period).min()
    return upper, lower

def calc_vwap_rolling(df, prefix, period=20):
    """
    Rolling Volume Weighted Average Price (VWAP)
    """
    typical_price = (df[f'{prefix}_HIGH'] + df[f'{prefix}_LOW'] + df[f'{prefix}_CLOSE']) / 3
    cum_vol_price = (typical_price * df[f'{prefix}_VOLUME']).rolling(window=period).sum()
    cum_vol = df[f'{prefix}_VOLUME'].rolling(window=period).sum()
    return cum_vol_price / cum_vol

def build_database():
    print("[DB Architect] Establishing local SQLite connection...")
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'market_data.db')
    conn = sqlite3.connect(db_path)
    
    print("[DB Architect] Fetching 25-Year Core Market Data (+TLT, +TNX, +CL=F, +MOVE, Volatility Term Structure)...")
    # Phase 134: Global FX Integration (DXY & USD/JPY)
    tickers = ["SPY", "^VIX", "^VIX3M", "^VIX6M", "GC=F", "HG=F", "CL=F", "TLT", "^TNX", "^MOVE", "IWM", "^VVIX", "^SKEW", "RSP", "DX-Y.NYB", "JPY=X"]
    
    # Download the core data array simultaneously
    data = yf.download(tickers, start="2000-01-01", end=None, group_by='ticker', auto_adjust=True, progress=False)
    
    df = pd.DataFrame(index=data.index)
    
    # Map the OHLC arrays into a flattened pandas schema
    print("[DB Architect] Flattening OHLC structures...")
    for ticker in tickers:
        prefix = ticker.replace('^', '').replace('=F', '').replace('.NYB', '').replace('-', '').replace('=X', '')
        
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
            if 'Volume' in tk_data.columns:
                df[f'{prefix}_VOLUME'] = tk_data['Volume']
        except Exception as e:
            print(f"Error parsing {ticker}: {e}")
            
    df.ffill(inplace=True)
    
    import numpy as np
    print("[DB Architect] Synthesizing Proprietary SPY Candlestick Rejection Vectors...")
    open_close_max = df[['SPY_OPEN', 'SPY_CLOSE']].max(axis=1)
    open_close_min = df[['SPY_OPEN', 'SPY_CLOSE']].min(axis=1)
    
    df['SPY_toptail'] = (df['SPY_HIGH'] - open_close_max) / df['SPY_HIGH']
    df['SPY_revtail'] = (open_close_min - df['SPY_LOW']) / open_close_min
    
    df['SPY_Hollow_Red'] = np.where(
        (df['SPY_CLOSE'] < df['SPY_CLOSE'].shift(6)) & 
        (df['SPY_CLOSE'] < df['SPY_CLOSE'].shift(1)) &
        (df['SPY_OPEN'] < df['SPY_CLOSE'].shift(1)) &
        (df['SPY_CLOSE'] > df['SPY_OPEN']) & 
        (df['SPY_toptail'] < df['SPY_revtail']) &
        (df['SPY_CLOSE'] > df['SPY_CLOSE'].shift(20)), 
        1, 0
    )
    
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
    
    print("[DB Architect] Synthesizing Phase 94 Volatility Term-Structure Physics (Contango vs Backwardation)...")
    # Spot VIX / 3-Month VIX Ratio (Values > 1.0 indicate strict Backwardation panic)
    df['VIX_TERM_STRUCTURE_3M'] = df['VIX_CLOSE'] / df['VIX3M_CLOSE']
    df['VIX_TERM_STRUCTURE_6M'] = df['VIX_CLOSE'] / df['VIX6M_CLOSE']
    # Contango Roll Yield Premium: The physical steepness percentage
    df['VIX_CONTANGO_ROLL_YIELD_3M'] = ((df['VIX3M_CLOSE'] - df['VIX_CLOSE']) / df['VIX_CLOSE']) * 100
    df['VIX_CONTANGO_ROLL_YIELD_6M'] = ((df['VIX6M_CLOSE'] - df['VIX_CLOSE']) / df['VIX_CLOSE']) * 100
    
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
    
    print("[DB Architect] Computing 12-26-9 PPO Arrays for SPY, TLT, and Gold (GC)...")
    df['SPY_PPO_LINE'], df['SPY_PPO_SIGNAL'], df['SPY_PPO_HIST'] = calc_ppo(df['SPY_CLOSE'])
    df['TLT_PPO_LINE'], df['TLT_PPO_SIGNAL'], df['TLT_PPO_HIST'] = calc_ppo(df['TLT_CLOSE'])
    if 'GC_CLOSE' in df.columns:
        df['GC_PPO_LINE'], df['GC_PPO_SIGNAL'], df['GC_PPO_HIST'] = calc_ppo(df['GC_CLOSE'])
        
    print("[DB Architect] Computing 20-Period Chaikin Money Flow (CMF) Arrays...")
    if 'SPY_VOLUME' in df.columns:
        df['SPY_CMF'] = calc_cmf(df, 'SPY')
    if 'TLT_VOLUME' in df.columns:
        df['TLT_CMF'] = calc_cmf(df, 'TLT')
    if 'GC_VOLUME' in df.columns:
        df['GC_CMF'] = calc_cmf(df, 'GC')
    
    print("[DB Architect] Mapping Macro Credit Spreads and Yield Curves from FRED...")
    try:
        def fetch_fred(series_id):
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            df_fred = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.')
            df_fred.columns = [series_id]
            return df_fred

        fred_data = pd.concat([fetch_fred('T10Y2Y'), fetch_fred('T10YFF'), fetch_fred('BAMLC0A0CM'), fetch_fred('NFCI'), fetch_fred('TEDRATE'), fetch_fred('T10YIE')], axis=1)
        fred_data.index.name = 'Date'
        
        # Merge FRED onto main df
        df = df.join(fred_data, how='left')
    except Exception as e:
        print(f"[DB Architect] WARNING! Failed to hit FRED CSV endpoints. {e}")

    print("[DB Architect] Creating MOVE vs VIX Momentum Velocity Matrix...")
    for period in [5, 10]:
        diff_move = df['MOVE_CLOSE'].pct_change(periods=period) - df['VIX_CLOSE'].pct_change(periods=period)
        df[f'VIX_MOVE_SPREAD_{period}D'] = diff_move
        
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
        
    print("[DB Architect] Computing Phase 128 Global Macro Accelerations & Jerk...")
    if 'BAMLC0A0CM' in df.columns:
        df['Credit_Velocity_30D'] = df['BAMLC0A0CM'].diff(30)
        df['Credit_Acceleration_30D'] = df['Credit_Velocity_30D'].diff(30)
    
    if 'TEDRATE' in df.columns:
        df['TED_Velocity_30D'] = df['TEDRATE'].diff(30)
        df['TED_Acceleration_30D'] = df['TED_Velocity_30D'].diff(30)
        
    print("[DB Architect] Computing Phase 129 SPY / RSP Breadth Momentum...")
    if 'RSP_CLOSE' in df.columns:
        df['SPY_RSP_RATIO'] = df['SPY_CLOSE'] / df['RSP_CLOSE']
        df['SPY_RSP_MOMENTUM_60D'] = df['SPY_RSP_RATIO'].pct_change(periods=60)
        
    print("[DB Architect] Synthesizing Phase 96: Dr. Copper vs Gold Cointegration Arbitrage...")
    if 'GC_CLOSE' in df.columns and 'HG_CLOSE' in df.columns:
        df['GC_HG_RATIO'] = df['GC_CLOSE'] / df['HG_CLOSE']
        df['GC_HG_ZSCORE_60'] = (df['GC_HG_RATIO'] - df['GC_HG_RATIO'].rolling(60).mean()) / df['GC_HG_RATIO'].rolling(60).std()
        df['GC_HG_ZSCORE_252'] = (df['GC_HG_RATIO'] - df['GC_HG_RATIO'].rolling(252).mean()) / df['GC_HG_RATIO'].rolling(252).std()
        
    print("[DB Architect] Synthesizing Phase 121 Absolute Black-Swan Forensics...")
    if 'IWM_CLOSE' in df.columns and 'SPY_CLOSE' in df.columns:
        df['IWM_SPY_RATIO'] = df['IWM_CLOSE'] / df['SPY_CLOSE']
    if 'VIX_CLOSE' in df.columns and 'VVIX_CLOSE' in df.columns:
        df['VIX_VVIX_RATIO'] = df['VIX_CLOSE'] / df['VVIX_CLOSE']
    if 'SKEW_CLOSE' in df.columns:
        df['SKEW_ZSCORE_252'] = (df['SKEW_CLOSE'] - df['SKEW_CLOSE'].rolling(252).mean()) / df['SKEW_CLOSE'].rolling(252).std()
    
    print("[DB Architect] Integrating Pure StockCharts Market Breadth & Sentiment Arrays...")
    breadth_files = {
        'NYADV': '_NYADV.csv',
        'NYDEC': '_NYdec.csv',
        'NYUPV': '_NYupv.csv',
        'NYDNV': '_NYdnv.csv',
        'NYA200R': '_NYA200R.csv',
        'CPC': '_cpc.csv',
        'CPCE': '_cpce.csv',
        'NYADU': '_NYADu.csv'
    }
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    breadth_df = pd.DataFrame()
    for name, filename in breadth_files.items():
        filepath = os.path.join(base_dir, filename)
        try:
            temp_df = pd.read_csv(filepath, header=1, skipinitialspace=True)
            temp_df['Date'] = pd.to_datetime(temp_df['Date'])
            temp_df.set_index('Date', inplace=True)
            breadth_df[name] = temp_df['Close']
            print(f" -> Mapping {name} vector...")
        except Exception as e:
            print(f"[DB Warning] Could not load {filename}: {e}")

    # Synthesize Custom McClellan Oscillators natively
    if 'NYADV' in breadth_df.columns and 'NYDEC' in breadth_df.columns:
        print(" -> Synthesizing McClellan Price Oscillator...")
        net_adv = breadth_df['NYADV'] - breadth_df['NYDEC']
        ema19 = net_adv.ewm(span=19, adjust=False).mean()
        ema39 = net_adv.ewm(span=39, adjust=False).mean()
        breadth_df['MCO_PRICE'] = ema19 - ema39
        
    if 'NYUPV' in breadth_df.columns and 'NYDNV' in breadth_df.columns:
        print(" -> Synthesizing McClellan Volume Oscillator...")
        net_vol = breadth_df['NYUPV'] - breadth_df['NYDNV']
        ema19_v = net_vol.ewm(span=19, adjust=False).mean()
        ema39_v = net_vol.ewm(span=39, adjust=False).mean()
        breadth_df['MCO_VOLUME'] = ema19_v - ema39_v
        
    print(" -> Synthesizing Advanced A/D Line Mechanics...")
    if 'NYADV' in breadth_df.columns and 'NYDEC' in breadth_df.columns:
        ad_line = (breadth_df['NYADV'] - breadth_df['NYDEC']).cumsum()
        breadth_df['AD_LINE'] = ad_line
        breadth_df['AD_LINE_SMA_200'] = ad_line.rolling(window=200).mean()
        breadth_df['AD_LINE_PCT_SMA'] = ((ad_line - breadth_df['AD_LINE_SMA_200']) / breadth_df['AD_LINE_SMA_200']) * 100
        breadth_df['AD_LINE_5D_ROC'] = ad_line.pct_change(periods=5) * 100
        breadth_df['AD_LINE_10D_ROC'] = ad_line.pct_change(periods=10) * 100
        breadth_df['AD_LINE_20D_ROC'] = ad_line.pct_change(periods=20) * 100
        
    print(" -> Smoothing Options Ratios (CPC / CPCE)...")
    if 'CPC' in breadth_df.columns:
        breadth_df['CPC_SMA_5'] = breadth_df['CPC'].rolling(window=5).mean()
        breadth_df['CPC_5D_ROC'] = breadth_df['CPC_SMA_5'].pct_change(periods=5) * 100
    if 'CPCE' in breadth_df.columns:
        breadth_df['CPCE_SMA_5'] = breadth_df['CPCE'].rolling(window=5).mean()
        breadth_df['CPCE_5D_ROC'] = breadth_df['CPCE_SMA_5'].pct_change(periods=5) * 100
        
    # Standardize Index Names
    breadth_df.index.name = 'Date'
    df = df.join(breadth_df, how='left')
        
    # Extract ALFRED API First-Vintages natively
    alfred_tickers = [
        'RECPROUSM156N', 'BOGMBASE', 'WALCL', 'TREAST', 
        'TSIFRGHT', 'JPNASSETS', 'ECBASSETSW', 
        'DEXJPUS', 'DEXUSEU'
    ]
    alfred_data = {}
    for ticker in alfred_tickers:
        alfred_data[ticker] = fetch_alfred_vintage(ticker)
        
    alfred_df = pd.DataFrame(alfred_data)
    # CRITICAL: We must forward-fill ALFRED natively before cross-multiplying, 
    # because Central Banks report on the 1st of the month (often Weekends) when Forex is closed!
    alfred_df.ffill(inplace=True)
    
    print("[DB Architect] Synthesizing World Central Bank Master Array...")
    boj_usd = alfred_df['JPNASSETS'] / alfred_df['DEXJPUS']
    ecb_usd = alfred_df['ECBASSETSW'] * alfred_df['DEXUSEU']
    fed_usd = alfred_df['WALCL']
    alfred_df['World_CentralBank_BalSh'] = fed_usd + ecb_usd + boj_usd
    
    print("[DB Architect] Executing Phase 130: Principal Component Analysis (PCA)...")
    try:
        from sklearn.decomposition import PCA
        liq_df = alfred_df[['WALCL', 'ECBASSETSW', 'JPNASSETS', 'BOGMBASE']].copy()
        liq_df['ECB_USD'] = liq_df['ECBASSETSW'] * alfred_df['DEXUSEU']
        liq_df['BOJ_USD'] = liq_df['JPNASSETS'] / alfred_df['DEXJPUS']
        
        liq_clean = liq_df[['WALCL', 'ECB_USD', 'BOJ_USD', 'BOGMBASE']].ffill().bfill().fillna(0)
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(liq_clean)
        alfred_df['Global_Liquidity_Variance'] = pca_result.flatten()
        alfred_df['Global_Liquidity_Velocity_21d'] = alfred_df['Global_Liquidity_Variance'].diff(21)
    except ImportError:
        print("[DB Architect] Warning: sklearn not found, passing Phase 130.")
    except Exception as e:
        print(f"[DB Architect] PCA Failed: {e}")
    
    # Merge FRED ALFRED data into primary timeline and strictly fill-forward older prints
    df = df.join(alfred_df, how='left')
    df.ffill(inplace=True)
    
    print("[DB Architect] Synthesizing User-Requested ALFRED Matrix Derivatives...")
    
    # FederalReserveTreasuryHoldings
    df['FederalReserveTreasuryHoldings_20dDiff'] = df['TREAST'].diff(20)
    df['FederalReserveTreasuryHoldings_45d%Chg'] = df['TREAST'].pct_change(45) * 100
    
    # FederalReserveBalanceSheetSize
    df['FederalReserveBalanceSheetSize_20d%Chg'] = df['WALCL'].pct_change(20) * 100
    df['FederalReserveBalanceSheetSize_45d%Chg'] = df['WALCL'].pct_change(45) * 100
    
    # Phase 131: Liquidity Surprise Vector
    df['Fed_Liquidity_Surprise'] = df['WALCL'] - df['WALCL'].rolling(50).mean()
    
    # Phase 134: Global FX Shock Absorbers
    print("[DB Architect] Synthesizing Global FX Margin Call Math...")
    df['FX_DXY_Velocity_20d'] = df['DXY_CLOSE'].pct_change(20) * 100
    df['FX_Yen_Shock_5d'] = df['JPY_CLOSE'].pct_change(5) * 100
    
    # World Central Bank Absolute Balance Sheet
    df['World_CentralBank_BalSh_45d%Chg'] = df['World_CentralBank_BalSh'].pct_change(45) * 100
    
    # Crude Oil (WTI) Demand Trajectories
    df['OIL_45d%Chg'] = df['CL_CLOSE'].pct_change(45) * 100
    df['OIL_90d%Chg'] = df['CL_CLOSE'].pct_change(90) * 100
    df['OIL_180d%Chg'] = df['CL_CLOSE'].pct_change(180) * 100
    
    # Freight Index
    df['Freight_30d%Chg'] = df['TSIFRGHT'].pct_change(30) * 100
    
    # Advanced Multi-Order Momentum Mechanics
    # MonetaryBase_50dMA_20dDiff_10dDiff
    df['MonetaryBase_50dMA'] = df['BOGMBASE'].rolling(window=50).mean()
    df['MonetaryBase_50dMA_20dDiff'] = df['MonetaryBase_50dMA'].diff(20)
    df['MonetaryBase_50dMA_20dDiff_10dDiff'] = df['MonetaryBase_50dMA_20dDiff'].diff(10)
    
    # FederalReserveRecessionProbability_50dMA_5dDiff
    df['FederalReserveRecessionProbability_50dMA'] = df['RECPROUSM156N'].rolling(window=50).mean()
    df['FederalReserveRecessionProbability_50dMA_5dDiff'] = df['FederalReserveRecessionProbability_50dMA'].diff(5)
    
    print("[DB Architect] Extracting Phase 92 Encyclopedia Indicators (TRIX, Donchian, VWAP)...")
    
    # 1. TRIX (15-Day Array)
    df['SPY_TRIX_15'] = calc_trix(df['SPY_CLOSE'], period=15)
    df['TLT_TRIX_15'] = calc_trix(df['TLT_CLOSE'], period=15)
    
    # 2. Donchian Channels (20-Day Array) -> Normalized to percentage offsets for Stationary AI compatibility
    df['SPY_DONCHIAN_UPPER'], df['SPY_DONCHIAN_LOWER'] = calc_donchian(df['SPY_HIGH'], df['SPY_LOW'], period=20)
    df['SPY_PCT_FROM_DONCHIAN_UPPER'] = ((df['SPY_CLOSE'] - df['SPY_DONCHIAN_UPPER']) / df['SPY_DONCHIAN_UPPER']) * 100
    df['SPY_PCT_FROM_DONCHIAN_LOWER'] = ((df['SPY_CLOSE'] - df['SPY_DONCHIAN_LOWER']) / df['SPY_DONCHIAN_LOWER']) * 100
    
    # 3. 20-Day VWAP -> Normalized to percentage offset
    if 'SPY_VOLUME' in df.columns:
        df['SPY_VWAP_20'] = calc_vwap_rolling(df, 'SPY', period=20)
        df['SPY_PCT_FROM_VWAP_20'] = ((df['SPY_CLOSE'] - df['SPY_VWAP_20']) / df['SPY_VWAP_20']) * 100
        
    print("[DB Architect] Synthesizing Phase 122 Trigonometric Seasonality & Real Yields...")
    df['SIN_YEAR'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['COS_YEAR'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    
    if 'T10YIE' in df.columns and 'TNX_CLOSE' in df.columns:
        df['REAL_YIELD_10Y'] = df['TNX_CLOSE'] - df['T10YIE']
        
    print("[DB Architect] Mapping Absolute 63-Day Rolling Correlated Collapse Regimes...")
    if 'SPY_CLOSE' in df.columns and 'TLT_CLOSE' in df.columns and 'GC_CLOSE' in df.columns:
        spy_log_ret = np.log(df['SPY_CLOSE'] / df['SPY_CLOSE'].shift(1))
        tlt_log_ret = np.log(df['TLT_CLOSE'] / df['TLT_CLOSE'].shift(1))
        gc_log_ret = np.log(df['GC_CLOSE'] / df['GC_CLOSE'].shift(1))
        
        corr_spy_tlt = spy_log_ret.rolling(63).corr(tlt_log_ret)
        corr_spy_gc = spy_log_ret.rolling(63).corr(gc_log_ret)
        
        df['CORR_SPY_TLT_63_ZSCORE'] = (corr_spy_tlt - corr_spy_tlt.rolling(252).mean()) / corr_spy_tlt.rolling(252).std()
        df['CORR_SPY_GLD_63_ZSCORE'] = (corr_spy_gc - corr_spy_gc.rolling(252).mean()) / corr_spy_gc.rolling(252).std()
        
    print("[DB Architect] Extracting Phase 123 CVR3 Absolute Geometric Vectors...")
    if 'VIX_CLOSE' in df.columns:
        # VIX Bollinger Bands (20, 2)
        df['VIX_SMA20'] = df['VIX_CLOSE'].rolling(window=20).mean()
        df['VIX_STD20'] = df['VIX_CLOSE'].rolling(window=20).std()
        df['VIX_BB_UPPER'] = df['VIX_SMA20'] + (df['VIX_STD20'] * 2)
        df['VIX_BB_LOWER'] = df['VIX_SMA20'] - (df['VIX_STD20'] * 2)
        
        # Absolute Distance and Width
        df['VIX_DIST_UPPER'] = (df['VIX_CLOSE'] - df['VIX_BB_UPPER']) / df['VIX_BB_UPPER']
        df['VIX_DIST_LOWER'] = (df['VIX_CLOSE'] - df['VIX_BB_LOWER']) / df['VIX_BB_LOWER']
        df['VIX_BB_WIDTH'] = (df['VIX_BB_UPPER'] - df['VIX_BB_LOWER']) / df['VIX_SMA20']

        # CVR3 Tri-Confluence Exhaustion Triggers
        vix_ema10 = df['VIX_CLOSE'].ewm(span=10, adjust=False).mean()
        vix_ppo = ((df['VIX_CLOSE'] - vix_ema10) / vix_ema10) * 100
        
        buy_spatial = df['VIX_LOW'] > vix_ema10
        buy_magnitude = vix_ppo >= 10.0
        buy_morphology = df['VIX_CLOSE'] < df['VIX_OPEN']
        
        df['CVR3_BUY_SIGNAL'] = ((buy_spatial.rolling(3).max() == 1) & 
                                 (buy_magnitude.rolling(3).max() == 1) & 
                                 (buy_morphology.rolling(3).max() == 1)).astype(int)
                                 
        sell_spatial = df['VIX_HIGH'] < vix_ema10
        sell_magnitude = vix_ppo <= -10.0
        sell_morphology = df['VIX_CLOSE'] > df['VIX_OPEN']
        
        df['CVR3_SELL_SIGNAL'] = ((sell_spatial.rolling(3).max() == 1) & 
                                  (sell_magnitude.rolling(3).max() == 1) & 
                                  (sell_morphology.rolling(3).max() == 1)).astype(int)
    
    # Drop rows that contain NaNs due to TSI EMA spin-ups and rolling windows
    null_counts = df.isnull().sum()
    print("[DB Architect] NaN distribution before dropna:")
    print(null_counts[null_counts > 0])
    df.dropna(inplace=True)
    
    # Force Datetime Index structure to securely prevent SQL logic crashes
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    print("[DB Architect] Committing core_market_table ledger...")
    df.to_sql('core_market_table', conn, if_exists='replace')
    
    print(f"[DB Architect] Successfully serialized {len(df)} core market instances.")
    print(f"[DB Architect] Database firmly locked at: {db_path}")
    conn.close()

if __name__ == "__main__":
    build_database()

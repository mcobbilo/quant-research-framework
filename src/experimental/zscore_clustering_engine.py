import pandas as pd
import numpy as np
import sqlite3
import os
import matplotlib.pyplot as plt

warnings = pd.options.mode.chained_assignment = None

def calculate_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stochastic(high, low, close, window):
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    return 100 * ((close - lowest_low) / (highest_high - lowest_low))

def calculate_tsi(close, r, s):
    m = close.diff()
    m1 = m.ewm(span=r, adjust=False).mean()
    m2 = m1.ewm(span=s, adjust=False).mean()
    abs_m = m.abs()
    abs_m1 = abs_m.ewm(span=r, adjust=False).mean()
    abs_m2 = abs_m1.ewm(span=s, adjust=False).mean()
    return 100 * (m2 / abs_m2)

def run_clustering_engine(z_window=252):
    if z_window == -1:
        print("\n[Z-Score Cluster Engine] Formulating EXPANDING (Percentile Rank) Arrays...")
    elif z_window == 0:
        print("\n[Z-Score Cluster Engine] Formulating EXPANDING (Cumulative Cumulative) Arrays...")
    else:
        print(f"\n[Z-Score Cluster Engine] Formulating {z_window}-Day Arrays...")
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_data.db')
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM core_market_table', conn, index_col='Date')
    df.index = pd.to_datetime(df.index)
    
    # Analyze multiple timeframes explicitly as requested
    feature_cols = []
    
    # 1. SPY SMA Percent Displacement [10, 20, 50, 100, 200, 252]
    for w in [10, 20, 50, 100, 200, 252]:
        sma = df['SPY_CLOSE'].rolling(w).mean()
        df[f'SPY_PCT_SMA_{w}'] = (df['SPY_CLOSE'] - sma) / sma
        feature_cols.append(f'SPY_PCT_SMA_{w}')
        
    # 2. SPY RSI [10, 20, 50, 100, 200]
    for w in [10, 20, 50, 100, 200]:
        df[f'SPY_RSI_{w}'] = calculate_rsi(df['SPY_CLOSE'], w)
        feature_cols.append(f'SPY_RSI_{w}')
        
    # 3. SPY Stochastics [5, 10, 20, 50, 100, 200, 252]
    for w in [5, 10, 20, 50, 100, 200, 252]:
        df[f'SPY_STOCH_{w}'] = calculate_stochastic(df['SPY_HIGH'], df['SPY_LOW'], df['SPY_CLOSE'], w)
        feature_cols.append(f'SPY_STOCH_{w}')
        
    # 4. Normal SPY TSI + Signal Line
    df['SPY_TSI_25_13'] = calculate_tsi(df['SPY_CLOSE'], 25, 13)
    df['SPY_TSI_SIGNAL_13'] = df['SPY_TSI_25_13'].ewm(span=13, adjust=False).mean()
    feature_cols.extend(['SPY_TSI_25_13', 'SPY_TSI_SIGNAL_13'])
    
    # 5. VIX/TNX SMA Percent Displacement [5, 10, 20, 50, 100, 200, 252]
    vix_tnx = df['VIX_CLOSE'] / df['TNX_CLOSE']
    for w in [5, 10, 20, 50, 100, 200, 252]:
        vix_tnx_sma = vix_tnx.rolling(w).mean()
        df[f'VIX_TNX_PCT_SMA_{w}'] = (vix_tnx - vix_tnx_sma) / vix_tnx_sma
        feature_cols.append(f'VIX_TNX_PCT_SMA_{w}')
        
    # 6. High Yield Corporate Credit Spreads (BAMLC0A0CM) moving averages
    if 'BAMLC0A0CM' in df.columns:
        df['BAMLC0A0CM_SMA_50'] = df['BAMLC0A0CM'].rolling(50).mean()
        df['BAMLC0A0CM_SMA_200'] = df['BAMLC0A0CM'].rolling(200).mean()

    # Add Raw Macro Indicators to feature array
    macro_cols = ['BAMLC0A0CM', 'BAMLC0A0CM_SMA_50', 'BAMLC0A0CM_SMA_200', 'T10Y2Y', 'T10YFF', 'VIX_MOVE_SPREAD_5D', 'VIX_MOVE_SPREAD_10D',
                  'NYA200R', 'CPC', 'CPCE', 'MCO_PRICE', 'MCO_VOLUME',
                  'AD_LINE_PCT_SMA', 'AD_LINE_5D_ROC', 'AD_LINE_10D_ROC', 'AD_LINE_20D_ROC', 
                  'CPC_5D_ROC', 'CPCE_5D_ROC']
    for col in macro_cols:
        if col in df.columns:
            feature_cols.append(col)

    # 2. Convert ALL feature columns to strictly rolling Z-scores (No Lookahead Bias!)
    z_score_cols = []
    
    if z_window == -1:
        print(f"[Z-Score Cluster Engine] Computing EXPANDING Percentiles for {len(feature_cols)} Vectors...")
        for col in feature_cols:
            p_col = f'P_{col}'
            # Calculate the percentage of historical values that are strictly <= the current Day's closing value
            df[p_col] = df[col].expanding(min_periods=252).apply(lambda x: (x <= x[-1]).mean() * 100, raw=True)
            z_score_cols.append(p_col)
    elif z_window == 0:
        print(f"[Z-Score Cluster Engine] Computing EXPANDING Exact Z-Scores for {len(feature_cols)} Vectors...")
        for col in feature_cols:
            # Requires at least 1 year (252 days) of warmup before generating a valid standard deviation
            roll_mean = df[col].expanding(min_periods=252).mean().shift(1)
            roll_std = df[col].expanding(min_periods=252).std().shift(1)
            z_col = f'Z_{col}'
            df[z_col] = (df[col] - roll_mean) / roll_std
            z_score_cols.append(z_col)
    elif z_window == -2:
        import scipy.stats as stats
        print(f"[Z-Score Cluster Engine] Computing ROLLING 252-Day Heavy-Tail (Student-T df=3) Transformed Z-Scores...")
        for col in feature_cols:
            roll_mean = df[col].rolling(252).mean().shift(1)
            roll_std = df[col].rolling(252).std().shift(1)
            raw_z = (df[col] - roll_mean) / roll_std
            
            # THE HEAVY-TAIL TRANSFORMATION MATH:
            # 1. Evaluate the probability of the event under a Fat-Tailed (Student-T) environment.
            # 2. Map that probability back to an inverse Normal Gaussian Z-Score.
            # This mathematically compresses impossible Outliers (e.g. Z = 25) into stable domains (Z = 3.5), 
            # allowing clustering engines to trigger without breaking.
            
            df[f'Z_{col}'] = stats.norm.ppf(stats.t.cdf(raw_z, df=3))
            z_score_cols.append(f'Z_{col}')
    else:
        print(f"[Z-Score Cluster Engine] Computing Rolling {z_window}-Day Exact Gaussian Z-Scores for {len(feature_cols)} Vectors...")
        for col in feature_cols:
            roll_mean = df[col].rolling(z_window).mean().shift(1) # Prevent lookahead
            roll_std = df[col].rolling(z_window).std().shift(1)
            z_col = f'Z_{col}'
            df[z_col] = (df[col] - roll_mean) / roll_std
            z_score_cols.append(z_col)
        
    df = df.dropna(subset=z_score_cols).copy()
    
    # 3. The Clustering Logic ("Count of Extreme Standard Deviation Events")
    # Finding clusters of massive panic: 
    # -> SPY heavily oversold (Z < -2.5) 
    # -> VIX ratio violently skyrocketing (Z > +2.5)
    
    panic_conditions = []
    euphoria_conditions = []
    
    # Explicitly map directionality of standard deviation extremes for panic states
    upside_panic_keywords = ['VIX', 'BAMLC0A0CM', 'CPC', 'CPCE'] # VIX escalating, Credit Spreads blowing out, Puts skyrocketing
    
    for col in z_score_cols:
        is_upside = any(keyword in col for keyword in upside_panic_keywords)
        is_percentile = col.startswith('P_')
        
        if is_upside:
            if is_percentile:
                panic_conditions.append(df[col] >= 99.0)
                euphoria_conditions.append(df[col] <= 1.0)
            else:
                panic_conditions.append(df[col] > 2.5)
                # Euphoria: Volatility collapses, credit spreads are tight, no one buying puts
                euphoria_conditions.append(df[col] < -2.5)
        else:
            if is_percentile:
                panic_conditions.append(df[col] <= 1.0)
                euphoria_conditions.append(df[col] >= 99.0)
            else:
                # Everything else (SPY dropping, Treasury Yields collapsing) generates a panic trigger on the downside.
                panic_conditions.append(df[col] < -2.5)
                # Euphoria: Equities rocketing upward
                euphoria_conditions.append(df[col] > 2.5)
            
    # Sum the booleans vertically to create the "Conditional Formatting Visual Heatmap" Score!
    extreme_matrix = pd.concat(panic_conditions, axis=1)
    df['Panic_Cluster_Score'] = extreme_matrix.sum(axis=1)
    
    euphoria_matrix = pd.concat(euphoria_conditions, axis=1)
    df['Euphoria_Cluster_Score'] = euphoria_matrix.sum(axis=1)
    
    max_cluster = df['Panic_Cluster_Score'].max()
    max_euphoria = df['Euphoria_Cluster_Score'].max()
    
    print(f"Max Panic Score recorded in 25 Years: {max_cluster} / {len(feature_cols)} aligning simultaneously")
    print(f"Max Euphoria Score recorded in 25 Years: {max_euphoria} / {len(feature_cols)} aligning simultaneously")
    
    # 4. Trigger logic (e.g. 5 indicators must violently breach 2.5 sigma on the exact same day)
    # This precisely mimics visual "dark green/red blocks" clustering in Excel.
    df['Is_Trigger'] = df['Panic_Cluster_Score'] >= 5
    
    trigger_dates = df[df['Is_Trigger']].index
    
    # Prevent identical clustering dates (lockout for 10 days)
    filtered_dates = []
    for d in trigger_dates:
        if not filtered_dates or (d - filtered_dates[-1]).days > 10:
            filtered_dates.append(d)
            
    print(f"Found {len(filtered_dates)} Isolated Crash Events where >= 5 Technicals breached 2.5 Sigma thresholds simultaneously.\n")
    
    # 5. Execute 1.0x Out-Of-Sample Strategy
    positions = np.zeros(len(df))
    strategy_returns = np.zeros(len(df))
    
    in_trade = False
    trade_count = 0
    days_held = []
    current_hold = 0
    
    # Normalization Exit Re-engaged (Strategically superior to Euphoria targets)
    
    for i in range(1, len(df)-1):
        today_open = df['SPY_OPEN'].iloc[i]
        today_close = df['SPY_CLOSE'].iloc[i]
        yesterday_close = df['SPY_CLOSE'].iloc[i-1]
        
        # Execute exactly at Trade Open T+1
        if df.index[i-1] in filtered_dates and not in_trade:
            in_trade = True
            trade_count += 1
            current_hold = 0
            if today_open > 0:
                strategy_returns[i] = (today_close - today_open) / today_open
            positions[i] = 1.0
            
        elif in_trade:
            # ORIGINAL RULE: Revert to Cash the moment the Heatmap Score drops below 1
            if df['Panic_Cluster_Score'].iloc[i-1] < 1:
                in_trade = False
                days_held.append(current_hold)
                current_hold = 0
                if yesterday_close > 0:
                    strategy_returns[i] = (today_open - yesterday_close) / yesterday_close
                positions[i] = 0.0
            else:
                if yesterday_close > 0:
                    strategy_returns[i] = (today_close - yesterday_close) / yesterday_close
                positions[i] = 1.0
                current_hold += 1

    df['Position'] = positions
    df['Strategy_Returns_Raw'] = strategy_returns
    
    # Execution constraints
    trades = df['Position'].diff().abs()
    cost_per_trade = 0.001
    df['Net_Returns'] = df['Strategy_Returns_Raw'] - (trades * cost_per_trade) 
    
    risk_free_rate = 0.03 / 252.0  
    df['Cash_Yield'] = np.where(df['Position'] == 0, risk_free_rate, 0.0)
    
    # Compounding Math
    df['Strat_Growth'] = (1 + df['Net_Returns'] + df['Cash_Yield']).cumprod()
    df['SPY_Growth'] = (1 + df['SPY_CLOSE'].pct_change().fillna(0)).cumprod()
    
    spy_final = df['SPY_Growth'].iloc[-1]
    strat_final = df['Strat_Growth'].iloc[-1]
    
    spy_cummax = df['SPY_Growth'].cummax()
    spy_max_dd = ((df['SPY_Growth'] - spy_cummax) / spy_cummax).min() * 100
    strat_cummax = df['Strat_Growth'].cummax()
    strat_max_dd = ((df['Strat_Growth'] - strat_cummax) / strat_cummax).min() * 100
    avg_hold = sum(days_held) / len(days_held) if days_held else 0
    
    print("=======================================================")
    print("   Z-SCORE CLUSTERING ENGINE (NORMALIZATION EXIT)      ")
    print("=======================================================")
    print(f" Analyzed Variables: {len(feature_cols)} Independent Time-Series")
    if z_window == -1:
        print(f" Logic Matrix:       Trigger if >= 5 PANIC arrays hit <1% or >99% Rank simultaneously")
    else:
        print(f" Logic Matrix:       Trigger if >= 5 PANIC arrays breach 2.5 Sigma simultaneously")
    print(f" Exit Protocol:      Hold until Heatmap Score Normalizes (<1)")
    print(f"")
    print(f" Total Triggers:     {trade_count} Cluster Events")
    print(f" Average Hold:       {avg_hold:.1f} Trading Days")
    print(f" Leveraging:         1.0x (No margin) / Yielding 3% Risk-Free in Cash")
    print(f"")
    print(f" [BENCHMARK] SPY Buy/Hold:  ${(10000 * spy_final):,.2f}  Return: {spy_final*100-100:>6.1f}% | Max DD: {spy_max_dd:>6.1f}%")
    print(f" [ALGORITHM] Z-Cluster:     ${(10000 * strat_final):,.2f}  Return: {strat_final*100-100:>6.1f}% | Max DD: {strat_max_dd:>6.1f}%")
    print("=======================================================\n")

    # Generate the requested comparative visual artifact
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['SPY_Growth'] * 10000, label=f'SPY Buy & Hold (Max DD: {spy_max_dd:.1f}%)', color='blue', alpha=0.6)
    
    if z_window == -2:
        plt.plot(df.index, df['Strat_Growth'] * 10000, label=f'Student-T Z-Score Engine (Max DD: {strat_max_dd:.1f}%)', color='purple', linewidth=2)
        plt.title('Heavy-Tailed Z-Score Engine vs S&P 500 Total Return', fontsize=14, pad=15)
        chart_path = '/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/equity_curve_heavy_tail.png'
    else:
        plt.plot(df.index, df['Strat_Growth'] * 10000, label=f'Gaussian Z-Score Engine (Max DD: {strat_max_dd:.1f}%)', color='green', linewidth=2)
        plt.title('Classic Gaussian Z-Score Engine vs S&P 500 Total Return', fontsize=14, pad=15)
        chart_path = '/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/equity_curve_gaussian.png'
        
    plt.ylabel('Portfolio Value ($10,000 Starting)', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    print(f"Saved comparative equity curve to: {chart_path}")

if __name__ == "__main__":
    import sys
    window = int(sys.argv[1]) if len(sys.argv) > 1 else 252
    run_clustering_engine(window)

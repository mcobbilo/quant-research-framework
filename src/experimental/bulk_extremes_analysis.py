import pandas as pd
import numpy as np
import sqlite3
import os
import warnings
warnings.filterwarnings('ignore')
from zscore_clustering_engine import calculate_rsi, calculate_stochastic, calculate_tsi

def run_bulk_analysis():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_data.db')
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM core_market_table', conn, index_col='Date')
    df.index = pd.to_datetime(df.index)
    
    feature_cols = []
    
    # 1. SPY SMA
    for w in [10, 20, 50, 100, 200, 252]:
        sma = df['SPY_CLOSE'].rolling(w).mean()
        col = f'SPY_PCT_SMA_{w}'
        df[col] = (df['SPY_CLOSE'] - sma) / sma
        feature_cols.append(col)
        
    # 2. SPY RSI
    for w in [10, 20, 50, 100, 200]:
        col = f'SPY_RSI_{w}'
        df[col] = calculate_rsi(df['SPY_CLOSE'], w)
        feature_cols.append(col)
        
    # 3. SPY Stochastics
    for w in [5, 10, 20, 50, 100, 200, 252]:
        col = f'SPY_STOCH_{w}'
        df[col] = calculate_stochastic(df['SPY_HIGH'], df['SPY_LOW'], df['SPY_CLOSE'], w)
        feature_cols.append(col)
        
    # 4. TSI
    df['SPY_TSI_25_13'] = calculate_tsi(df['SPY_CLOSE'], 25, 13)
    df['SPY_TSI_SIGNAL_13'] = df['SPY_TSI_25_13'].ewm(span=13, adjust=False).mean()
    feature_cols.extend(['SPY_TSI_25_13', 'SPY_TSI_SIGNAL_13'])
    
    # 5. VIX/TNX
    vix_tnx = df['VIX_CLOSE'] / df['TNX_CLOSE']
    df['VIX_TNX_TSI'] = calculate_tsi(vix_tnx, 25, 13)
    feature_cols.append('VIX_TNX_TSI')
    for w in [5, 10, 20, 50, 100, 200, 252]:
        vix_tnx_sma = vix_tnx.rolling(w).mean()
        col = f'VIX_TNX_PCT_SMA_{w}'
        df[col] = (vix_tnx - vix_tnx_sma) / vix_tnx_sma
        feature_cols.append(col)
        
    # 6. SPY vs VUSTX Diff
    df['SPY_VUSTX_DIFF_5D'] = df['SPY_CLOSE'].pct_change(5) - df['VUSTX_CLOSE'].pct_change(5)
    df['SPY_VUSTX_DIFF_10D'] = df['SPY_CLOSE'].pct_change(10) - df['VUSTX_CLOSE'].pct_change(10)
    feature_cols.extend(['SPY_VUSTX_DIFF_5D', 'SPY_VUSTX_DIFF_10D'])
    
    macro_cols = ['BAMLC0A0CM', 'T10Y2Y', 'T10YFF', 'VIX_MOVE_SPREAD_5D', 'VIX_MOVE_SPREAD_10D',
                  'NYA200R', 'CPC', 'CPCE', 'MCO_PRICE', 'MCO_VOLUME',
                  'AD_LINE_PCT_SMA', 'AD_LINE_5D_ROC', 'AD_LINE_10D_ROC', 'AD_LINE_20D_ROC', 
                  'CPC_5D_ROC', 'CPCE_5D_ROC']
    for col in macro_cols:
        if col in df.columns:
            feature_cols.append(col)

    # Forward Returns
    df['Fwd_5D'] = (df['SPY_CLOSE'].shift(-5) / df['SPY_CLOSE']) - 1
    df['Fwd_10D'] = (df['SPY_CLOSE'].shift(-10) / df['SPY_CLOSE']) - 1
    df['Fwd_20D'] = (df['SPY_CLOSE'].shift(-20) / df['SPY_CLOSE']) - 1
    df['Fwd_60D'] = (df['SPY_CLOSE'].shift(-60) / df['SPY_CLOSE']) - 1

    results = []

    print(f"Executing Top 50 / Bottom 50 Extrema Maps across {len(feature_cols)} Target Vectors...")

    for col in feature_cols:
        # Require target variable AND the 60D forward return to exist (preventing dropping the last 60 rows for everything)
        temp_df = df.dropna(subset=[col, 'Fwd_60D']).copy() 
        if len(temp_df) < 100:
            continue
            
        top_50 = temp_df.nlargest(50, col)
        bottom_50 = temp_df.nsmallest(50, col)
        
        # Assemble Top 50 Analytics
        results.append({
            'Target_Variable': col,
            'Execution_Extreme': 'Top 50 (Highest Readings)',
            'Average_Setup_Val': top_50[col].mean(),
            'Fwd_5D_Ret_%': top_50['Fwd_5D'].mean() * 100,
            'Fwd_5D_Win_%': (top_50['Fwd_5D'] > 0).mean() * 100,
            'Fwd_10D_Ret_%': top_50['Fwd_10D'].mean() * 100,
            'Fwd_10D_Win_%': (top_50['Fwd_10D'] > 0).mean() * 100,
            'Fwd_20D_Ret_%': top_50['Fwd_20D'].mean() * 100,
            'Fwd_20D_Win_%': (top_50['Fwd_20D'] > 0).mean() * 100,
            'Fwd_60D_Ret_%': top_50['Fwd_60D'].mean() * 100,
            'Fwd_60D_Win_%': (top_50['Fwd_60D'] > 0).mean() * 100,
        })
        
        # Assemble Bottom 50 Analytics
        results.append({
            'Target_Variable': col,
            'Execution_Extreme': 'Bottom 50 (Lowest Readings)',
            'Average_Setup_Val': bottom_50[col].mean(),
            'Fwd_5D_Ret_%': bottom_50['Fwd_5D'].mean() * 100,
            'Fwd_5D_Win_%': (bottom_50['Fwd_5D'] > 0).mean() * 100,
            'Fwd_10D_Ret_%': bottom_50['Fwd_10D'].mean() * 100,
            'Fwd_10D_Win_%': (bottom_50['Fwd_10D'] > 0).mean() * 100,
            'Fwd_20D_Ret_%': bottom_50['Fwd_20D'].mean() * 100,
            'Fwd_20D_Win_%': (bottom_50['Fwd_20D'] > 0).mean() * 100,
            'Fwd_60D_Ret_%': bottom_50['Fwd_60D'].mean() * 100,
            'Fwd_60D_Win_%': (bottom_50['Fwd_60D'] > 0).mean() * 100,
        })

    final_df = pd.DataFrame(results)
    
    desktop_path = os.path.expanduser('~/Desktop/bulk_statistical_edges.csv')
    # Round heavily to preserve Google Sheets numeric legibility natively
    final_df.round(2).to_csv(desktop_path, index=False)
    
    print("==========================================================")
    print(f" BULK EXTRACTION COMPLETE!")
    print(f" Analyzed Variables: {len(feature_cols)} Unique Timeseries")
    print(f" Mapped Physics:     {len(final_df)} Extrema Footprints")
    print(f" Extracted Payload:  {desktop_path}")
    print("==========================================================")

if __name__ == '__main__':
    run_bulk_analysis()

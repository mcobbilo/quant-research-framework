import pandas as pd
import numpy as np
import yfinance as yf
import os

def analyze_vvix():
    print("Downloading ^VVIX, ^VIX, and SPY...")
    tickers = ['^VVIX', '^VIX', 'SPY']
    data = yf.download(tickers, period='max', auto_adjust=False)['Close']
    data = data.dropna() # VVIX data begins around Jan 2007

    df = pd.DataFrame(index=data.index)
    df['VVIX'] = data['^VVIX']
    df['VIX'] = data['^VIX']
    df['SPY_CLOSE'] = data['SPY']

    feature_cols = []

    # 1. VVIX SMAs and % Above/Below
    for w in [5, 10, 20, 50, 100]:
        sma = df['VVIX'].rolling(w).mean()
        col = f'VVIX_PCT_SMA_{w}'
        df[col] = (df['VVIX'] - sma) / sma
        feature_cols.append(col)

    # 2. VVIX / VIX Ratio
    df['VVIX_VIX_RATIO'] = df['VVIX'] / df['VIX']
    feature_cols.append('VVIX_VIX_RATIO')

    # Forward Returns for SPY
    df['Fwd_5D'] = (df['SPY_CLOSE'].shift(-5) / df['SPY_CLOSE']) - 1
    df['Fwd_10D'] = (df['SPY_CLOSE'].shift(-10) / df['SPY_CLOSE']) - 1
    df['Fwd_20D'] = (df['SPY_CLOSE'].shift(-20) / df['SPY_CLOSE']) - 1
    df['Fwd_60D'] = (df['SPY_CLOSE'].shift(-60) / df['SPY_CLOSE']) - 1

    out_lines = []
    out_lines.append("=======================================================")
    out_lines.append("     VVIX STRUCTURAL ANOMALY ISOLATION MATRIX")
    out_lines.append("=======================================================\n")

    for col in feature_cols:
        temp_df = df.dropna(subset=[col, 'Fwd_60D']).copy()
        if len(temp_df) < 100:
            continue
            
        top_50 = temp_df.nlargest(50, col)
        bottom_50 = temp_df.nsmallest(50, col)

        # TOP 50
        out_lines.append(f"--- [ {col} ] : TOP 50 HIGHEST READINGS ---")
        out_lines.append(f"Average Extreme Value: {top_50[col].mean():.3f}")
        out_lines.append(f"-> Fwd  5-Day Return: {top_50['Fwd_5D'].mean()*100:>6.2f}%  (Win Rate: {(top_50['Fwd_5D'] > 0).mean()*100:.1f}%)")
        out_lines.append(f"-> Fwd 10-Day Return: {top_50['Fwd_10D'].mean()*100:>6.2f}%  (Win Rate: {(top_50['Fwd_10D'] > 0).mean()*100:.1f}%)")
        out_lines.append(f"-> Fwd 20-Day Return: {top_50['Fwd_20D'].mean()*100:>6.2f}%  (Win Rate: {(top_50['Fwd_20D'] > 0).mean()*100:.1f}%)")
        out_lines.append(f"-> Fwd 60-Day Return: {top_50['Fwd_60D'].mean()*100:>6.2f}%  (Win Rate: {(top_50['Fwd_60D'] > 0).mean()*100:.1f}%)\n")

        # BOTTOM 50
        out_lines.append(f"--- [ {col} ] : BOTTOM 50 LOWEST READINGS ---")
        out_lines.append(f"Average Extreme Value: {bottom_50[col].mean():.3f}")
        out_lines.append(f"-> Fwd  5-Day Return: {bottom_50['Fwd_5D'].mean()*100:>6.2f}%  (Win Rate: {(bottom_50['Fwd_5D'] > 0).mean()*100:.1f}%)")
        out_lines.append(f"-> Fwd 10-Day Return: {bottom_50['Fwd_10D'].mean()*100:>6.2f}%  (Win Rate: {(bottom_50['Fwd_10D'] > 0).mean()*100:.1f}%)")
        out_lines.append(f"-> Fwd 20-Day Return: {bottom_50['Fwd_20D'].mean()*100:>6.2f}%  (Win Rate: {(bottom_50['Fwd_20D'] > 0).mean()*100:.1f}%)")
        out_lines.append(f"-> Fwd 60-Day Return: {bottom_50['Fwd_60D'].mean()*100:>6.2f}%  (Win Rate: {(bottom_50['Fwd_60D'] > 0).mean()*100:.1f}%)\n")

    out_path = os.path.join(os.path.dirname(__file__), 'vvix_metrics.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
        
    print(f"VVIX Extraction Complete. Output written to {out_path}.")

if __name__ == '__main__':
    analyze_vvix()

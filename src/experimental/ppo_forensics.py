import os
import sys
import pandas as pd
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hardcoded_wrapper import attach_features

def run_ppo_study():
    print("[Forensics] Indexing 25-Year Data History for PPO Extrema...")
    spy_data = yf.download("SPY", start="2000-01-01", end="2025-01-01", progress=False, auto_adjust=True)
    vix_data = yf.download("^VIX", start="2000-01-01", end="2025-01-01", progress=False)
    gold_data = yf.download("GC=F", start="2000-01-01", end="2025-01-01", progress=False)
    copper_data = yf.download("HG=F", start="2000-01-01", end="2025-01-01", progress=False)
    
    if isinstance(spy_data.columns, pd.MultiIndex):
        df = pd.DataFrame(index=spy_data.index)
        df["SPY"] = spy_data[('Close', 'SPY')]
        df["VIX"] = vix_data[('Close', '^VIX')]
        df["VIX_OPEN"] = vix_data[('Open', '^VIX')]
        df["VIX_HIGH"] = vix_data[('High', '^VIX')]
        df["VIX_LOW"] = vix_data[('Low', '^VIX')]
        df["GOLD"] = gold_data[('Close', 'GC=F')]
        df["COPPER"] = copper_data[('Close', 'HG=F')]
    else:
        df = pd.DataFrame(index=spy_data.index)
        df["SPY"] = spy_data['Close']
        df["VIX"] = vix_data['Close']
        df["VIX_OPEN"] = vix_data['Open']
        df["VIX_HIGH"] = vix_data['High']
        df["VIX_LOW"] = vix_data['Low']
        df["GOLD"] = gold_data['Close']
        df["COPPER"] = copper_data['Close']
        
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    df = attach_features(df)
    
    # Calculate geometric forward returns structurally mapped to T+3 and T+5
    df['SPY_T+3_RET'] = (df['SPY'].shift(-3) - df['SPY']) / df['SPY'] * 100
    df['SPY_T+5_RET'] = (df['SPY'].shift(-5) - df['SPY']) / df['SPY'] * 100
    
    # Drop absolute tail rows that lack forward data
    df.dropna(subset=['SPY_T+5_RET'], inplace=True)
    
    # Sort the global timeline logically by standard PPO descending
    df_sorted = df.sort_values(by='VIX_PPO_7', ascending=False)
    
    # De-duplicate adjacent days. If a crash lasts 4 days, we only want the absolute mathematical peak PPO day.
    top_50 = []
    for index, row in df_sorted.iterrows():
        is_isolated = True
        for (t_date, _) in top_50:
            if abs((index - t_date).days) < 15: # 15 calendar days buffer between macro events
                is_isolated = False
                break
        if is_isolated:
            top_50.append((index, row))
        if len(top_50) >= 50:
            break
            
    print(f"\n=========================================================================")
    print(f"Top 50 Distinct Baseline PPO Spikes (25-Year History)")
    print(f"{'Date':<12} | {'VIX PPO_7':<10} | {'SPY Return (T+3)':<18} | {'SPY Return (T+5)':<18}")
    print(f"-------------------------------------------------------------------------")
    
    avg_3d = []
    avg_5d = []
    win_3d = 0
    win_5d = 0
    
    # We re-sort the Top 50 chronologically so it's readable for the user
    top_50_chrono = sorted(top_50, key=lambda x: x[0])
    
    for (idx, r) in top_50_chrono:
        ppo = r['VIX_PPO_7']
        ret3 = r['SPY_T+3_RET']
        ret5 = r['SPY_T+5_RET']
        
        avg_3d.append(ret3)
        avg_5d.append(ret5)
        if ret3 > 0: win_3d += 1
        if ret5 > 0: win_5d += 1
        
        print(f"{idx.strftime('%Y-%m-%d'):<12} | {ppo:>8.2f}%  | {ret3:>13.2f}%    | {ret5:>13.2f}%")
        
    print(f"-------------------------------------------------------------------------")
    print(f"AVERAGE T+3 RETURN: {sum(avg_3d)/len(avg_3d):.2f}% (Win Rate: {win_3d/len(avg_3d)*100:.1f}%)")
    print(f"AVERAGE T+5 RETURN: {sum(avg_5d)/len(avg_5d):.2f}% (Win Rate: {win_5d/len(avg_5d)*100:.1f}%)")
    print(f"=========================================================================")

if __name__ == "__main__":
    run_ppo_study()

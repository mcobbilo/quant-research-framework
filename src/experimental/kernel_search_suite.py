import os
import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import entropy

def calculate_hurst(ts):
    """ Calculate Hurst Exponent (Fractal Dimension proxy) """
    if len(ts) < 20: return 0.5
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def evaluate_kernels():
    db_path = os.path.join("src", "data", "market_data.db")
    if not os.path.exists(db_path):
        print("[ERROR] No database found.")
        return
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM core_market_table", conn)
    conn.close()
    
    df = df.dropna(subset=['SPY_CLOSE']).copy()
    df['SPY_Ret'] = df['SPY_CLOSE'].pct_change()
    df['Target'] = df['SPY_Ret'].shift(-10).rolling(10).sum() # Forward 10D return
    df = df.dropna()
    
    print(f"Evaluating 18th Dimension Candidates on {len(df)} days...")
    
    # 1. Shannon Entropy of SPY Returns (Disorder)
    def shannon_entropy(x):
        if len(x) < 2: return 0.0
        hist, _ = np.histogram(x, bins=10, density=True)
        return entropy(hist)
    
    df['Entropy_SPY'] = df['SPY_Ret'].rolling(63).apply(shannon_entropy, raw=True)
    
    # 2. Hurst Exponent of SPY (Persistence)
    df['Hurst_SPY'] = df['SPY_CLOSE'].rolling(63).apply(calculate_hurst, raw=False)
    
    # 3. Breadth Divergence Volatility (Idea 1 expansion)
    # Correlation between ADV and SPY
    df['Breadth_Corr'] = df['NYADV'].rolling(63).corr(df['SPY_CLOSE'])
    
    df = df.dropna()
    
    candidates = ['Entropy_SPY', 'Hurst_SPY', 'Breadth_Corr']
    results = {}
    
    for c in candidates:
        corr = df[c].corr(df['Target'])
        results[c] = abs(corr)
        print(f" -> {c}: Abs Correlation with 10D Target: {abs(corr):.4f}")
        
    winner = max(results, key=results.get)
    print(f"\n[WINNER] Selected Curiosity Kernel: {winner}")
    return winner

if __name__ == "__main__":
    evaluate_kernels()

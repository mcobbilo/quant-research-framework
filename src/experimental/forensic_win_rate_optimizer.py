import pandas as pd
import sqlite3
import os
from sklearn.tree import DecisionTreeClassifier, export_text

def optimize_win_rate():
    print("""
    =========================================================
    [Win-Rate Matrix] Initiating Secondary Filter Scanner
    =========================================================
    """)
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_data.db')
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM core_market_table', conn, index_col='Date')
    df.index = pd.to_datetime(df.index)
    
    # Calculate Target Output: T+5 SPY Return
    df['SPY_FWD_5D'] = (df['SPY_CLOSE'].shift(-5) - df['SPY_CLOSE']) / df['SPY_CLOSE'] * 100
    
    # Isolate original 100 anomalous events
    sorted_df = df.dropna(subset=['VIX_TNX_PPO_7']).sort_values(by='VIX_TNX_PPO_7', ascending=False)
    top_dates = []
    used_dates = []
    
    for idx, row in sorted_df.iterrows():
        if len(top_dates) >= 100: break
        conflict = False
        for used in used_dates:
            if abs((idx - used).days) <= 10:
                conflict = True
                break
        if not conflict:
            top_dates.append(idx)
            used_dates.append(idx)
            
    events_df = df.loc[top_dates].copy()
    
    # Security Rule: We can only analyze indicators strictly at Time T (or T-1). 
    # Analyzing T+1 creates illegal lookahead bias.
    
    # Drop raw prices from features—we only want to analyze normalized technical/macro indicators
    drop_cols = ['SPY_CLOSE', 'SPY_OPEN', 'SPY_HIGH', 'SPY_LOW', 
                 'VIX_CLOSE', 'VIX_OPEN', 'VIX_HIGH', 'VIX_LOW',
                 'GC=F_CLOSE', 'GC=F_OPEN', 'GC=F_HIGH', 'GC=F_LOW',
                 'VUSTX_CLOSE', 'VUSTX_OPEN', 'VUSTX_HIGH', 'VUSTX_LOW',
                 'TNX_CLOSE', 'TNX_OPEN', 'TNX_HIGH', 'TNX_LOW',
                 'HG=F_CLOSE', 'HG=F_OPEN', 'HG=F_HIGH', 'HG=F_LOW']
                 
    features = events_df.drop(columns=[c for c in drop_cols if c in events_df.columns])
    features = features.drop(columns=['SPY_FWD_5D'])
    features.dropna(axis=1, inplace=True)
    
    # Target Vector: 1 = Winner (Profit > 0), 0 = Loser (Profit <= 0)
    y = (events_df['SPY_FWD_5D'] > 0).astype(int)
    X = features
    
    print(f"[Matrix] Analyzing 100 Events across {len(X.columns)} Secondary DB Indicators...")
    
    # Statistical Divergence Matrix: Averages of Winners vs Losers
    print("\n[Matrix] Top 5 Mathematical Divergences (Winners vs Losers):")
    winners = events_df[events_df['SPY_FWD_5D'] > 0]
    losers = events_df[events_df['SPY_FWD_5D'] <= 0]
    
    diffs = []
    for col in features.columns:
        w_mean = winners[col].mean()
        l_mean = losers[col].mean()
        # Normalized difference (Z-Score)
        std = features[col].std()
        if std > 0:
            z_diff = abs(w_mean - l_mean) / std
            diffs.append((col, w_mean, l_mean, z_diff))
            
    diffs.sort(key=lambda x: x[3], reverse=True)
    
    for col, w, l, z in diffs[:5]:
        print(f" -> {col:<25} | Winners Avg: {w:>8.2f} | Losers Avg: {l:>8.2f} | Delta: {z:.2f}σ")


    # Decision Tree ML Extraction
    print("\n=========================================================")
    print(" [Matrix] Decision Tree Logic (Extracting the Loser Filter)")
    print("=========================================================")
    
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, random_state=42)
    clf.fit(X, y)
    
    tree_rules = export_text(clf, feature_names=list(X.columns))
    print(tree_rules)
    print("   [Key] Class 1 = Winner (Profit). Class 0 = Loser (Loss).")

if __name__ == "__main__":
    optimize_win_rate()

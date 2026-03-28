import os
import sys
import pandas as pd
import numpy as np
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hardcoded_wrapper import attach_features, MetaStrategyClassifier
from execution.openalice import calc_kelly

def main():
    print("[Regime Backtest] Loading physical market_data.db local SQLite volume...")
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'market_data.db')
    
    if not os.path.exists(db_path):
        print("[Fatal] Could not locate market_data.db")
        return

    conn = sqlite3.connect(db_path)
    
    try:
        df = pd.read_sql(f"SELECT * FROM core_market_table", conn, parse_dates=['Date'], index_col='Date')
    except Exception as e:
        print(f"[Fatal SQL Error] {e}")
        return
        
    df.index = pd.to_datetime(df.index)
    
    # Slice exactly to 2015-01-01 -> 2025-01-01
    df = df[(df.index >= '2015-01-01') & (df.index <= '2025-01-01')]
    
    # To run legacy tools from hardcoded wrapper, we need 'VIX', 'SPY', 'COPPER', 'GOLD'
    # Ensure backwards compatibility if the db columns are mapped with _CLOSE
    if 'SPY_CLOSE' in df.columns and 'SPY' not in df.columns:
        df['SPY'] = df['SPY_CLOSE']
    if 'VIX_CLOSE' in df.columns and 'VIX' not in df.columns:
        df['VIX'] = df['VIX_CLOSE']
    if 'GC_CLOSE' in df.columns and 'GOLD' not in df.columns:
        df['GOLD'] = df['GC_CLOSE']
    if 'HG_CLOSE' in df.columns and 'COPPER' not in df.columns:
        df['COPPER'] = df['HG_CLOSE']
        
    print(f"[Regime Backtest] Loaded {len(df)} operational business days. Attaching native algorithmic geometries...")
    df = attach_features(df)
    
    # Must wait for 200 days to buffer the SMAs before executing
    df = df.iloc[200:]
    
    print(f"[Regime Backtest] Initializing MetaStrategyClassifier (Encyclopedia + Regime Routing)...")
    clf = MetaStrategyClassifier()
    
    capital = 10000.0
    strategy_returns = []
    
    trading_days = df.index.tolist()
    previous_size = 0.0
    
    for i in range(len(trading_days) - 1):
        row = df.iloc[i]
        
        current_vix = row['VIX']
        
        # Meta Classifier determines Regime (Panic/Expansion/Chop) and routes to specific isolated logic automatically
        prob_up = clf.evaluate(row)
        size = calc_kelly(prob_up, current_vix)
        
        transaction_cost = capital * abs(size - previous_size) * 0.001
        margin_borrowed = capital * max(0.0, size - 1.0)
        margin_interest = margin_borrowed * (0.05 / 252.0)
        previous_size = size
        
        price_start = row['SPY']
        price_end = df.iloc[i+1]['SPY']
        forward_return = (price_end - price_start) / price_start
        
        trade_pnl = capital * size * forward_return
        net_pnl = trade_pnl - transaction_cost - margin_interest
        
        strategy_returns.append(net_pnl / capital)
        capital += net_pnl
        
    strat_arr = np.array(strategy_returns)
    strat_sharpe = (strat_arr.mean() / strat_arr.std()) * np.sqrt(252) if strat_arr.std() > 0 else 0
    strat_total_return = ((capital / 10000.0) - 1.0) * 100
    
    # SPY Baseline
    spy_start = df['SPY'].iloc[0]
    spy_end = df['SPY'].iloc[-1]
    
    spy_daily_returns = df['SPY'].pct_change().dropna().values
    spy_sharpe = (spy_daily_returns.mean() / spy_daily_returns.std()) * np.sqrt(252)
    spy_total_return = ((spy_end / spy_start) - 1.0) * 100
    
    print("\n================= 10-YEAR ENCYCLOPEDIA REGIME VALIDATION =================")
    print(f"S&P 500 (Buy & Hold) | Total Return: {spy_total_return:>8,.2f}% | Sharpe Ratio: {spy_sharpe:.2f}")
    print(f"Meta-Classifier      | Total Return: {strat_total_return:>8,.2f}% | Sharpe Ratio: {strat_sharpe:.2f}")
    print("==========================================================================\n")

if __name__ == "__main__":
    main()

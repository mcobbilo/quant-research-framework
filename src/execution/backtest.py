import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hardcoded_wrapper import attach_features, StrategyA, StrategyB, StrategyC, StrategyD
from execution.openalice import calc_kelly

def run_strategy(model, df):
    capital = 10000.0
    strategy_returns = []
    
    trading_days = df.index.tolist()
    previous_size = 0.0
    
    # We step DAILY evaluating deterministic trigger points
    for i in range(200, len(trading_days) - 1, 1):
        row = df.iloc[i]
        
        current_vix = row['VIX'].item() if isinstance(row['VIX'], pd.Series) else row['VIX']
        
        prob_up = model.evaluate(row)
        size = calc_kelly(prob_up, current_vix)
        
        transaction_cost = capital * abs(size - previous_size) * 0.001
        margin_borrowed = capital * max(0.0, size - 1.0)
        margin_interest = margin_borrowed * (0.05 / 252.0)
        previous_size = size
        
        price_start = df['SPY'].iloc[i].item() if isinstance(df['SPY'].iloc[i], pd.Series) else df['SPY'].iloc[i]
        price_end = df['SPY'].iloc[i+1].item() if isinstance(df['SPY'].iloc[i+1], pd.Series) else df['SPY'].iloc[i+1]
        forward_return = (price_end - price_start) / price_start
        
        trade_pnl = capital * size * forward_return
        net_pnl = trade_pnl - transaction_cost - margin_interest
        
        strategy_returns.append(net_pnl / capital)
        capital += net_pnl
        
    strat_arr = np.array(strategy_returns)
    strat_sharpe = (strat_arr.mean() / strat_arr.std()) * np.sqrt(252) if strat_arr.std() > 0 else 0
    strat_total_return = ((capital / 10000.0) - 1.0) * 100
    
    return strat_total_return, strat_sharpe

def main():
    print("[Backtest] Initiating Event-Driven Daily Backtest (2000-2025)...")
    
    spy_data = yf.download("SPY", start="2000-01-01", end="2025-01-01", progress=False, auto_adjust=True)
    vix_data = yf.download("^VIX", start="2000-01-01", end="2025-01-01", progress=False)
    
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy = spy_data[('Close', 'SPY')]
        vix = vix_data[('Close', '^VIX')]
    else:
        spy = spy_data['Close']
        vix = vix_data['Close']
    
    df = pd.DataFrame(index=spy.index)
    df["SPY"] = spy
    df["VIX"] = vix
    df = df.dropna()
    print(f"[Backtest] Base data loaded. Calculating technical vectors...")
    
    df = attach_features(df)
    df = df.dropna()
    
    spy_start = df['SPY'].iloc[200].item() if isinstance(df['SPY'].iloc[200], pd.Series) else df['SPY'].iloc[200]
    spy_end = df['SPY'].iloc[-1].item() if isinstance(df['SPY'].iloc[-1], pd.Series) else df['SPY'].iloc[-1]
    
    # Calculate daily returns for S&P 500 Sharpe Ratio matching the simulation bounds
    spy_daily_returns = df['SPY'].iloc[200:].pct_change().dropna().values
    spy_sharpe = (spy_daily_returns.mean() / spy_daily_returns.std()) * np.sqrt(252)
    spy_total_return = ((spy_end / spy_start) - 1.0) * 100

    models = [
        StrategyD(entry_z=-3.5, exit_z=3.0, baseline_prob=0.75)  # 0.75 maps to exactly 1.0x Kelly Sizing
    ]
    
    print("\n================= 25-YEAR SYSTEM YIELD =================")
    print(f"S&P 500 (Buy & Hold) | Total Return: {spy_total_return:>8,.2f}% | Sharpe Ratio: {spy_sharpe:.2f}")
    
    for m in models:
        ret, sharpe = run_strategy(m, df)
        print(f"{m.name:<20} | Total Return: {ret:>8,.2f}% | Sharpe Ratio: {sharpe:.2f}")
    print("========================================================\n")

if __name__ == "__main__":
    main()

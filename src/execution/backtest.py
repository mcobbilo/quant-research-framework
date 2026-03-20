import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.xgboost_wrapper import WalkForwardXGBoost
from execution.openalice import calc_kelly

def run_backtest():
    print("[Backtest] Initiating 25-Year Walk-Forward Framework...")
    
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
    df['Returns'] = df['SPY'].pct_change()
    df = df.dropna()
    
    print(f"[Backtest] Data loaded. {len(df)} daily trading days available.")
    
    model = WalkForwardXGBoost()
    
    capital = 10000.0
    equity_curve = [capital]
    dates = [df.index[0]]
    
    strategy_returns = []
    spy_returns = []
    
    print("[Backtest] Commencing temporal expanding-window simulation...")
    
    trading_days = df.index.tolist()
    
    previous_size = 0.0
    
    for i in range(512, len(trading_days), 21): # Start at 512 so we have full context window!
        current_date = trading_days[i]
        
        history_window = df.iloc[:i]
        current_vix = history_window['VIX'].iloc[-1].item() if isinstance(history_window['VIX'].iloc[-1], pd.Series) else history_window['VIX'].iloc[-1]
        
        prob_up = model.train_and_predict(history_window, current_vix)
        size = calc_kelly(prob_up, current_vix)
        
        transaction_cost = capital * abs(size - previous_size) * 0.001 # 10 bps slippage/commission
        margin_borrowed = capital * max(0.0, size - 1.0)
        margin_interest = margin_borrowed * (0.05 / 12.0) # ~5% annualized rate 
        previous_size = size
        
        if i + 21 < len(trading_days):
            price_start = df['SPY'].iloc[i].item() if isinstance(df['SPY'].iloc[i], pd.Series) else df['SPY'].iloc[i]
            price_end = df['SPY'].iloc[i+21].item() if isinstance(df['SPY'].iloc[i+21], pd.Series) else df['SPY'].iloc[i+21]
            forward_return = (price_end - price_start) / price_start
            
            trade_pnl = capital * size * forward_return
            net_pnl = trade_pnl - transaction_cost - margin_interest
            
            strategy_ret = net_pnl / capital
            strategy_returns.append(strategy_ret)
            spy_returns.append(forward_return)
            
            capital += net_pnl
            
            equity_curve.append(capital)
            dates.append(current_date)

    print(f"\n[Backtest] Walk-Forward Simulation Complete.")
    
    # Calculate metrics
    strat_arr = np.array(strategy_returns)
    spy_arr = np.array(spy_returns)
    
    strat_sharpe = (strat_arr.mean() / strat_arr.std()) * np.sqrt(12) if strat_arr.std() > 0 else 0
    spy_sharpe = (spy_arr.mean() / spy_arr.std()) * np.sqrt(12) if spy_arr.std() > 0 else 0
    
    strat_total_return = ((capital / 10000.0) - 1.0) * 100
    
    spy_start = df['SPY'].iloc[512].item() if isinstance(df['SPY'].iloc[512], pd.Series) else df['SPY'].iloc[512]
    spy_end = df['SPY'].iloc[-1].item() if isinstance(df['SPY'].iloc[-1], pd.Series) else df['SPY'].iloc[-1]
    spy_total_return = ((spy_end / spy_start) - 1.0) * 100

    print(f"\n=== 25-Year Performance Benchmark ===")
    print(f"S&P 500 (Buy & Hold) | Total Return: {spy_total_return:,.2f}% | Sharpe Ratio: {spy_sharpe:.2f}")
    print(f"TimesFM (2.0x Kelly) | Total Return: {strat_total_return:,.2f}% | Sharpe Ratio: {strat_sharpe:.2f}")

if __name__ == "__main__":
    run_backtest()

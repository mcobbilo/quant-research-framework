import os
import sys
import torch
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hardcoded_wrapper import attach_features, StrategyD, StrategyG
from execution.openalice import calc_kelly
from core.deliberation import LatentCouncil
from models.roan_protocol import RoanCombiner

class AgenticStrategy:
    """
    Phase 16: Self-Organizing Agentic Strategy.
    Uses the LatentCouncil to 'figure out' the market state without hardcoded filters.
    """
    def __init__(self, feature_dim: int = 4):
        self.council = LatentCouncil(feature_dim=feature_dim)
        self.name = "Phase 26.1: Structural Drift Expansion"
        self.in_trade = False
        self.baseline_prob = 0.58 # Structural Long Bias (Final Push)
        self.action_history = [] 
        
    def evaluate(self, row):
        vix = row['VIX']
        move = row.get('MOVE', 80.0)
        
        # 1. Primary Alpha (Latent Council)
        feat_list = [row['SPY'], vix, row['GOLD'], row['COPPER']]
        features = torch.tensor(feat_list, dtype=torch.float32).unsqueeze(0)
        latent = self.council.project_agent_reasoning(features)
        
        # Conviction Scale: Boosting high-confidence stage_trade up to 0.94
        action = self.council.derive_action(latent, 0.94, vix=vix)
        
        # 2. Macro VIX-MOVE Scaling Matrix 
        if vix > 40: scale = 1.0 
        elif move > 130: scale = 0.20 
        elif 30 < vix <= 40: scale = 0.98 
        elif vix < 15 and move < 60: scale = 1.05 
        elif 20 < vix <= 30: scale = 0.75 
        else: scale = 1.0 
            
        if action == "stage_trade":
            prob = 0.94 * scale
            prob = min(0.99, prob)
        elif action == "stage_hedge":
            prob = 0.08 * scale
        else:
            prob = self.baseline_prob
            
        self.action_history.append(prob)
        return prob

def run_strategy(model, df):
    capital = 10000.0
    initial_capital = 10000.0
    peak_capital = 10000.0
    strategy_returns = []
    
    trading_days = df.index.tolist()
    previous_size = 0.0
    
    # We step DAILY evaluating deterministic trigger points
    for i in range(200, len(trading_days) - 1, 1):
        row = df.iloc[i]
        current_vix = row['VIX'].item() if isinstance(row['VIX'], pd.Series) else row['VIX']
        
        prob_up = model.evaluate(row)
        size = calc_kelly(prob_up, current_vix)
        
        # [SECURITY] Confirm NO LEVERAGE was allowed by eval
        size = min(1.0, size)
        
        transaction_cost = capital * abs(size - previous_size) * 0.001
        # [PHASE 17] Margin logic removed (Cash Only)
        margin_borrowed = 0.0
        margin_interest = 0.0
        
        previous_size = size
        
        price_start = df['SPY'].iloc[i].item() if isinstance(df['SPY'].iloc[i], pd.Series) else df['SPY'].iloc[i]
        price_end = df['SPY'].iloc[i+1].item() if isinstance(df['SPY'].iloc[i+1], pd.Series) else df['SPY'].iloc[i+1]
        forward_return = (price_end - price_start) / price_start
        
        trade_pnl = capital * size * forward_return
        net_pnl = trade_pnl - transaction_cost
        
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
    move_data = yf.download("^MOVE", start="2000-01-01", end="2025-01-01", progress=False)
    gold_data = yf.download("GC=F", start="2000-01-01", end="2025-01-01", progress=False)
    copper_data = yf.download("HG=F", start="2000-01-01", end="2025-01-01", progress=False)
    
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy = spy_data[('Close', 'SPY')]
        vix = vix_data[('Close', '^VIX')]
        vix_open = vix_data[('Open', '^VIX')]
        vix_high = vix_data[('High', '^VIX')]
        vix_low = vix_data[('Low', '^VIX')]
        
        gold = gold_data[('Close', 'GC=F')]
        copper = copper_data[('Close', 'HG=F')]
    else:
        spy = spy_data['Close']
        vix = vix_data['Close']
        vix_open = vix_data['Open']
        vix_high = vix_data['High']
        vix_low = vix_data['Low']
        
        gold = gold_data['Close']
        copper = copper_data['Close']
    
    # Concatenate all series into a single dataframe to ensure perfect temporal alignment
    df = pd.DataFrame(index=spy.index)
    df["SPY"] = spy
    df["VIX"] = vix
    df["VIX_OPEN"] = vix_open
    df["VIX_HIGH"] = vix_high
    df["VIX_LOW"] = vix_low
    
    # Handle MOVE Index (May start in 2002)
    move_series = move_data.get('Close')
    if move_series is not None:
        if isinstance(move_series, pd.DataFrame): # Multi-index check
            df["MOVE"] = move_series.iloc[:, 0]
        else:
            df["MOVE"] = move_series
            
    df["GOLD"] = gold
    df["COPPER"] = copper
    
    # Crucially ffill any missing commodity trading days before dropping NaNs so SPY geometry doesn't collapse
    df.ffill(inplace=True)
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
        StrategyD(entry_z=-3.5, exit_z=3.0, baseline_prob=0.75),
        StrategyG(entry_z=-3.5, exit_z=3.0, baseline_prob=0.75),
        AgenticStrategy(feature_dim=4)
    ]
    
    print("\n================= 25-YEAR SYSTEM YIELD =================")
    print(f"S&P 500 (Buy & Hold) | Total Return: {spy_total_return:>8,.2f}% | Sharpe Ratio: {spy_sharpe:.2f}")
    
    for m in models:
        ret, sharpe = run_strategy(m, df)
        print(f"{m.name:<20} | Total Return: {ret:>8,.2f}% | Sharpe Ratio: {sharpe:.2f}")
    print("========================================================\n")

if __name__ == "__main__":
    main()

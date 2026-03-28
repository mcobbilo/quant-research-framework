import os
import sys
import pandas as pd
import numpy as np
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hardcoded_wrapper import attach_features, MetaStrategyClassifier
from execution.openalice import calc_kelly

def main():
    print("[Full Backtest] Loading physical market_data.db local SQLite volume...")
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'market_data.db')
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM core_market_table", conn, parse_dates=['Date'], index_col='Date')
    df.index = pd.to_datetime(df.index)
    
    if 'SPY_CLOSE' in df.columns and 'SPY' not in df.columns:
        df['SPY'] = df['SPY_CLOSE']
    if 'VIX_CLOSE' in df.columns and 'VIX' not in df.columns:
        df['VIX'] = df['VIX_CLOSE']
    if 'GC_CLOSE' in df.columns and 'GOLD' not in df.columns:
        df['GOLD'] = df['GC_CLOSE']
    if 'HG_CLOSE' in df.columns and 'COPPER' not in df.columns:
        df['COPPER'] = df['HG_CLOSE']
        
    print(f"[Full Backtest] Loaded {len(df)} operational business days. Attaching native algorithmic geometries...")
    df = attach_features(df)
    
    # Must wait for 200 days to buffer the SMAs before executing
    df = df.iloc[200:]
    
    print(f"[Full Backtest] Initializing MetaStrategyClassifier (1.0x Hard Capped)...")
    clf = MetaStrategyClassifier()
    
    capital = 10000.0
    spy_capital = 10000.0
    
    trading_days = df.index.tolist()
    previous_size = 0.0
    
    portfolio_values = []
    spy_values = []
    dates = []
    
    for i in range(len(trading_days) - 1):
        row = df.iloc[i]
        date = trading_days[i+1]
        
        current_vix = row['VIX']
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
        net_pnL = trade_pnl - transaction_cost - margin_interest
        capital += net_pnL
        
        spy_capital += (spy_capital * forward_return)
        
        portfolio_values.append(capital)
        spy_values.append(spy_capital)
        dates.append(date)
        
    res = pd.DataFrame({'Strategy': portfolio_values, 'SPY': spy_values}, index=dates)
    
    strat_total_return = ((capital / 10000.0) - 1.0) * 100
    spy_total_return = ((spy_capital / 10000.0) - 1.0) * 100
    
    res['Strat_Peak'] = res['Strategy'].cummax()
    res['Strat_DD'] = (res['Strategy'] - res['Strat_Peak']) / res['Strat_Peak']
    worst_dd_strat = res['Strat_DD'].min() * 100
    
    res['SPY_Peak'] = res['SPY'].cummax()
    res['SPY_DD'] = (res['SPY'] - res['SPY_Peak']) / res['SPY_Peak']
    worst_dd_spy = res['SPY_DD'].min() * 100
    
    print("\n================= 26-YEAR FULL TIMELINE BENCHMARK =================")
    print(f"S&P 500 (Buy & Hold) | Total Return: {spy_total_return:>8,.2f}% | Worst Drawdown: {worst_dd_spy:.2f}%")
    print(f"Meta-Classifier      | Total Return: {strat_total_return:>8,.2f}% | Worst Drawdown: {worst_dd_strat:.2f}%")
    print("===================================================================\n")
    
    print("YEAR-BY-YEAR ANALYSIS:")
    print(f"{'Year':<6} | {'Strategy%':<12} | {'SPY%':<12} | {'Alpha%':<12}")
    print("-" * 50)
    
    res['Year'] = res.index.year
    years = res['Year'].unique()
    
    md_output = "# 26-Year Execution Tearsheet (MetaStrategy)\n\n"
    md_output += "## 1. Absolute Performance Metrics\n"
    md_output += f"- **S&P 500 (Buy & Hold)**: Total Return `{spy_total_return:,.2f}%` | Maximum Drawdown `{worst_dd_spy:.2f}%`\n"
    md_output += f"- **Meta-Classifier (1.0x)**: Total Return `{strat_total_return:,.2f}%` | Maximum Drawdown `{worst_dd_strat:.2f}%`\n\n"
    
    md_output += "## 2. Year-By-Year Alpha Generation\n"
    md_output += "| Year | Strategy Yield | S&P 500 Yield | Annual Alpha |\n"
    md_output += "|---|---|---|---|\n"
    
    for y in years:
        yr_data = res[res['Year'] == y]
        if len(yr_data) < 2: continue
        
        start_val_strat = yr_data['Strategy'].iloc[0]
        end_val_strat = yr_data['Strategy'].iloc[-1]
        yr_ret_strat = ((end_val_strat / start_val_strat) - 1) * 100
        
        start_val_spy = yr_data['SPY'].iloc[0]
        end_val_spy = yr_data['SPY'].iloc[-1]
        yr_ret_spy = ((end_val_spy / start_val_spy) - 1) * 100
        
        alpha = yr_ret_strat - yr_ret_spy
        
        print(f"{y:<6} | {yr_ret_strat:>11.2f}% | {yr_ret_spy:>11.2f}% | {alpha:>11.2f}%")
        md_output += f"| **{y}** | `{yr_ret_strat:.2f}%` | `{yr_ret_spy:.2f}%` | **`{alpha:+.2f}%`** |\n"
        
    print("-" * 50)
    
    with open('/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/execution_tearsheet.md', 'w') as f:
        f.write(md_output)

if __name__ == "__main__":
    main()

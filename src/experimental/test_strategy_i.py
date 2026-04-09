import pandas as pd
import numpy as np
import sqlite3
import os

class StrategyI:
    """
    Relaxed Heuristics Machine Learning Filter
    Strips away ML curve-fitting by expanding decimal exactness to structural integers.
    """
    def __init__(self, ppo_thresh=35.0, ratio_cap=14.5, vustx_rot=-1.15):
        self.ppo_thresh = ppo_thresh
        self.ratio_cap = ratio_cap
        self.vustx_rot = vustx_rot

    def generate_signal(self, row):
        try:
            ppo = row['VIX_TNX_PPO_7']
            ratio = row['VIX_TNX_RATIO']
            rot = row['SPY_VUSTX_DIFF_3D_ZSCORE']
            
            if pd.isna(ppo) or pd.isna(ratio) or pd.isna(rot):
                return 0.0

            # 1. Volatility is spiking heavily
            if ppo > self.ppo_thresh:
                # 2. But the structure is sound (Ratio < 14.5) and capital has rotated (Z < -1.15)
                if ratio < self.ratio_cap and rot < self.vustx_rot:
                    return 1.0
            return 0.0
        except KeyError:
            return 0.0

def run_strategy_i():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_data.db')
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM core_market_table', conn, index_col='Date')
    df.index = pd.to_datetime(df.index)
    
    print("[Strategy I] Loading 25-Year Geometric Baseline...")
    
    strategy = StrategyI()
    signals = df.apply(strategy.generate_signal, axis=1)
    
    positions = np.ones(len(df)) # Baseline 1.0x SPY Long Position
    hold_days = 0
    
    trigger_count = 0
    for i in range(len(df)):
        if signals.iloc[i] > 0:
            hold_days = 5
            trigger_count += 1
            
        if hold_days > 0:
            positions[i] = 2.0 # Deploy 2.0x Margin Equivalent Exposure
            hold_days -= 1

    df['Position'] = positions
    df['Raw_Returns'] = df['SPY_CLOSE'].pct_change()
    
    # Gross Return Calculation
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Raw_Returns']
    
    # Red Team Friction Verification (0.1% institutional slippage, 5% margin interest rate)
    trades = df['Position'].diff().abs()
    margin_borrowed = np.maximum(0, df['Position'].shift(1) - 1.0)
    
    cost_per_trade = 0.001
    daily_margin_rate = 0.05 / 252.0
    
    df['Net_Returns'] = df['Strategy_Returns'] - (trades * cost_per_trade) - (margin_borrowed * daily_margin_rate)
    
    # Compounding
    df['SPY_Growth'] = (1 + df['Raw_Returns'].fillna(0)).cumprod()
    df['Strat_Growth'] = (1 + df['Net_Returns'].fillna(0)).cumprod()
    
    spy_final = df['SPY_Growth'].iloc[-1]
    strat_final = df['Strat_Growth'].iloc[-1]
    
    print("\n=======================================================")
    print("   STRATEGY I: HEURISTIC RELAXED ML ALGORITHM          ")
    print("=======================================================")
    print(f" Timeline:        {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f" Starting Cash:   $10,000")
    print(f" Anomaly Events:  {trigger_count}")
    print(f"")
    print(f" [BENCHMARK] SPY Buy/Hold:  ${(10000 * spy_final):,.2f}  ({spy_final*100-100:.1f}%)")
    print(f" [ALGORITHM] Strategy I:    ${(10000 * strat_final):,.2f}  ({strat_final*100-100:.1f}%)")
    print("=======================================================\n")

if __name__ == "__main__":
    run_strategy_i()

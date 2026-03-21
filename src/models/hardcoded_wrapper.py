import numpy as np
import pandas as pd

def attach_features(df):
    """
    Vectorized calculation of all required technical indicators 
    across the entire historical dataframe to radically speed up evaluation.
    """
    df['SMA_200'] = df['SPY'].rolling(200).mean()
    
    # RSI 5
    delta = df['SPY'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
    rs = gain / loss
    df['RSI_5'] = 100 - (100 / (1 + rs))
    
    # Strategy C: Bollinger Bands
    df['SMA_20'] = df['SPY'].rolling(20).mean()
    df['STD_20'] = df['SPY'].rolling(20).std()
    df['Lower_BB'] = df['SMA_20'] - (2 * df['STD_20'])
    
    # Strategy B: Volatility Exhaustion
    df['VIX_prev'] = df['VIX'].shift(1)
    df['SPY_ret_prev_1'] = df['SPY'].pct_change().shift(1)
    df['SPY_ret_prev_2'] = df['SPY'].pct_change().shift(2)
    df['SPY_ret_prev_3'] = df['SPY'].pct_change().shift(3)
    
    # Strategy D: Fourier Aftershock
    df['Z_Score'] = (df['SPY'] - df['SMA_20']) / df['STD_20']
    
    return df

class StrategyA:
    def __init__(self):
        self.name = "Strategy A: Bull Market Pullback"
        
    def evaluate(self, row):
        # High VIX, Bull Market Trend, Severe short-term oversold condition
        c1 = row['VIX'] > 25
        c2 = row['SPY'] > row['SMA_200']
        c3 = row['RSI_5'] < 30
        return 1.0 if (c1 and c2 and c3) else 0.50

class StrategyB:
    def __init__(self):
        self.name = "Strategy B: Volatility Exhaustion"
        
    def evaluate(self, row):
        # Massive panic, but falling VIX + 3 days of localized price dumping
        c1 = row['VIX'] > 35
        c2 = row['VIX'] < row['VIX_prev']
        c3 = (row['SPY_ret_prev_1'] < 0) and (row['SPY_ret_prev_2'] < 0) and (row['SPY_ret_prev_3'] < 0)
        return 1.0 if (c1 and c2 and c3) else 0.50

class StrategyC:
    def __init__(self):
        self.name = "Strategy C: Structurally Broken Vol Expansion"
        
    def evaluate(self, row):
        # Fear + Severe deviation below 2 standard deviations
        c1 = row['VIX'] > 20
        c2 = row['SPY'] < row['Lower_BB']
        return 1.0 if (c1 and c2) else 0.50

class StrategyD:
    def __init__(self, entry_z=-3.0, exit_z=2.5, baseline_prob=0.75):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.baseline_prob = baseline_prob
        self.name = f"Phase 9: SPY Baseline + {abs(entry_z)} Sigma Aftershock"
        self.in_trade = False
        
    def evaluate(self, row):
        z = row['Z_Score']
        
        # If we are currently in our resting baseline (1.0x SPY)
        if not self.in_trade:
            if z < self.entry_z: 
                self.in_trade = True
                return 1.0 # Max leverage (2.0x)
            return self.baseline_prob
        
        # If we are currently holding the 2.0x leveraged position
        else:
            if z > self.exit_z: 
                self.in_trade = False
                return self.baseline_prob
            return 1.0 # Continue holding through the chop

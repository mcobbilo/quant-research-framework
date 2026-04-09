import os
import sys
import torch
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.backtest import AgenticStrategy, run_strategy, attach_features
from src.execution.openalice import calc_kelly

class RedTeamAgenticStrategy(AgenticStrategy):
    """
    Wraps AgenticStrategy to allow adversarial overrides during evaluation.
    """
    def __init__(self, noise_level=0.0, consensus_override=None):
        super().__init__()
        self.noise_level = noise_level
        self.consensus_override = consensus_override

    def evaluate(self, row):
        # Inject noise if requested
        if self.noise_level > 0:
            row = row.copy()
            for col in ['SPY', 'VIX', 'GOLD', 'COPPER']:
                row[col] *= (1 + np.random.normal(0, self.noise_level))

        # Re-run evaluate logic with noise
        recursive_context = (self.action_history[-1] if self.action_history else 0.5)
        feat_list = [row['SPY'], row['VIX'], row['GOLD'], row['COPPER']]
        features = torch.tensor(feat_list, dtype=torch.float32).unsqueeze(0)
        
        latent = self.council.project_agent_reasoning(features)
        
        # Scenario: Consensus Poisoning
        consensus_score = self.consensus_override if self.consensus_override is not None else 0.92
        
        action = self.council.derive_action(latent, consensus_score)
        vix = row['VIX']
        kelly_multiplier = 2.0 
        scale = min(1.4, max(0.6, (vix / 19.0) ** 1.2))
        
        if action == "stage_trade":
            prob = 0.92 * scale
            prob = min(0.99, prob * kelly_multiplier)
        elif action == "stage_hedge":
            prob = 0.08 / scale
        else:
            prob = self.baseline_prob
            
        self.action_history.append(prob)
        return prob

def run_red_team_backtest(model, df, transaction_cost_rate=0.001):
    capital = 10000.0
    strategy_returns = []
    previous_size = 0.0
    
    for i in range(200, len(df) - 1):
        row = df.iloc[i]
        current_vix = row['VIX']
        
        prob_up = model.evaluate(row)
        size = calc_kelly(prob_up, current_vix)
        
        # Custom transaction cost for Red Teaming
        transaction_cost = capital * abs(size - previous_size) * transaction_cost_rate
        margin_borrowed = capital * max(0.0, size - 1.0)
        margin_interest = margin_borrowed * (0.05 / 252.0)
        previous_size = size
        
        price_start = df['SPY'].iloc[i]
        price_end = df['SPY'].iloc[i+1]
        forward_return = (price_end - price_start) / price_start
        
        trade_pnl = capital * size * forward_return
        net_pnl = trade_pnl - transaction_cost - margin_interest
        
        strategy_returns.append(net_pnl / capital)
        capital += net_pnl
        
    strat_arr = np.array(strategy_returns)
    sharpe = (strat_arr.mean() / strat_arr.std()) * np.sqrt(252) if strat_arr.std() > 0 else 0
    total_return = ((capital / 10000.0) - 1.0) * 100
    mdd = 0
    if len(strategy_returns) > 0:
        cum_returns = np.cumprod(1 + strat_arr)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        mdd = np.min(drawdown) * 100

    return total_return, sharpe, mdd

def main():
    print("--- [RED TEAM] Initializing Stress Test Harness ---")
    
    # 1. Load Data
    spy_data = yf.download("SPY", start="2000-01-01", end="2025-01-01", progress=False, auto_adjust=True)
    vix_data = yf.download("^VIX", start="2000-01-01", end="2025-01-01", progress=False)
    gold_data = yf.download("GC=F", start="2000-01-01", end="2025-01-01", progress=False)
    copper_data = yf.download("HG=F", start="2000-01-01", end="2025-01-01", progress=False)
    
    df = pd.DataFrame(index=spy_data.index)
    df["SPY"] = spy_data['Close']
    df["VIX"] = vix_data['Close']
    df["GOLD"] = gold_data['Close']
    df["COPPER"] = copper_data['Close']
    df.ffill(inplace=True)
    df = df.dropna()
    df = attach_features(df)
    df = df.dropna()

    results = []

    # SCENARIO 0: Baseline (No Noise, 10bps)
    print("\n[Scenario 0] Establishing baseline...")
    m0 = RedTeamAgenticStrategy(noise_level=0.0)
    ret, shp, mdd = run_red_team_backtest(m0, df, transaction_cost_rate=0.001)
    results.append({"Case": "Baseline", "Return": ret, "Sharpe": shp, "MDD": mdd})

    # SCENARIO 1: Feature Noise (5%)
    print("[Scenario 1] Injecting 5% Feature Noise...")
    m1 = RedTeamAgenticStrategy(noise_level=0.05)
    ret, shp, mdd = run_red_team_backtest(m1, df, transaction_cost_rate=0.001)
    results.append({"Case": "5% Feature Noise", "Return": ret, "Sharpe": shp, "MDD": mdd})

    # SCENARIO 2: Regime "Hammer" (Fake VIX spike to 90 during bull run)
    print("[Scenario 2] Simulation of Extreme VIX Hammer (Regime Stress)...")
    df_hammer = df.copy()
    # Force VIX to 90 during 2021 bull run
    df_hammer.loc['2021-01-01':'2021-12-31', 'VIX'] = 90.0
    m2 = RedTeamAgenticStrategy(noise_level=0.0)
    ret, shp, mdd = run_red_team_backtest(m2, df_hammer, transaction_cost_rate=0.001)
    results.append({"Case": "VIX Hammer (2021 Spike)", "Return": ret, "Sharpe": shp, "MDD": mdd})

    # SCENARIO 3: Slippage Sensitivity (30bps fee)
    print("[Scenario 3] Stressing Slippage (30bps)...")
    m3 = RedTeamAgenticStrategy(noise_level=0.0)
    ret, shp, mdd = run_red_team_backtest(m3, df, transaction_cost_rate=0.003)
    results.append({"Case": "30bps Slippage", "Return": ret, "Sharpe": shp, "MDD": mdd})

    # SCENARIO 4: Consensus Poisoning (Agent Disagreement)
    print("[Scenario 4] Simulating Council Disagreement (Consensus 0.5)...")
    m4 = RedTeamAgenticStrategy(noise_level=0.0, consensus_override=0.50)
    ret, shp, mdd = run_red_team_backtest(m4, df, transaction_cost_rate=0.001)
    results.append({"Case": "Low Consensus (0.50)", "Return": ret, "Sharpe": shp, "MDD": mdd})

    # Result Summary Table
    res_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("RED TEAM AUDIT SUMMARY")
    print("="*60)
    print(res_df.to_string(index=False))
    print("="*60 + "\n")

    # Final Red Team Vibe Check
    baseline_ret = results[0]['Return']
    noise_ret = results[1]['Return']
    if noise_ret < baseline_ret * 0.3:
        print("[RED TEAM] WARNING: High Fragility detected. Noise collapsed alpha by >70%.")
    else:
        print("[RED TEAM] PASS: Feature robustness confirmed.")

    if results[4]['Return'] > 0:
        print("[RED TEAM] WARNING: Failure to Fallback. Model trading through Low Consensus.")
    else:
        print("[RED TEAM] PASS: Consensus Guard triggered successfully.")

if __name__ == "__main__":
    main()

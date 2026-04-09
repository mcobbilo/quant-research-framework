import os
import sqlite3
import pandas as pd
import numpy as np
import subprocess
import sys
import shutil

class StressTestEngine:
    """
    Adversarial Regime Generator for J-EPA Strategies.
    Generates synthetic 'Panic Mode' data to evaluate downside resilience.
    """
    def __init__(self, db_path="src/data/stress_test.db"):
        self.db_path = db_path
        self.assets = ['SPY', 'IWM', 'RSP', 'GLD', 'CL', 'HG', 'VUSTX']
        self.cols = ['DATE', 'target_returns'] + [a + '_CLOSE' for a in self.assets]
        
    def generate_panic_regime(self, n_days=250):
        """
        Creates a tripartite synthetic regime: Normal -> Panic -> Recovery.
        """
        print(f"[STRESS] Generating {n_days}-day Adversarial Regime (Crash Scenario)...")
        
        # 1. Normal Regime (0-100)
        n_normal = 100
        normal_ret = np.random.normal(0.0005, 0.01, (n_normal, len(self.assets)))
        
        # 2. Panic Regime (100-150) -> Deep Drawdown + High Vol
        n_panic = 50
        panic_ret = np.random.normal(-0.025, 0.05, (n_panic, len(self.assets)))
        
        # 3. Recovery Regime (150-250)
        n_recovery = 100
        recovery_ret = np.random.normal(0.001, 0.03, (n_recovery, len(self.assets)))
        
        all_rets = np.vstack([normal_ret, panic_ret, recovery_ret])
        
        # Cumulative Prices
        prices = 100.0 * np.exp(np.cumsum(all_rets, axis=0))
        
        df = pd.DataFrame(prices, columns=[a + '_CLOSE' for a in self.assets])
        df['DATE'] = pd.date_range('2026-01-01', periods=n_days).astype(str)
        df['target_returns'] = df['SPY_CLOSE'].pct_change().shift(-1).fillna(0.0)
        
        # Save to Stress DB
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        df.to_sql('core_market_table', conn, if_exists='replace', index=False)
        conn.close()
        return df

    def run_stress_test(self, strategy_code):
        """
        Executes the strategy against the Stress DB.
        """
        print("[STRESS] Injecting Adversarial Regime into Research Loop...")
        
        # Backup real DB and swap
        real_db = "src/data/market_data.db"
        real_db_bak = "src/data/market_data.db.stress_bak"
        
        try:
            if os.path.exists(real_db):
                shutil.copy(real_db, real_db_bak)
            shutil.copy(self.db_path, real_db)
            
            # Execute strategy using the same temp hypothesis file
            with open("temp_stress_test.py", "w") as f:
                f.write(strategy_code)
                
            result = subprocess.run([sys.executable, "temp_stress_test.py"], capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                return {"status": "failure", "reason": f"Crash in Panic Regime: {result.stderr}"}
                
            # Parse metrics
            yield_val, sharpe_val = 0.0, 0.0
            for line in result.stdout.split('\n'):
                if "RESULT_YIELD:" in line: yield_val = float(line.split(":")[1].strip())
                if "RESULT_SHARPE:" in line: sharpe_val = float(line.split(":")[1].strip())
                
            return {
                "status": "success",
                "panic_yield": yield_val,
                "panic_sharpe": sharpe_val,
                "fragility": "STABLE" if yield_val > -10.0 else "FRAGILE"
            }
            
        finally:
            # Restore real DB
            if os.path.exists(real_db_bak):
                shutil.move(real_db_bak, real_db)

if __name__ == "__main__":
    # Test script standalone logic if needed
    pass

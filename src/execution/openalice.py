import json
import hashlib
from datetime import datetime
import math

def calc_kelly(probability, current_vix=None):
    """
    Leveraged Kelly Criterion calculation bounded by rigorous Security Circuit Breakers.
    """
    # [CIRCUIT BREAKER 1] NaN Corruption Check
    if probability is None or math.isnan(probability):
        print("[OpenAlice Guard] FATAL: Probability input is NaN. Forcing position size to 0.0")
        return 0.0
        
    # [CIRCUIT BREAKER 2] Volatility Sanity Bounds
    if current_vix is not None:
        if current_vix < 5.0 or current_vix > 100.0:
            print(f"[OpenAlice Guard] FATAL: VIX ({current_vix}) outside structural bounds (5-100). Forcing size to 0.0")
            return 0.0
            
    edge = probability - 0.5
    if edge <= 0:
        return 0.0
        
    size = max(0.01, edge * 4.0)  # Max Size 2.0x (100% margin utilization) Note: Requires active VIX hedging!
    
    # [CIRCUIT BREAKER 3] Hard Capped Leverage Limit
    return min(size, 2.0)

class OpenAliceUTA:
    def __init__(self, account_id="alpha_fund"):
        self.account_id = account_id
        self.staged_order = None
        self.committed_order = None
        print(f"[OpenAlice] Connected to Unified Trading Account: {self.account_id}")

    def stage(self, asset, action, size):
        self.staged_order = {
            "asset": asset,
            "action": action,
            "size": round(size, 4),
            "timestamp": datetime.utcnow().isoformat()
        }
        print(f"[OpenAlice] Staged {action} order for {self.staged_order['size']} units of {asset}.")

    def commit(self, rationale_hash):
        if not self.staged_order:
            raise ValueError("No order staged.")
        
        self.committed_order = {
            "order": self.staged_order,
            "rationale_hash": rationale_hash,
            "commit_id": hashlib.sha256(str(self.staged_order).encode()).hexdigest()
        }
        print(f"[OpenAlice] Committed order with cryptographic rationale: {rationale_hash}")

    def push(self):
        if not self.committed_order:
            raise ValueError("No order committed.")
        
        # Guard Pipelines
        if self.committed_order['order']['size'] > 0.5:
             print("[OpenAlice] GUARD BLOCK: Order size exceeds maximum position limit.")
             return False
             
        print(f"[OpenAlice] PUSH EXECUTED: Successfully simulated broker execution for commit {self.committed_order['commit_id'][:8]}.")
        self.staged_order = None
        self.committed_order = None
        return True

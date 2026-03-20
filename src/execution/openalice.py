import json
import hashlib
from datetime import datetime

def calc_kelly(probability):
    """
    Mock Kelly Criterion calculation.
    """
    edge = probability - 0.5
    size = max(0.01, edge * 2) 
    return size

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

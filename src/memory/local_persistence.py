import json
import os
from datetime import datetime

class LocalMemoryStore:
    def __init__(self, db_path="memory_store.json"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w") as f:
                json.dump({"factual": [], "experiential": [], "working": []}, f)

    def save_experiential_memory(self, run_id, model_params, sharpe_ratio, rationale):
        """Saves backtest results and model iterations for agent persistence."""
        with open(self.db_path, "r+") as f:
            data = json.load(f)
            data["experiential"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "run_id": run_id,
                "model_params": model_params,
                "performance_metric": sharpe_ratio,
                "rationale": rationale
            })
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        print(f"[Memory] Saved experiential learning for run: {run_id}")

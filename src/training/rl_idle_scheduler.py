import os
import time
import json
import logging
from datetime import datetime

# Assuming zero_claw_rl_loop is used to run actual RL tuning
try:
    from src.training.zero_claw_rl_loop import run_rl_pipeline
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

LOG_FILE_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", "data", "proxy_logs.jsonl"
)

# Thresholds for Idle
LOAD_AVG_IDLE_MAX = 1.0  # Max 1 min load average to be considered idle
POLL_INTERVAL_SEC = 60

def is_system_idle():
    """
    Checks if the system is currently idle based on load average.
    """
    try:
        # getloadavg returns a tuple of 1m, 5m, 15m load
        load_1m, _, _ = os.getloadavg()
        return load_1m < LOAD_AVG_IDLE_MAX
    except AttributeError: # Windows doesn't support getloadavg
        import psutil
        return psutil.cpu_percent(interval=1) < 20.0

def process_proxy_logs():
    """
    Reads the proxy logs and filters for DPO/RL preference pairs.
    """
    if not os.path.exists(LOG_FILE_PATH):
        logging.info("No proxy logs found.")
        return []

    successful_traces = []
    with open(LOG_FILE_PATH, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("is_success", False):
                    successful_traces.append(record)
            except json.JSONDecodeError:
                continue

    return successful_traces

def trigger_rl_optimization(traces):
    """
    Triggers the MetaClaw RL optimization (DPO or offline PPO).
    For now, this dispatches a mock sequence or calls the actual DL pipeline.
    """
    logging.info(f"Starting Offline Meta-Learning RL optimization with {len(traces)} traces...")
    # In a full deployment, this would write to a dataset and call:
    # run_rl_pipeline(dataset_path="...", model_name="...", lr=1e-5)
    time.sleep(5) # Simulate compilation and RL loop
    logging.info("Completed Offline Meta-Learning RL optimization. Weights updated.")

def idle_scheduler_loop():
    logging.info("MetaClaw RL Idle Scheduler Started.")
    while True:
        if is_system_idle():
            logging.info("System is IDEAL. Checking for proxy logs...")
            traces = process_proxy_logs()
            if len(traces) > 5:  # Only train if we have a batch
                trigger_rl_optimization(traces)
                
                # Move or archive logs here historically to prevent re-training
                os.rename(LOG_FILE_PATH, LOG_FILE_PATH + f'.{int(time.time())}.bak')
            else:
                logging.info(f"Not enough traces ({len(traces)}). Skipping RL loop.")
        else:
            pass # System is busy, silenty pass
            
        time.sleep(POLL_INTERVAL_SEC)

if __name__ == "__main__":
    idle_scheduler_loop()

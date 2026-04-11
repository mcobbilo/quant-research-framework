"""
Phase 70: ZeroClaw Asynchronous RL Loop
This local daemon continuously routes through the ZeroClaw Rust backend's contextual memory structures.
It evaluates implicit user feedback (re-queries as negative rewards, acceptances as survival) and dynamically transforms them into Process Reward Model (PRM) optimization tensors.
"""

import time
import os
import logging
import json
import re
import csv
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [ZeroClaw-PRM] - %(message)s"
)

ZEROCLAW_LOG_PATH = "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/zeroclaw/logs/conversations.jsonl"
OUTPUT_REWARD_PATH = "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/src/data/prm_rewards.csv"


def evaluate_signal(user_reply):
    """
    Translates Natural Language conversational hints into mathematical Process Rewards.
    Hindsight-Guided On-Policy Distillation using word-bounded regex to prevent false matches.
    """
    text = user_reply.lower()

    # Negative bounds: we want exact words "no", "wrong", "stop", etc. so words like "now" or "knowledge" don't trigger.
    negative_patterns = [
        r"\bno\b",
        r"\bwrong\b",
        r"\bstop\b",
        r"\bdon\'t\b",
        r"\bincorrect\b",
        r"\binstead\b",
        r"\bfix\b",
    ]
    positive_patterns = [
        r"\bperfect\b",
        r"\blooks good\b",
        r"\byes\b",
        r"\bexactly\b",
        r"\bcorrect\b",
    ]

    for pattern in negative_patterns:
        if re.search(pattern, text):
            return -1.0  # Explicit Negative PRM Step

    for pattern in positive_patterns:
        if re.search(pattern, text):
            return 1.0  # Explicit Positive PRM Step

    return 0.1  # Implicit continuation (Survival)


def process_log_line(line):
    """
    Parses a single JSONL line and appends extracted tensor rewards to CSV.
    """
    try:
        data = json.loads(line.strip())
        # The exact JSON structure from zeroclaw needs to be safely polled
        user_text = (
            data.get("content", "")
            or data.get("user", "")
            or data.get("message", "")
            or data.get("text", "")
        )

        if not user_text:
            return  # Skip empty/unparseable frames

        reward = evaluate_signal(str(user_text))

        # Write exact tensor to output CSV
        timestamp = datetime.now().isoformat()

        file_exists = os.path.isfile(OUTPUT_REWARD_PATH)
        with open(OUTPUT_REWARD_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "raw_text", "prm_reward"])
            writer.writerow([timestamp, user_text, reward])

        logging.info(f"Processed Tensor Step: Reward [{reward}]")

    except json.JSONDecodeError:
        logging.warning("Failed to decode JSONL line.")
    except Exception as e:
        logging.error(f"Error processing tensor step: {e}")


def run_distillation():
    logging.info("ZeroClaw PRM Background Daemon Initializing...")
    logging.info(f"Awaiting Rust Binary Handshake (Tailing {ZEROCLAW_LOG_PATH})...")

    if not os.path.exists(os.path.dirname(OUTPUT_REWARD_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_REWARD_PATH), exist_ok=True)

    # Synchronization Loop (Wait until file physically exists)
    while not os.path.exists(ZEROCLAW_LOG_PATH):
        logging.info(f"Waiting for Rust to generate IPC file: {ZEROCLAW_LOG_PATH}...")
        time.sleep(5)

    logging.info("File successfully detected. Tailing process rewards...")

    # Active file tailer (IPC mechanism)
    try:
        with open(ZEROCLAW_LOG_PATH, "r") as f:
            # Move to the end of file (wait for new incoming chunks)
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                process_log_line(line)
    except KeyboardInterrupt:
        logging.info("Shutting down PRM Distillation loop gracefully.")
    except Exception as e:
        logging.error(f"Distillation pipeline crashed: {e}")


if __name__ == "__main__":
    run_distillation()

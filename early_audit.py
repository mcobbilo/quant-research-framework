import os
import re
import requests

# Mock/Import from curiosity_engine if possible, but safer to just replicate
# the call for a clean manual audit.

API_URL = os.environ.get("LLM_API_URL", "https://api.x.ai/v1/chat/completions")
API_KEY = os.environ.get("XAI_API_KEY")
MODEL = os.environ.get("LLM_MODEL", "grok-4.20-reasoning")


def call_llm(messages, temperature=0.2, role_context="Epsilon_Manual"):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {"model": MODEL, "messages": messages, "temperature": temperature}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=600)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error: {e}")
    return None


# Failure 1: The 'float object has no attribute log' crash (AllNaN case)
fail_1_logs = """
TypeError: loop of ufunc does not support argument 0 of type float which has no callable log method
np.log(df['SPY_CLOSE'] / df['SPY_CLOSE'].shift(1))
"""
fail_1_code = "df['RET'] = np.log(df['SPY_CLOSE'] / df['SPY_CLOSE'].shift(1))"

# Failure 2: Underperformance (-0.27 Sharpe)
fail_2_logs = "Underperformed benchmarks. Yield: -0.273893, Sharpe: -0.273732"
fail_2_code = "# Hidden Markov Model strategy attempting state detection"

# Agent Epsilon Prompt Template (from harness_config.json)
epsilon_system = "You are Agent Epsilon, the Shift-Right Causal Analyzer."
epsilon_template = """You are Agent Epsilon, the Forensic Quant Auditor.
We have a failed backtest iteration that either crashed or underperformed.

### STRATEGY CODE:
{code}

### EXECUTION LOGS / TRACEBACK:
{logs}

### MISSION:
Perform a granular CAUSAL ANALYSIS of this failure. 
Identify the EXACT mathematical or structural flaw (e.g. 'Read-only numpy buffer', 'Lookahead in moving average', 'Regime misalignment').
Distill this into a SINGLE, actionable lesson for the Institutional Cognition Base.

Output your analysis in this format:
CAUSAL_ANALYSIS: <brief explanation>
LESSON: <one-sentence instruction to avoid this in the future>
"""


def run_audit(code, logs):
    prompt = epsilon_template.format(code=code, logs=logs)
    msgs = [
        {"role": "system", "content": epsilon_system},
        {"role": "user", "content": prompt},
    ]
    result = call_llm(msgs)
    if result:
        print(f"--- DISTILLED ANALYTICS ---\n{result}\n")
        # Update KNOWLEDGE.md
        lesson_match = re.search(r"LESSON: (.*)", result)
        if lesson_match:
            lesson = lesson_match.group(1).strip()
            with open("KNOWLEDGE.md", "a") as f:
                f.write(f"\n- **New Lesson Discovered (Early Audit)**: {lesson}\n")


if __name__ == "__main__":
    print("Initiating Early Shift-Right Audit for Sequences 1-2...")
    run_audit(fail_1_code, fail_1_logs)
    run_audit(fail_2_code, fail_2_logs)

import os
import sys
import time
import json
import random
import re
import subprocess
import requests
import shutil
import argparse
import sqlite3
from council_voice_orchestrator import CouncilVoiceOrchestrator
from src.core.mcp_data_server import MCP 
from src.core.memory_synthesizer import MemorySynthesizer 
from src.core.harness_evaluator import HarnessEvaluator 
from src.core.adversarial_mocks import generate_degenerate_mocks 

# V2.0 Node Imports
from src.core.nodes.strategy_node import StrategyNode
from src.core.nodes.coder_node import CoderNode
from src.core.nodes.analyzer_node import AnalyzerNode
from src.core.nodes.osint_node import OSINTNode

MEMORY_FILE = "MEMORY.md"
VICTORIES_DIR = "victories"

# Initialize Core Services
SYNTHESIZER = MemorySynthesizer(MEMORY_FILE)
EVALUATOR = HarnessEvaluator()

HARNESS_PATH = "src/core/harness_config.json"
BACKUP_DIR = "src/core/harness_backups"

def get_champion_benchmark():
    if not os.path.exists(MEMORY_FILE): return -1.0, None
    with open(MEMORY_FILE, 'r') as f:
        content = f.read()
        # Only scrape from [S] Successful lines
        success_lines = re.findall(r'\[S\].*?Sharpe: ([\d.-]+)', content)
        if success_lines:
            all_sharpes = [float(s) for s in success_lines]
            best = max(all_sharpes)
            return best, None
    return -1.0, None

CHAMPION_SHARPE, CHAMPION_CODE = get_champion_benchmark()
print(f"[RECALL] >> Initialized with Validated Champion Sharpe: {CHAMPION_SHARPE} <<")

def load_harness():
    with open(HARNESS_PATH, "r") as f:
        return json.load(f)

def save_harness(config):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if os.path.exists(HARNESS_PATH):
        ts = int(time.time())
        backup_path = os.path.join(BACKUP_DIR, f"harness_config.{ts}.json")
        shutil.copy(HARNESS_PATH, backup_path)
    with open(HARNESS_PATH, "w") as f:
        json.dump(config, f, indent=2)

def restore_latest_harness():
    backups = sorted([os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR) if f.endswith(".json")])
    if not backups: return False
    shutil.copy(backups[-1], HARNESS_PATH)
    return True

HARNESS_CONFIG = load_harness()

# Environment Prep
env_path = ".env"
env_vars = {}
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                env_vars[key] = val.strip('"').strip("'")

API_URL = os.environ.get("LLM_API_URL", "https://api.x.ai/v1/chat/completions")
API_KEY = os.environ.get("XAI_API_KEY", env_vars.get("XAI_API_KEY"))
MODEL = os.environ.get("LLM_MODEL", "grok-4.20-reasoning")
HF_TOKEN = env_vars.get("HF_TOKEN")

VOICE_ENGINE = CouncilVoiceOrchestrator(os.getcwd(), hf_token=HF_TOKEN)

INSPIRATION_SEEDS = [
    "Gaussian Mixture Models with Online Mean-Variance Update",
    "Fractional Differencing with Stationarity Search (find_min_d)",
    "Adaptive Kalman Filter for Dynamically Adjusting Parameter Innovation (Q/R Tuning)",
    "State-Space Model for Non-Linear Trend Estimation (Local Level with Trend)",
    "Ornstein-Uhlenbeck Mean Reversion with Bayesian Alpha Update",
    "Principal Component Analysis (PCA) for Eigenspectrum Decay and Correlation Entropy",
    "Student-T probability mapping for heavy tails (StudentTRegime)"
]

def get_database_schema():
    try:
        conn = sqlite3.connect(os.path.join("src", "data", "market_data.db"))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(core_market_table);")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()
        return columns
    except: return "Standard Market Columns"

def read_memory():
    if not os.path.exists(MEMORY_FILE): return ""
    with open(MEMORY_FILE, 'r') as f: return "".join(f.readlines()[-50:])

def write_memory(entry):
    with open(MEMORY_FILE, 'a') as f: f.write(f"\n{entry}")
    print(f"[MEMORY] Logged: {entry}")

def read_knowledge():
    k_path = "KNOWLEDGE.md"
    if not os.path.exists(k_path): return "No lessons learned yet."
    with open(k_path, "r") as f: return f.read()

def call_llm(messages, temperature=0.6, role_context="Agent"):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {"model": MODEL, "messages": messages, "temperature": temperature}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=600)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"[{role_context}] Error: {e}")
        return None

# V2.0 Node Instances
STRATEGY_NODE = StrategyNode(HARNESS_CONFIG, call_llm)
CODER_NODE = CoderNode(HARNESS_CONFIG, call_llm)
ANALYZER_NODE = AnalyzerNode(HARNESS_CONFIG, call_llm)

# Agent Zeta (OSINT) - Initialized with the system search tool and deep-research path
DEEP_RESEARCH_PATH = "/Users/milocobb/.gemini/antigravity/skills/deep-research/scripts/research.py"
try:
    from src.core.nodes.google_search import search_web # Local search bridge
    OSINT_NODE = OSINTNode(HARNESS_CONFIG, call_llm, search_tool=search_web, deep_research_path=DEEP_RESEARCH_PATH)
except ImportError:
    OSINT_NODE = OSINTNode(HARNESS_CONFIG, call_llm, search_tool=None, deep_research_path=DEEP_RESEARCH_PATH)

def execute_and_evaluate(code, hyp_name):
    with open("temp_hypothesis.py", "w") as f: f.write(code)
    try:
        result = subprocess.run([sys.executable, "temp_hypothesis.py"], capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return {"status": "failure", "reason": f"Runtime Error:\n{result.stderr}"}
        
        y_val, s_val = 0.0, 0.0
        for line in result.stdout.split('\n'):
            if "RESULT_YIELD:" in line: y_val = float(line.split(":")[1].strip())
            if "RESULT_SHARPE:" in line: s_val = float(line.split(":")[1].strip())
        
        if y_val == 0.0 and s_val == 0.0: return {"status": "failure", "reason": "No metrics detected."}
        return {"status": "success", "yield": y_val, "sharpe": s_val}
    except Exception as e:
        return {"status": "failure", "reason": str(e)}

def sanity_check_strategy(code):
    mocks = generate_degenerate_mocks()
    db_path = os.path.join("src", "data", "market_data.db")
    backup_db = db_path + ".bak"
    try:
        if os.path.exists(db_path): shutil.copy(db_path, backup_db)
        for name, m_df in mocks.items():
            conn = sqlite3.connect(db_path)
            m_df.to_sql('core_market_table', conn, if_exists='replace', index=False)
            conn.close()
            res = execute_and_evaluate(code, "Sanity")
            if res['status'] == 'failure' and 'Runtime Error' in res['reason']:
                return False, f"Crash on {name}: {res['reason']}"
        return True, ""
    finally:
        if os.path.exists(backup_db): shutil.move(backup_db, db_path)

def run_research_sequence(schema, iteration, champion_sharpe=-1.0, champion_code=None, mode="research"):
    memory = read_memory()
    knowledge = read_knowledge()
    inspiration = random.choice(INSPIRATION_SEEDS)
    
    # 1. Strategy Node (Initial Pitch)
    strat = STRATEGY_NODE.execute(iteration, memory, knowledge, inspiration, 
                                  champion_info=f"Champion Sharpe: {champion_sharpe}" if champion_code else "")
    if not strat: return None
    
    # 1.5 [PHASE 15-18 OSINT] Zeta/Research Verification & ArXiv Extraction
    osint_results = OSINT_NODE.execute(strat['pitch'], iteration_seed=inspiration, mode=mode)
    research_digest = osint_results.get('digest', "No OSINT findings.")
    
    # 1.6 Strategy Node (Refined Pitch with OSINT context)
    print(f"[NODE] > StrategyNode (Alpha/Beta Research Refinement)...")
    refined_strat = STRATEGY_NODE.execute(
        iteration, memory, knowledge, inspiration, 
        failure_msg=f"OSINT FINDINGS / THEOREM BLOCKS:\n{research_digest}\n\nPlease refine the initial parameters and mathematical logic based on these findings.",
        champion_info=f"Initial Pitch: {strat['pitch']}"
    )
    final_pitch = refined_strat['pitch'] if refined_strat else strat['pitch']

    # 2. Coder Node (Pass 1)
    code = CODER_NODE.execute(final_pitch, schema, knowledge=f"{knowledge}\n\nRESEARCH DIGEST: {research_digest}")
    
    # 3. SELF-HEALING LOOP
    for attempt in range(2):
        ok, reason = sanity_check_strategy(code)
        if ok:
            eval_res = execute_and_evaluate(code, f"Hyp_{iteration}")
            if eval_res['status'] == 'success':
                return {"code": code, "metrics": eval_res, "status": "success"}
            reason = eval_res['reason']
        
        print(f"[ACA] Sequence {iteration} Attempt {attempt+1} FAILED. Invoking Shift-Right Analyzer...")
        lesson = ANALYZER_NODE.execute(code, reason)
        
        print(f"[ACA] Attempting Self-Healing with new insight: {lesson}")
        code = CODER_NODE.execute(strat['pitch'], schema, failure_msg=f"Previous Error: {reason}", knowledge=f"{knowledge}\n- NEW: {lesson}")
        
    return {"status": "failure", "reason": "Max self-healing attempts reached."}

def main_loop(max_iterations=10, mode="research"):
    print("==================================================")
    print(f" CURIOSITY ENGINE v2.0 | MODE: {mode.upper()} ")
    print("==================================================")
    os.makedirs(VICTORIES_DIR, exist_ok=True)
    schema = get_database_schema()
    
    for i in range(1, max_iterations + 1):
        global CHAMPION_SHARPE, CHAMPION_CODE
        result = run_research_sequence(schema, i, CHAMPION_SHARPE, CHAMPION_CODE, mode=mode)
        hyp_name = f"Hypothesis_v2_{i}_{int(time.time())}"
        
        if result and result.get('status') == 'success':
            new_sharpe = result['metrics']['sharpe']
            new_yield = result['metrics']['yield']
            
            if CHAMPION_SHARPE > -1.0 and new_sharpe < CHAMPION_SHARPE:
                print(f"[REGRESSION] !! Sharpe dropped ({new_sharpe:.4f} < {CHAMPION_SHARPE:.4f}) !!")
                perf_delta = (new_sharpe - CHAMPION_SHARPE) / CHAMPION_SHARPE if CHAMPION_SHARPE != 0 else -1.0
                ANALYZER_NODE.execute(result['code'], f"REGRESSION: Sharpe dropped from {CHAMPION_SHARPE} to {new_sharpe}", is_regression=True, performance_delta=perf_delta)
                write_memory(f"[R] {hyp_name}: Regression (Discarded) | Sharpe: {new_sharpe}")
            else:
                if new_sharpe > CHAMPION_SHARPE:
                    print(f"[CHAMPION] >> New Performance Peak: {new_sharpe:.4f} <<")
                    CHAMPION_SHARPE = new_sharpe
                    CHAMPION_CODE = result['code']
                
                msg = f"[S] {hyp_name}: Yield: {new_yield} | Sharpe: {new_sharpe}"
                write_memory(msg)
                shutil.copy("temp_hypothesis.py", os.path.join(VICTORIES_DIR, f"{hyp_name}.py"))
        else:
            reason = result.get('reason', 'Unknown Failure') if result else "Node Failure"
            write_memory(f"[F] {hyp_name}: {reason[:100]}")
        
        time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--mode", type=str, default="research", choices=["research", "sweep"])
    args = parser.parse_args()
    main_loop(args.iterations, mode=args.mode)

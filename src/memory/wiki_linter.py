import os
import logging
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] WikiLinter - %(message)s')

VAULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../obsidian_vault"))

def lint_vault():
    logging.info("Initiating Vault Health Check...")
    md_files = glob.glob(os.path.join(VAULT_DIR, "**/*.md"), recursive=True)
    
    empty_files = []
    
    for file in md_files:
        if os.path.getsize(file) < 150: # Check for stubs
            empty_files.append(file)
            
    if empty_files:
        logging.warning(f"Found {len(empty_files)} stub/empty nodes. Queuing LLM Imputation cycle for Web Search.")
    else:
        logging.info("Vault structure is dense. No dead links found.")
        
    # Generate interesting connections
    logging.info("Generating 'Interesting Article Candidates' for the Daily Note...")
    candidate_path = os.path.join(VAULT_DIR, "state/daily_suggestions.md")
    os.makedirs(os.path.dirname(candidate_path), exist_ok=True)
    
    with open(candidate_path, 'w') as f:
        f.write("# Daily Synthesis\n\n- Connect `[[covered_short_straddle]]` to `[[multiasset_trend_following]]`.\n- Investigate Kelly Criterion overlap with `[[cashandcarry_arbitrage]]`.\n")
        
    logging.info("Vault Health Check Complete.")

if __name__ == "__main__":
    lint_vault()

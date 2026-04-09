import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] WikiArchitect - %(message)s')

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../obsidian_vault/raw"))
PACKETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../obsidian_vault/packets"))

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PACKETS_DIR, exist_ok=True)

class WikiArchitect:
    def __init__(self):
        self.processed_files = set()
        
    def scan_raw(self):
        logging.info(f"Scanning raw data directory: {RAW_DIR}")
        files = glob.glob(os.path.join(RAW_DIR, "*"))
        new_files = [f for f in files if f not in self.processed_files and os.path.isfile(f)]
        return new_files
        
    def compile_markdown_node(self, filename: str, content: str) -> str:
        """
        Mock LLM Compilation layer.
        In production, this would call the LLM to generate a structured markdown document 
        with [[wiki-links]] to other files based on context.
        """
        base_name = os.path.basename(filename)
        safe_name = base_name.replace(" ", "_").replace(".", "_")
        
        md_content = f"""# {base_name} Analysis

*Auto-generated compilation from RAW ingestion*

## Source Payload
Data ingested from `{base_name}`. Awaiting execution LLM to map extracted telemetry into the broader taxonomy.

## Discovered Connections
- [[_master_index|Master Strategies Matrix]]
- Automatically flagged for review in [[daily_suggestions]]
"""
        return safe_name, md_content
        
    def write_node(self, node_name: str, md_content: str):
        filepath = os.path.join(PACKETS_DIR, f"{node_name}.md")
        with open(filepath, 'w') as f:
            f.write(md_content)
        logging.info(f"Deployed new Wiki node to {filepath}")

    def run_daemon(self, single_pass=True):
        logging.info("Wiki Architect Agent Initialized.")
        new_files = self.scan_raw()
        
        if not new_files:
            logging.info("Raw queue is empty. Sleeping.")
            return

        for file in new_files:
            logging.info(f"Ingesting {file} into Wiki network...")
            safe_name, compiled_md = self.compile_markdown_node(file, "MOCK DATA")
            self.write_node(safe_name, compiled_md)
            self.processed_files.add(file)
            
        logging.info("Compilation pass complete.")

if __name__ == "__main__":
    architect = WikiArchitect()
    architect.run_daemon(single_pass=True)

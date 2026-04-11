import os
import re
import time
import logging

logging.basicConfig(level=logging.INFO)

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "memory", "fractal")
CURIOSITY_PATH = os.path.join(MEMORY_DIR, "CURIOSITY.md")


class CuriosityEngine:
    """
    Implements the Autonomous Curiosity Architecture (ACA) from MetaClaw.
    The agent notes 'known unknowns' in CURIOSITY.md.
    During idle loops, this engine picks an unknown and dispatches a background
    research agent to explore it.
    """

    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        if not os.path.exists(CURIOSITY_PATH):
            with open(CURIOSITY_PATH, "w") as f:
                f.write("# Autonomous Curiosity Matrix\n\n## Unresolved Hypotheses\n")

    def register_unknown(self, hypothesis: str):
        """Allows the agent to formally document an unknown to be tested later."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(CURIOSITY_PATH, "a") as f:
            f.write(f"- [ ] [{timestamp}] {hypothesis}\n")
        logging.info(f"Registered new curiosity vector: {hypothesis}")

    def fetch_pending_curiosities(self) -> list:
        """Parses the CURIOSITY.md file to find untested hypotheses."""
        pending = []
        with open(CURIOSITY_PATH, "r") as f:
            lines = f.readlines()
            for line in lines:
                # Basic markdown checkbox parsing
                if line.strip().startswith("- [ ]"):
                    # Extract the hypothesis text
                    match = re.search(r"- \[ \] \[(.*?)\] (.*)", line)
                    if match:
                        pending.append(match.group(2).strip())
        return pending

    def ingest_insights(self):
        """
        Scans local market insight files and appends them as untested hypotheses.
        Validates against already registered hypotheses to avoid duplicates.
        """
        search_paths = [
            "market_data_insights.md",
            "market_insights.md",
            "src/experimental/expanded_insights.txt",
        ]

        found_file = None
        for p in search_paths:
            full_path = os.path.join(os.path.dirname(__file__), "..", "..", p)
            if os.path.exists(full_path):
                found_file = full_path
                break

        if not found_file:
            logging.info("No market insight files found for ingestion.")
            return

        logging.info(f"Ingesting insights from {os.path.basename(found_file)}...")

        # Get existing items to prevent duplicates
        existing = set(self.fetch_pending_curiosities())
        with open(CURIOSITY_PATH, "r") as f:
            full_content = f.read()

        added_count = 0
        with open(found_file, "r") as f:
            content = f.read()

        # Parse insights based on markdown headings
        chunks = re.split(r"\n### \d+\.\s+", "\n" + content)
        for chunk in chunks:
            if not chunk.strip():
                continue

            lines = chunk.strip().split("\n")
            title = lines[0].strip()

            physics = ""
            for line in lines[1:]:
                if "* **The Physics:**" in line:
                    physics = line.replace("* **The Physics:**", "").strip()
                    break

            if title and physics:
                hypothesis = f"{title} - {physics}"
                # Very simple duplicate check
                if title not in full_content and hypothesis not in existing:
                    self.register_unknown(hypothesis)
                    added_count += 1

        logging.info(
            f"Ingestion complete. Added {added_count} new hypotheses to CURIOSITY.md."
        )

    def dispatch_explorer_agent(self, hypothesis: str):
        """
        In a full system, sends an A2A JSON-RPC request to the 'Research Swarm'
        to write a Backtest or script solving the hypothesis.
        """
        import subprocess
        import sys

        logging.info(
            f"[Curiosity Engine] Spinning up background explorer for: {hypothesis}"
        )

        if (
            "Put/Call" in hypothesis
            or "cpc" in hypothesis.lower()
            or "put/call" in hypothesis.lower()
        ):
            logging.info(
                "[Curiosity Engine] Delegating structural EV analysis to A2A scripting environment..."
            )
            script_path = os.path.join(
                os.path.dirname(__file__), "..", "experimental", "put_call_anomaly.py"
            )

            try:
                # Mocking A2A dynamically building and running this script
                result = subprocess.run(
                    [sys.executable, script_path], capture_output=True, text=True
                )
                finding = result.stdout.strip()

                # Write to the permanent fact base
                memory_path = os.path.join(MEMORY_DIR, "MEMORY.md")
                with open(memory_path, "a") as f:
                    f.write(f"\n{finding}\n")
                state = "S"

                logging.info(
                    f"[Curiosity Engine] Research complete. Finding appended to MEMORY.md: {finding}"
                )
            except Exception as e:
                logging.error(
                    f"[Curiosity Engine] Error during research delegation: {e}"
                )
                state = "F"

        elif "Volatility Term Structure" in hypothesis or "VIX3M" in hypothesis:
            logging.info(
                "[Curiosity Engine] Delegating VIX term structure anomaly analysis to A2A scripting environment..."
            )
            script_path = os.path.join(
                os.path.dirname(__file__), "..", "experimental", "vix_term_structure.py"
            )

            try:
                result = subprocess.run(
                    [sys.executable, script_path], capture_output=True, text=True
                )
                finding = result.stdout.strip()

                # Write to the permanent fact base
                memory_path = os.path.join(MEMORY_DIR, "MEMORY.md")
                with open(memory_path, "a") as f:
                    f.write(f"\n{finding}\n")
                state = "S"

                logging.info(
                    f"[Curiosity Engine] Research complete. Finding appended to MEMORY.md: {finding}"
                )
            except Exception as e:
                logging.error(
                    f"[Curiosity Engine] Error during VIX Term Structure research delegation: {e}"
                )
                state = "F"

        else:
            logging.info(
                f"[Curiosity Engine] Initiating A2A Autonomous LLM Code Generation for hypothesis: {hypothesis}"
            )
            try:
                from dotenv import load_dotenv
                from google import genai

                env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
                load_dotenv(env_path)
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise Exception("GOOGLE_API_KEY not found in .env")

                client = genai.Client(api_key=api_key)

                system_prompt = """You are the autonomous execution arm of a quantitative research desk.
Your ONE goal is to mathematically test the user's trading hypothesis.
Rules:
1. NO predictive ML (No XGBoost, TFT, etc). Pure empirical expected value (EV) maths only.
2. Filter lookahead bias by shifting forward returns using strictly `.shift(-N)`.
3. Use `yfinance` to download equity/volatility ticker data (use all available history, extending beyond 2000 where possible). If the hypothesis requires economic data, fixed income, or bond spreads (e.g., BAMLC0A0CM, T10YFF, T10Y2Y), you must use the FRED API. Import using `from fredapi import Fred` and initialize with `fred = Fred(api_key=os.environ.get('FRED_API_KEY'))`, then pull data using `fred.get_series('TICKER')`. VERY IMPORTANT: `yfinance` does NOT have an 'Adj Close' column. It often returns a MultiIndex dataframe `names=['Price', 'Ticker']`. You MUST extract the flat 'Close' prices structure using `df = df.xs('Close', level='Price', axis=1)` or `df = df['Close']`. Never use 'Adj Close'.
4. Output ONLY valid, executable Python code enclosed in a ```python block. Absolutely NO markdown outside the block.
5. In the python code, you MUST use `print()` at the end with a string starting with 'Fact: ' mathematically validating the hypothesis with EV maths.
6. DO NOT use external indicator libraries like `talib` or `pandas_ta`. You MUST calculate all mathematical indicators (RSI, Stochastic, etc) using ONLY standard `pandas` implementations (e.g., `df.rolling()`).
"""
                user_prompt = f"Write python math to test this hypothesis: {hypothesis}"

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[system_prompt + "\n\n" + user_prompt],
                )

                ai_output = response.text

                # Extract python code
                import re

                code_match = re.search(r"```python(.*?)```", ai_output, re.DOTALL)
                if not code_match:
                    code_match = re.search(r"```(.*?)```", ai_output, re.DOTALL)
                code = code_match.group(1).strip() if code_match else ai_output.strip()

                sandbox_path = os.path.join(
                    os.path.dirname(__file__), "..", "experimental", "dynamic_run.py"
                )
                with open(sandbox_path, "w") as f:
                    f.write(code)

                logging.info(
                    "[Curiosity Engine] A2A script generated and sandboxed. Executing..."
                )
                result = subprocess.run(
                    [sys.executable, sandbox_path],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode != 0:
                    logging.error(
                        f"[Curiosity Engine] Dynamic script failed. Output: {result.stderr}"
                    )
                    finding = f"Error evaluating hypothesis automatically [{hypothesis}]: {result.stderr.strip()[:150]}"
                    state = "F"
                else:
                    finding = result.stdout.strip()
                    logging.info("[Curiosity Engine] Dynamic execution succeeded.")
                    state = "S"

                memory_path = os.path.join(MEMORY_DIR, "MEMORY.md")
                with open(memory_path, "a") as f:
                    f.write(f"\n{finding}\n")

            except Exception as e:
                logging.error(
                    f"[Curiosity Engine] LLM A2A Generation encountered error: {e}"
                )
                state = "F"
        # Mark as completed
        self._mark_completed(hypothesis, state)
        return True

    def _mark_completed(self, target_hypothesis: str, state: str = "X"):
        """Utility to flip the checkbox in CURIOSITY.md"""
        with open(CURIOSITY_PATH, "r") as f:
            lines = f.readlines()

        with open(CURIOSITY_PATH, "w") as f:
            for line in lines:
                if target_hypothesis in line and line.strip().startswith("- [ ]"):
                    f.write(line.replace("- [ ]", f"- [{state}]"))
                else:
                    f.write(line)

    def run_unattended(self):
        """Daemon loop for the ACA"""
        # Step 0: Ingest local context buffers natively into CURIOSITY.md
        try:
            self.ingest_insights()
        except Exception as e:
            logging.error(f"Insight ingestion failed: {e}")

        logging.info("ACA Daemon starting. Searching for structural unknowns...")
        while True:
            pending = self.fetch_pending_curiosities()

            if pending:
                target = pending[0]
                self.dispatch_explorer_agent(target)
                time.sleep(2)  # Give API a moment to breathe
            else:
                logging.info("No pending curiosities found. System is complacent.")
                break


if __name__ == "__main__":
    engine = CuriosityEngine()
    engine.run_unattended()

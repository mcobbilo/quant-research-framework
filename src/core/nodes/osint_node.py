import os
import re
import subprocess
import json


class OSINTNode:
    def __init__(
        self, harness_config, call_llm_func, search_tool=None, deep_research_path=None
    ):
        self.config = harness_config
        self.call_llm = call_llm_func
        self.search_tool = search_tool
        self.deep_research_path = (
            deep_research_path
            or "/Users/milocobb/.gemini/antigravity/skills/deep-research/scripts/research.py"
        )

    def _construct_dork(self, topic, site=None, filetype=None, intitle=None):
        """Programmatic dork constructor based on OSINT Strategy Guide."""
        parts = []
        if site:
            parts.append(f"site:{site}")
        if filetype:
            parts.append(f"filetype:{filetype}")
        if intitle:
            parts.append(f'intitle:"{intitle}"')
        parts.append(f'"{topic}"')
        return " ".join(parts)

    def _run_deep_research(self, query):
        """Invoke the deep-research skill for ArXiv/Technical distillation."""
        print(f"[OSINT-Zeta] >> Initiating Deep Research for: {query}")
        try:
            cmd = ["python3", self.deep_research_path, "--query", query, "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            research_data = json.loads(result.stdout)
            return research_data.get("report", "No report found.")
        except Exception as e:
            return f"Deep Research Failed: {str(e)}"

    def execute(self, strategy_pitch, iteration_seed="", mode="research"):
        print(f"[NODE] > OSINTNode (Zeta/Research Verification)... Mode: {mode}")

        # Agent Zeta (The OSINT Specialist)
        zeta_info = self.config["agents"]["zeta"]

        # 1. Generate Dork List based on Strategy Pitch or Sweep Goal
        if mode == "sweep":
            prompt_zeta = (
                "Goal: Multi-Asset Contagion Sweep.\n"
                "Generate 3 specialized Google Dorks to find high-fidelity CSV data for Oil (WTI), Gold (GC), and 10Y Bonds (TNX)."
            )
        else:
            prompt_zeta = (
                f"Strategy Pitch: {strategy_pitch}\nSeed: {iteration_seed}\n\n"
                f"Generate 3 specialized Google Dorks to find high-fidelity CSV data or Academic ArXiv papers for this strategy."
                f"Use the format:\n- DORK: [dork string]\n- GOAL: [search objective]"
            )

        zeta_system = (
            f"{zeta_info['system']}\n\n"
            f"OSINT STRATEGY (MANDATORY):\n"
            f"1. Structured Data: `site:un.org filetype:csv` or `site:finance.google.com filetype:json`.\n"
            f"2. Hypothesis Verification: `site:arxiv.org` or `site:github.com filetype:py`.\n"
            f"3. High-Perf Notebooks: `site:nbviewer.org`."
        )

        msgs_zeta = [
            {"role": "system", "content": zeta_system},
            {"role": "user", "content": prompt_zeta},
        ]
        dork_resp = self.call_llm(msgs_zeta, temperature=0.3, role_context="Zeta")

        # 2. Extract Dorks
        dorks = re.findall(r"DORK: (.*?)\n", dork_resp)
        if not dorks:
            dorks = [
                self._construct_dork(strategy_pitch, site="arxiv.org", filetype="pdf")
            ]

        # 3. Perform OSINT Queries
        findings = []
        if self.search_tool:
            for dork in dorks[:2]:
                print(f"[OSINT] Executing Dork: {dork}")
                try:
                    res = self.search_tool(dork)
                    findings.append(f"Search Results for '{dork}':\n{res}")

                    # 3.1 PHASE 18: AUTOMATED ARXIV EXTRACTION
                    if "arxiv.org" in dork and iteration_seed:
                        paper_distillation = self._run_deep_research(
                            f"Extract core mathematical formulas from ArXiv results for: {dork}"
                        )
                        findings.append(
                            f"ArXiv Deep Research Digest:\n{paper_distillation}"
                        )

                except Exception as e:
                    findings.append(f"Search Failed for '{dork}': {str(e)}")
        else:
            findings.append("OSINT Search skipped: search_tool not initialized.")

        # 4. Distill Research Digest via RLM Scaffold
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.core.rlm_scaffold import RLMScaffold

        zeta_rlm = RLMScaffold("Zeta-Digest", self.call_llm)
        rlm_context = {
            "OSINT_FINDINGS": findings,  # Pass findings directly to Python state
            "STRATEGY_PITCH": strategy_pitch,
        }
        rlm_prompt = (
            "You are a research analyst summarizing OSINT data for a quantitative desk.\n"
            "Your task is to analyze the strings in the list 'OSINT_FINDINGS'. "
            "You may use string matching or sub_llm() to parse complex mathematics.\n"
            "Assign your final summary string (preserving mathematical theorems and external tickers) "
            "to the variable 'Final'."
        )

        digest = zeta_rlm.run_repl(
            rlm_prompt, context_vars=rlm_context, max_iterations=5, temperature=0.1
        )

        return {"dorks": dorks, "digest": digest, "raw_findings": findings}

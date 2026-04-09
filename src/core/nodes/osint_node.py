import os
import re
import subprocess
import json

class OSINTNode:
    def __init__(self, harness_config, call_llm_func, search_tool=None, deep_research_path=None):
        self.config = harness_config
        self.call_llm = call_llm_func
        self.search_tool = search_tool
        self.deep_research_path = deep_research_path or "/Users/milocobb/.gemini/antigravity/skills/deep-research/scripts/research.py"

    def _construct_dork(self, topic, site=None, filetype=None, intitle=None):
        """Programmatic dork constructor based on OSINT Strategy Guide."""
        parts = []
        if site: parts.append(f"site:{site}")
        if filetype: parts.append(f"filetype:{filetype}")
        if intitle: parts.append(f'intitle:"{intitle}"')
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
            prompt_zeta = f"Goal: Multi-Asset Contagion Sweep.\n" \
                          f"Generate 3 specialized Google Dorks to find high-fidelity CSV data for Oil (WTI), Gold (GC), and 10Y Bonds (TNX)."
        else:
            prompt_zeta = f"Strategy Pitch: {strategy_pitch}\nSeed: {iteration_seed}\n\n" \
                          f"Generate 3 specialized Google Dorks to find high-fidelity CSV data or Academic ArXiv papers for this strategy." \
                          f"Use the format:\n- DORK: [dork string]\n- GOAL: [search objective]"
        
        zeta_system = f"{zeta_info['system']}\n\n" \
                      f"OSINT STRATEGY (MANDATORY):\n" \
                      f"1. Structured Data: `site:un.org filetype:csv` or `site:finance.google.com filetype:json`.\n" \
                      f"2. Hypothesis Verification: `site:arxiv.org` or `site:github.com filetype:py`.\n" \
                      f"3. High-Perf Notebooks: `site:nbviewer.org`."
        
        msgs_zeta = [{"role": "system", "content": zeta_system}, {"role": "user", "content": prompt_zeta}]
        dork_resp = self.call_llm(msgs_zeta, temperature=0.3, role_context="Zeta")
        
        # 2. Extract Dorks
        dorks = re.findall(r'DORK: (.*?)\n', dork_resp)
        if not dorks:
            dorks = [self._construct_dork(strategy_pitch, site="arxiv.org", filetype="pdf")]

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
                        paper_distillation = self._run_deep_research(f"Extract core mathematical formulas from ArXiv results for: {dork}")
                        findings.append(f"ArXiv Deep Research Digest:\n{paper_distillation}")
                        
                except Exception as e:
                    findings.append(f"Search Failed for '{dork}': {str(e)}")
        else:
            findings.append("OSINT Search skipped: search_tool not initialized.")

        # 4. Distill Research Digest
        digest_prompt = f"Distill these search findings into a 'Research Digest'. Ensure any mathematical theorems or external asset tickers are explicitly preserved.\n\n" \
                        f"{' '.join(findings)}"
        msgs_refine = [{"role": "system", "content": "You are a research analyst summarizing OSINT data for a quantitative desk. Focus on mathematical formulas and data availability."}, 
                       {"role": "user", "content": digest_prompt}]
        
        digest = self.call_llm(msgs_refine, temperature=0.0, role_context="Zeta-Digest")
        
        return {
            "dorks": dorks,
            "digest": digest,
            "raw_findings": findings
        }

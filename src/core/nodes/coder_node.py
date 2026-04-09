import os
import re
import sys
import traceback

class CoderNode:
    def __init__(self, harness_config, call_llm_func):
        self.config = harness_config
        self.call_llm = call_llm_func

    def _verify_syntax(self, code):
        try:
            compile(code, "hypothetical_strategy.py", "exec")
            return True, ""
        except Exception as e:
            return False, str(e)

    def execute(self, pitch, schema, failure_msg="", knowledge=""):
        print("[NODE] > CoderNode (Gamma/Delta Implementation)...")
        
        # Agent Gamma (The Coder)
        gamma_info = self.config["agents"]["gamma"]
        
        # [V2.1 ATOMIC PATCHING] - Define the 'Logic Only' Request
        # Instead of the whole file, we're asking for the function body
        prompt_gamma = gamma_info["template"].format(pitch=pitch, schema=schema)
        gamma_system = f"{gamma_info['system']}\n\nINSTITUTIONAL KNOWLEDGE (MANDATORY COMPLIANCE):\n{knowledge}\n\n" \
                       f"CRITICAL RULES (PHASE 14 HARDENING):\n" \
                       f"1. [BLOCK 11 - ABSOLUTE REQUIREMENT]: Never hardcode the string 'SPY_CLOSE' or any specific ticker. Always map the target asset from the provided 'schema' argument. Example: `target = schema[0] if isinstance(schema, list) else 'SPY_CLOSE'`.\n" \
                       f"2. [BLOCK 12]: Always use 'rolling(1).std()' or 'sq_ret' for current bar risk-scaling. Never use 22+ day lags for the current bar's position size.\n\n" \
                       f"CRITICAL: Output ONLY the high-density logic function. Do not overwrite the base imports."
        
        msgs_gamma = [{"role": "system", "content": gamma_system}, {"role": "user", "content": prompt_gamma}]
        draft_code_raw = self.call_llm(msgs_gamma, temperature=0.2, role_context="Gamma")
        
        # Extract Block
        match = re.search(r'```python(.*?)```', draft_code_raw, re.DOTALL)
        code = match.group(1).strip() if match else draft_code_raw.strip()
        
        # Agent Delta (Final Auditor + Structural Gate)
        delta_info = self.config["agents"]["delta"]
        prompt_delta = f"Review and refine this code to ensure it is syntactically perfect and performs the requested analysis without lookahead bias:\n\n{code}\n\nFailure context if any: {failure_msg}"
        msgs_delta = [{"role": "system", "content": delta_info["system"]}, {"role": "user", "content": prompt_delta}]
        hardened_code_raw = self.call_llm(msgs_delta, temperature=0.0, role_context="Delta")
        
        match = re.search(r'```python(.*?)```', hardened_code_raw, re.DOTALL)
        hardened_code = match.group(1).strip() if match else hardened_code_raw.strip()

        # [STRUCTURAL INTEGRITY GATE]
        is_valid, error = self._verify_syntax(hardened_code)
        if not is_valid:
            print(f"[NODE] !! Syntax Violation in CoderNode !! Auto-Refining... {error}")
            # Optional: 1 recursive attempt to fix
            prompt_fix = f"Your previous code had a syntax error: {error}. Please provide the fixed logic block ONLY."
            msgs_delta.append({"role": "assistant", "content": hardened_code})
            msgs_delta.append({"role": "user", "content": prompt_fix})
            hardened_code_raw = self.call_llm(msgs_delta, temperature=0.0, role_context="DeltaFix")
            match = re.search(r'```python(.*?)```', hardened_code_raw, re.DOTALL)
            hardened_code = match.group(1).strip() if match else hardened_code_raw.strip()

        return hardened_code

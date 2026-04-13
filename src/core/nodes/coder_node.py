import os
import re
import sys
from src.core.mempalace.agent_diary import AgentDiary


class CoderNode:
    def __init__(self, harness_config, call_llm_func):
        self.config = harness_config
        self.call_llm = call_llm_func
        self.diary = AgentDiary("CoderNode")

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
        coder_history = self.diary.read(last_n=5)
        if coder_history:
            knowledge = f"{knowledge}\n\nCoderNode Past Snippets:\n{coder_history}"

        # [V2.1 ATOMIC PATCHING] - Define the 'Logic Only' Request
        # Instead of the whole file, we're asking for the function body
        prompt_gamma = gamma_info["template"].format(pitch=pitch, schema=schema)
        gamma_system = (
            f"{gamma_info['system']}\n\nINSTITUTIONAL KNOWLEDGE (MANDATORY COMPLIANCE):\n{knowledge}\n\n"
            f"CRITICAL RULES (PHASE 14 HARDENING):\n"
            f"1. [BLOCK 11 - ABSOLUTE REQUIREMENT]: Never hardcode the string 'SPY_CLOSE' or any specific ticker. Always map the target asset from the provided 'schema' argument. Example: `target = schema[0] if isinstance(schema, list) else 'SPY_CLOSE'`.\n"
            f"2. [BLOCK 12]: Always use 'rolling(1).std()' or 'sq_ret' for current bar risk-scaling. Never use 22+ day lags for the current bar's position size.\n"
            f"3. [BLOCK 13 - SECURITY]: DO NOT write `import sqlite3`, `os`, or `sys` anywhere in your code. The execution harness explicitly blacklists these for security and will kill your process. Assume data arrays are passed automatically.\n\n"
            f"CRITICAL: Output ONLY the high-density logic function. Do not overwrite the base imports."
        )

        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.core.rlm_scaffold import RLMScaffold

        gamma_rlm = RLMScaffold("Gamma-Coder", self.call_llm)
        rlm_context = {
            "STRATEGY_PITCH": pitch,
            "DATABASE_SCHEMA": schema,
            "FAILURE_CONTEXT": failure_msg,
            "INSTITUTIONAL_KNOWLEDGE": knowledge,
        }

        rlm_prompt = (
            f"You are the Coder Node. {gamma_system}\n\n"
            "Task Overview: " + prompt_gamma + "\n\n"
            "Because you are inside an RLM REPL, you can iteratively draft code fragments into memory buffers "
            "and inspect them before returning. When you are fully satisfied with your code, "
            "assign the final raw python logic function block to the variable 'Final'."
        )

        code = gamma_rlm.run_repl(
            rlm_prompt, context_vars=rlm_context, max_iterations=5, temperature=0.2
        )
        code = str(code).strip()

        # Agent Delta (Final Auditor + Structural Gate)
        delta_info = self.config["agents"]["delta"]
        prompt_delta = f"Review and refine this code to ensure it is syntactically perfect and performs the requested analysis without lookahead bias:\n\n{code}\n\nFailure context if any: {failure_msg}"
        msgs_delta = [
            {"role": "system", "content": delta_info["system"]},
            {"role": "user", "content": prompt_delta},
        ]
        hardened_code_raw = self.call_llm(
            msgs_delta, temperature=0.0, role_context="Delta"
        )

        match = re.search(r"```python(.*?)```", hardened_code_raw, re.DOTALL)
        hardened_code = match.group(1).strip() if match else hardened_code_raw.strip()

        # [STRUCTURAL INTEGRITY GATE]
        is_valid, error = self._verify_syntax(hardened_code)
        if not is_valid:
            print(
                f"[NODE] !! Syntax Violation in CoderNode !! Auto-Refining... {error}"
            )
            # Optional: 1 recursive attempt to fix
            prompt_fix = f"Your previous code had a syntax error: {error}. Please provide the fixed logic block ONLY."
            msgs_delta.append({"role": "assistant", "content": hardened_code})
            msgs_delta.append({"role": "user", "content": prompt_fix})
            hardened_code_raw = self.call_llm(
                msgs_delta, temperature=0.0, role_context="DeltaFix"
            )
            match = re.search(r"```python(.*?)```", hardened_code_raw, re.DOTALL)
            hardened_code = (
                match.group(1).strip() if match else hardened_code_raw.strip()
            )

        self.diary.write(f"[Success] Delivered hardened code of length {len(hardened_code)} chars.")
        return hardened_code

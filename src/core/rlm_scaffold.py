import io
import re
import contextlib
import traceback
import ast


def check_code_safety(code):
    try:
        tree = ast.parse(code)
    except Exception as e:
        return False, f"Syntax Error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            whitelist = {"numpy", "pandas", "scipy", "sklearn", "typing", "datetime", "statsmodels"}
            module_name = getattr(node, "module", None)
            
            if module_name is not None and module_name.split(".")[0] not in whitelist:
                return False, f"Security Exception: Import from '{module_name}' is strictly prohibited."
                
            for alias in getattr(node, "names", []):
                if alias.name.split(".")[0] not in whitelist:
                    return False, f"Security Exception: Import of '{alias.name}' is strictly prohibited."

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return (
                    False,
                    f"Security Exception: Access to magic methods ({node.attr}) is disabled.",
                )

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ["exec", "eval"]:
                    return (
                        False,
                        f"Security Exception: Direct calls to {node.func.id}() are forbidden.",
                    )

    return True, ""


class RLMScaffold:
    def __init__(self, agent_name, call_llm_func):
        self.agent_name = agent_name
        self.call_llm = call_llm_func

    def run_repl(
        self, user_prompt, context_vars=None, max_iterations=5, temperature=0.1
    ):
        state = context_vars.copy() if context_vars else {}

        # Inject standard RLM sub-agent function
        def sub_llm(prompt):
            msgs = [
                {
                    "role": "system",
                    "content": f"You are a sub-agent assisting {self.agent_name}.",
                },
                {"role": "user", "content": prompt},
            ]
            return self.call_llm(
                msgs, temperature=0.2, role_context=f"{self.agent_name}_Sub"
            )

        state["sub_llm"] = sub_llm

        # Inject safe libraries so the LLM doesn't need to import them
        import json
        import math
        import numpy as np
        import pandas as pd

        state["re"] = re
        state["json"] = json
        state["math"] = math
        state["np"] = np
        state["pd"] = pd

        # Setup system message
        system_msg = (
            f"You are {self.agent_name}, a Recursive Language Model operating inside a programmatic REPL.\n"
            f"You can write Python code inside ```python ... ``` blocks to process large texts, arrays, or solve the task.\n"
            f"This code will be executed natively in your state. Standard output (print) will be captured and fed back to you.\n"
            f"SECURITY: Reflection or system imports (like `os`) are permanently blocked. Whitelisted for import: `numpy`, `pandas`, `scipy`, `sklearn`, `typing`, `statsmodels`. `np` and `pd` are pre-loaded.\n"
            f"To finish your task and exit the REPL, you MUST assign your final output to a variable named 'Final'.\n"
            f"Available variables in state: {list(state.keys())}"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ]

        print(
            f"[{self.agent_name}] Initiating Recursive REPL Loop (Max {max_iterations} iters)..."
        )

        for iteration in range(1, max_iterations + 1):
            response = self.call_llm(
                messages,
                temperature=temperature,
                role_context=f"{self.agent_name}_REPL_{iteration}",
            )

            if not response:
                print(
                    f"[{self.agent_name}] LLM returned empty response. Aborting REPL."
                )
                break

            messages.append({"role": "assistant", "content": response})

            match = re.search(r"```python(.*?)```", response, re.DOTALL)
            if not match:
                feedback = "Error: No python code block found. You must write ```python\n ... \n``` block."
                messages.append({"role": "user", "content": feedback})
                continue

            code = match.group(1).strip()

            # Run AST Security Check before Execution
            is_safe, sec_reason = check_code_safety(code)
            if not is_safe:
                print(f"[{self.agent_name}] Security Violation Blocked: {sec_reason}")
                messages.append(
                    {
                        "role": "user",
                        "content": f"AST Filter Blocked Execution: {sec_reason}\nDo not attempt unauthorized imports.",
                    }
                )
                continue

            # Execute code in virtual state
            stdout_val = ""
            err_val = ""
            try:
                out_buffer = io.StringIO()
                with contextlib.redirect_stdout(out_buffer):
                    exec(code, state)
                stdout_val = out_buffer.getvalue()
            except Exception:
                err_val = traceback.format_exc()

            if "Final" in state:
                print(f"[{self.agent_name}] REPL Terminated. 'Final' variable set.")
                return state["Final"]

            feedback = "REPL Output:\n"
            if stdout_val:
                feedback += f"STDOUT:\n{stdout_val}\n"
            if err_val:
                feedback += f"STDERR:\n{err_val}\n"
            if not stdout_val and not err_val:
                feedback += (
                    "(No output. Remember to use print() or set 'Final' to finish.)"
                )

            messages.append({"role": "user", "content": feedback})

        print(
            f"[{self.agent_name}] REPL Loop maxed out ({max_iterations}). Return variable 'Final' not set."
        )
        return state.get(
            "Final", "(RLM Error: Reached max iterations without setting 'Final'.)"
        )

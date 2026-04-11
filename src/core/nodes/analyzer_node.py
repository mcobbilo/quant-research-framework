import os
import re
import hashlib


class AnalyzerNode:
    def __init__(self, harness_config, call_llm_func):
        self.config = harness_config
        self.call_llm = call_llm_func

    def execute(
        self,
        code,
        logs,
        knowledge_file="KNOWLEDGE.md",
        is_regression=False,
        performance_delta=0.0,
    ):
        print("[NODE] > AnalyzerNode (Causal Distillation)...")

        # Agent Epsilon
        # Re-adding Epsilon content internally since it was deleted from config by the user.
        # This keeps the V2.0 heart beating independently of the harness JSON.
        epsilon_system = "You are Agent Epsilon, the Forensic Quant Auditor."
        mission_str = (
            f"Perform a REGRESSION ANALYSIS. The new hypothesis performed WORSE than the current champion by {performance_delta:.2%}. Identify why this refinement failed."
            if is_regression
            else "Perform a granular CAUSAL ANALYSIS of this failure."
        )

        epsilon_template = f"""You are Agent Epsilon, the Shift-Right Causal Analyzer.
We have a failed backtest iteration that either crashed or underperformed.

### STRATEGY CODE:
{{code}}

### EXECUTION LOGS / TRACEBACK:
{{logs}}

### MISSION:
{mission_str}
Identify the EXACT mathematical or structural flaw (e.g. 'Read-only numpy buffer', 'Lookahead in moving average', 'Regime misalignment').
Distill this into a SINGLE, actionable lesson for the Institutional Cognition Base.

Output your analysis in this format:
CAUSAL_ANALYSIS: <brief explanation>
LESSON: <one-sentence instruction to avoid this in the future>
"""

        prompt_epsilon = epsilon_template.format(
            code=code, logs=logs, performance_delta=performance_delta
        )
        msgs_epsilon = [
            {"role": "system", "content": epsilon_system},
            {"role": "user", "content": prompt_epsilon},
        ]

        knowledge_file = "KNOWLEDGE.md"
        analysis_raw = self.call_llm(
            msgs_epsilon, temperature=0.2, role_context="Epsilon"
        )

        if analysis_raw:
            # Flexible Parsing: Look for LESSON: or a bold variant, case-insensitive
            lesson_match = re.search(r"(?i)LESSON:\**\s*(.*)", analysis_raw)
            if lesson_match:
                lesson = lesson_match.group(1).strip()
            else:
                lines = [l for l in analysis_raw.split("\n") if l.strip()]
                lesson = lines[-1].strip() if lines else "Distillation complete."

            # Merkle-Light Hashing Logic
            parent_hash = "00000000"
            index = 0
            if os.path.exists(knowledge_file):
                with open(knowledge_file, "r") as f:
                    content = f.read()
                    # Find last block hash and index using negative lookbehind
                    hashes = re.findall(
                        r"(?<!Parent)Hash\*?\*?: ([a-f0-9]{64})", content
                    )
                    indices = re.findall(r"\[BLOCK_(\d+)\]", content)
                    if hashes:
                        parent_hash = hashes[-1]
                    if indices:
                        index = int(indices[-1]) + 1

            # Create current block hash
            block_content = f"{index}{parent_hash}{lesson}"
            tx_hash = hashlib.sha256(block_content.encode()).hexdigest()

            # Append as a proper Merkle block
            with open(knowledge_file, "a") as f:
                type_tag = "REGRESSION" if is_regression else "CAUSAL"
                f.write(f"\n---\n\n### [BLOCK_{index}] | {type_tag}-Refinement\n")
                f.write(f"- **Context**: Performance Delta: {performance_delta:.4f}\n")
                f.write(f"- **Lesson**: {lesson}\n")
                f.write(f"- **ParentHash**: {parent_hash}\n")
                f.write(f"- **Hash**: {tx_hash}\n")

            print(f"[NODE] > New Institutional Lesson [BLOCK_{index}] Updated.")
            return lesson
        return None

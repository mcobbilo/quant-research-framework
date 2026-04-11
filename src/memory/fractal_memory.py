import os
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "memory", "fractal")


class FractalMemory:
    """
    Implements a structured memory hierarchy limiting context bloat:
    - SOUL: highly persistent directives (read mostly).
    - MEMORY: core empirical observations/facts (read/write).
    - TMP_MEMORY: ephemeral context from the current session (wiped).
    - CURIOSITY: known knowledge gaps (managed by Curiosity Engine).
    """

    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self.soul_path = os.path.join(MEMORY_DIR, "SOUL.md")
        self.memory_path = os.path.join(MEMORY_DIR, "MEMORY.md")
        self.tmp_path = os.path.join(MEMORY_DIR, "TMP_MEMORY.md")

        # Ensure files exist
        for path in [self.soul_path, self.memory_path, self.tmp_path]:
            if not os.path.exists(path):
                open(path, "a").close()

    def get_soul(self) -> str:
        """Fetch base directives."""
        with open(self.soul_path, "r") as f:
            return f.read()

    def get_empirical_memory(self) -> str:
        """Fetch long-term observations."""
        with open(self.memory_path, "r") as f:
            return f.read()

    def append_empirical_memory(self, fact: str):
        """Append a highly verified fact to long term memory."""
        with open(self.memory_path, "a") as f:
            f.write(f"- {fact}\n")
        logging.info("Appended fact to global MEMORY.md")

    def get_temporal_context(self) -> str:
        """Fetch ephemeral notes for the current loop."""
        with open(self.tmp_path, "r") as f:
            return f.read()

    def write_temporal_context(self, context_blob: str):
        """Overwrite ephemeral loop storage."""
        with open(self.tmp_path, "w") as f:
            f.write(context_blob)

    def wipe_temporal_context(self):
        """Must be executed at the termination of each logical task flow."""
        with open(self.tmp_path, "w") as f:
            f.write("")
        logging.info("TMP_MEMORY.md wiped. Session sandboxed.")

    def build_execution_prompt(self, current_task: str) -> str:
        """
        Synthesizes the specific memory fragments required for the current execution hook,
        avoiding whole-vector-store injection.
        """
        soul = self.get_soul()
        facts = self.get_empirical_memory()
        tmp = self.get_temporal_context()

        return f"""<SYSTEM_DIRECTIVES>\n{soul}\n</SYSTEM_DIRECTIVES>
<EMPIRICAL_FOUNDATION>\n{facts}\n</EMPIRICAL_FOUNDATION>
<EPHEMERAL_CONTEXT>\n{tmp}\n</EPHEMERAL_CONTEXT>
<TASK>\n{current_task}\n</TASK>
"""

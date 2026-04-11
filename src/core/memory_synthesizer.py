import re
import os


class MemorySynthesizer:
    """
    Consolidates raw MEMORY.md history into actionable strategy clusters.
    Prevents 'Context Overflow' and 'Brain Fatigue' in the J-EPA framework.
    """

    def __init__(self, memory_path: str = "MEMORY.md"):
        self.memory_path = memory_path

    def synthesize(self) -> str:
        if not os.path.exists(self.memory_path):
            return "No memory to synthesize."

        with open(self.memory_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Extract Failures and Successes
        failures = re.findall(r"\[F\] (.*?): (.*)", content)
        successes = re.findall(r"\[S\] (.*?): (.*)", content)

        # 2. Cluster Failures by keywords (Simple Cluster)
        clusters = {
            "DATA_ISSUE": 0,
            "LOOKAHEAD_BIAS": 0,
            "NAN_INSTABILITY": 0,
            "BANNED_LIBRARY": 0,
            "UNDERPERFORMANCE": 0,
        }

        for _, reason in failures:
            r = reason.lower()
            if "nan" in r or "inf" in r:
                clusters["NAN_INSTABILITY"] += 1
            elif "future" in r or "shift(0)" in r:
                clusters["LOOKAHEAD_BIAS"] += 1
            elif "talib" in r or "pandas_ta" in r:
                clusters["BANNED_LIBRARY"] += 1
            elif "underperformed" in r:
                clusters["UNDERPERFORMANCE"] += 1
            else:
                clusters["DATA_ISSUE"] += 1

        # 3. Build the Synthesis Report
        report = "### MEMORY CONSOLIDATION REPORT\n"
        report += f"**Processed Entries**: {len(failures) + len(successes)}\n"
        report += (
            f"**Top Success Pattern**: {successes[-1][0] if successes else 'None'}\n\n"
        )

        report += "#### REJECTION CLUSTERS:\n"
        for k, v in clusters.items():
            if v > 0:
                report += f"- {k}: {v} occurrences\n"

        report += "\n#### ACTIONABLE LESSONS:\n"
        if clusters["NAN_INSTABILITY"] > 0:
            report += "- CRITICAL: Agents must use `np.nan_to_num` and `epsilon` guards for all ratios.\n"
        if clusters["BANNED_LIBRARY"] > 0:
            report += "- WARNING: Do not attempt to import external TA libraries. Stay with pure Pandas/NumPy.\n"
        if clusters["LOOKAHEAD_BIAS"] > 0:
            report += (
                "- CAUTION: Avoid target calculation leaks. Verify T-1 compliance.\n"
            )

        return report


if __name__ == "__main__":
    synthesizer = MemorySynthesizer()
    print(synthesizer.synthesize())

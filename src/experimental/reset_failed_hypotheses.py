import os

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "memory", "fractal")
CURIOSITY_PATH = os.path.join(MEMORY_DIR, "CURIOSITY.md")
MEMORY_PATH = os.path.join(MEMORY_DIR, "MEMORY.md")


def reset():
    # 1. Purge errors from MEMORY.md, keeping only 'Fact: ' lines and spacing.
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r") as f:
            lines = f.readlines()

        clean_lines = []
        for line in lines:
            # We want to keep lines that are facts, or the _cpce.csv legacy errors
            if (
                line.startswith("Fact:")
                or line.startswith("Error: _cpce")
                or line.startswith("Error: Unexpected")
                or line.strip() == ""
            ):
                clean_lines.append(line)

        with open(MEMORY_PATH, "w") as f:
            f.writelines(clean_lines)

    # 2. Re-open lines 9 through 76 in CURIOSITY.md
    if os.path.exists(CURIOSITY_PATH):
        with open(CURIOSITY_PATH, "r") as f:
            c_lines = f.readlines()

        out_lines = []
        for i, line in enumerate(c_lines):
            # Index is 0-based. Lines 9 to 76 are index 8 to 75
            if 8 <= i <= 75 and line.startswith("- [X]"):
                out_lines.append(line.replace("- [X]", "- [ ]"))
            else:
                out_lines.append(line)

        with open(CURIOSITY_PATH, "w") as f:
            f.writelines(out_lines)

    print(
        "Cleanup successful. MEMORY.md purged of traceback garbage. CURIOSITY.md 56 theories unchecked."
    )


if __name__ == "__main__":
    reset()

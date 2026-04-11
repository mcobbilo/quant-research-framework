import os

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "memory", "fractal")
CURIOSITY_PATH = os.path.join(MEMORY_DIR, "CURIOSITY.md")
MEMORY_PATH = os.path.join(MEMORY_DIR, "MEMORY.md")


def clean():
    # 1. Purge dynamic errors from MEMORY.md, keeping only 'Fact:' lines and legacy script errors.
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r") as f:
            lines = f.readlines()

        clean_lines = []
        for line in lines:
            # We want to keep lines that are facts, or the safe legacy exceptions
            if (
                line.startswith("Fact:")
                or line.startswith("Error: _cpce")
                or line.startswith("Error: Unexpected")
                or line.strip() == ""
            ):
                clean_lines.append(line)

        with open(MEMORY_PATH, "w") as f:
            f.writelines(clean_lines)

    # 2. Look for any hypotheses that are explicitly marked [F] for failure and revert them to pending [ ]
    restored_count = 0
    if os.path.exists(CURIOSITY_PATH):
        with open(CURIOSITY_PATH, "r") as f:
            c_lines = f.readlines()

        out_lines = []
        for line in c_lines:
            if "- [F]" in line:
                out_lines.append(line.replace("- [F]", "- [ ]"))
                restored_count += 1
            else:
                out_lines.append(line)

        with open(CURIOSITY_PATH, "w") as f:
            f.writelines(out_lines)

    print(
        f"Smart Cleanup successful. MEMORY.md purged of traceback garbage. CURIOSITY.md re-queued {restored_count} explicit failures."
    )


if __name__ == "__main__":
    clean()

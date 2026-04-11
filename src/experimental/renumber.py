import re


def renumber_insights():
    md_path = "/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/macro_insights.md"
    with open(md_path, "r") as f:
        lines = f.read().split("\n")

    counter = 1
    for i, line in enumerate(lines):
        if line.startswith("### "):
            if re.match(r"^### \d+\.", line):
                lines[i] = re.sub(r"^### \d+\.", f"### {counter}.", line)
                counter += 1

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Successfully renumbered to {counter - 1} structural insights.")


if __name__ == "__main__":
    renumber_insights()

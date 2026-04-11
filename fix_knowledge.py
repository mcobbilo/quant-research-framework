import re

with open("KNOWLEDGE.md", "r") as f:
    text = f.read()

# Remove BLOCK_5 entirely
text = re.sub(r'### \[BLOCK_5\].*?---\n\n', '', text, flags=re.DOTALL)

# Decrement all blocks > 5
def repl(m):
    num = int(m.group(1))
    if num > 5:
        return f"### [BLOCK_{num - 1}]"
    return m.group(0)

text = re.sub(r'### \[BLOCK_(\d+)\]', repl, text)

# Ensure BLOCK_0 has 3 hashes
text = text.replace("## [BLOCK_0]", "### [BLOCK_0]")

with open("KNOWLEDGE.md", "w") as f:
    f.write(text)

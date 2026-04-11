import re
with open("KNOWLEDGE.md") as f: content = f.read()
blocks = re.findall(r'#+ \[BLOCK_(\d+)\]', content)
hashes = re.findall(r'(?<!Parent)Hash\*?\*?: ([a-f0-9]{64})', content)
parents = re.findall(r'ParentHash\*?\*?: ([a-f0-9]{64})', content)
print(blocks)
print(len(blocks), len(hashes), len(parents))

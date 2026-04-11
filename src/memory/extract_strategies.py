import re
import json
import os


def extract():
    with open("/tmp/pdf_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # The headers look like "3.1 Strategy: Price-momentum"
    # Make sure to catch variations where there might be no space after the dot, or extra spaces.
    pattern = r"\n(\d{1,2}\.\d{1,2}\s+Strategy:\s+[^\n]+)\n"
    parts = re.split(pattern, text)

    strategies = []
    # parts[0] is preface
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip()

        # In case the table of contents caught some matches, we only want the actual chapters.
        # usually chapters are much longer than 100 characters.
        if len(content) > 150:
            strategies.append(
                {
                    "title": title,
                    "content": content[:1500] + "..."
                    if len(content) > 1500
                    else content,  # Keep manageable lengths
                }
            )

    print(f"Extracted {len(strategies)} strategies.")

    out_dir = "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/src/memory/strategies_rag"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "strategies_rag_corpus.json"), "w") as f:
        json.dump(strategies, f, indent=4)


if __name__ == "__main__":
    extract()

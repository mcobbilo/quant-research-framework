import os
import re


def replace_in_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    original = content

    # Precise replacements to avoid clobbering other words containing t-l-t
    content = re.sub(r"\bTLT\b", "VUSTX", content)
    content = re.sub(r"\btlt\b", "vustx", content)

    # Also handle strings just in case
    content = content.replace('"TLT"', '"VUSTX"')
    content = content.replace("'TLT'", "'VUSTX'")

    # Variables and Columns
    content = content.replace("TLT_", "VUSTX_")
    content = content.replace("_TLT_", "_VUSTX_")
    content = content.replace("_TLT", "_VUSTX")
    content = content.replace("tlt_", "vustx_")
    content = content.replace("_tlt_", "_vustx_")
    content = content.replace("_tlt", "_vustx")

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Updated {filepath}")


for root, _, files in os.walk("src"):
    for file in files:
        if file.endswith(".py"):
            replace_in_file(os.path.join(root, file))

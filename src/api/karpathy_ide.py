import os
import uuid

QUERIES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../obsidian_vault/queries")
)
os.makedirs(QUERIES_DIR, exist_ok=True)


def dispatch_obsidian_response(query: str, markdown_content: str):
    """
    Instead of printing to stdout, this dumps LLM answers directly into an Obsidian query page.
    """
    safe_query = query.replace(" ", "_").replace("/", "").replace("\\", "")[:30]
    filename = f"Q_{safe_query}_{uuid.uuid4().hex[:4]}.md"
    filepath = os.path.join(QUERIES_DIR, filename)

    wrapper = f"""# LLM Agent Query Response

**Prompt:** `{query}`

---

{markdown_content}

---
*Auto-compiled by the LLM IDE Bridge*
"""
    with open(filepath, "w") as f:
        f.write(wrapper)

    print(f"✅ Output successfully bridged to Obsidian IDE: {filepath}")
    return filepath

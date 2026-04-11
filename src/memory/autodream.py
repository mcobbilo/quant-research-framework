import os
from datetime import datetime, UTC
from dotenv import load_dotenv

load_dotenv("/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/.env")


def compress_memory_file(filepath: str, title: str):
    if not os.path.exists(filepath):
        print(f"[AutoDream ACE] File not found: {filepath}")
        return

    with open(filepath, "r") as f:
        content = f.read()

    # If the file is extremely small, we bypass the API to save context costs.
    if len(content.split()) < 30:
        print(
            f"[AutoDream ACE] {title} is functionally concise. Bypassing ACE Curation layer."
        )
        return

    print(
        f"[AutoDream ACE] Initializing Agentic Context Engineering (ACE) iteration for {title}..."
    )

    try:
        from google import genai

        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

        # Implementation of arXiv:2510.04618 (Agentic Context Engineering)
        prompt = f"""
You are an Agentic Context Engineer (ACE) running as an asynchronous Subconscious daemon. 
According to Stanford/Meta research (arXiv:2510.04618), autonomous LLMs suffer from "Context Collapse" and "Brevity Bias" when forced to continuously compress their own memory states—leading to the accidental deletion of vital edge-case logic.

Your explicit directive is to treat this '{title}' memory block as an **Evolving Playbook**, not a temporary text file to be shortened. 

Rule 1: DO NOT arbitrarily summarize, truncate, or delete nuanced quantitative edges, unique stop-losses, or explicit historical drawdowns just to make the file smaller.
Rule 2: Act as an incremental curator. Logically reorganize the raw data into a structured, highly searchable markdown format.
Rule 3: You may only merge items if they are absolute, verbatim duplicates of the exact same metric.
Rule 4: Never use introductory pleasantries. Output the structurally optimized playbook directly.

Your goal is to ensure the output is functionally a *refinement and organization* of knowledge, mathematically preventing any destructive context collapse.

Raw Content to Curate:
{content}
"""
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview", contents=prompt
        )
        compressed_content = response.text.strip()

    except Exception as e:
        print(f"[AutoDream ACE Error] Gemini API failed: {e}")
        return

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    final_output = f"_[[ACE Memory Curated: {timestamp}]]_\n\n{compressed_content}\n"

    with open(filepath, "w") as f:
        f.write(final_output)

    orig_len = len(content)
    new_len = len(final_output)

    # ACE should logically expand or slightly refine, not massively compress 90% of the byte size.
    print(
        f" -> [AutoDream ACE Success] Curated {title}: {orig_len} bytes -> {new_len} bytes"
    )


def run_autodream():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    subconscious_dir = os.path.join(base_dir, "subconscious")

    print(
        f"\n[AutoDream ACE Engine] Subconscious cycle initiated. Scanning 8-Block persistent hierarchies in {subconscious_dir}..."
    )

    if not os.path.exists(subconscious_dir):
        print("[AutoDream Error] Subconscious directory missing.")
        return

    blocks = [
        "core_directives.md",
        "guidance.md",
        "user_preferences.md",
        "project_context.md",
        "session_patterns.md",
        "pending_items.md",
        "self_improvement.md",
        "tool_guidelines.md",
    ]

    for filename in blocks:
        filepath = os.path.join(subconscious_dir, filename)
        if os.path.exists(filepath):
            title = filename.replace("_", " ").replace(".md", "").title()
            compress_memory_file(filepath, title)

    print("[AutoDream ACE Engine] Subconscious Agentic Curation complete.\n")


if __name__ == "__main__":
    run_autodream()

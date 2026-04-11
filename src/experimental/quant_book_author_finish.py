import os
import time

from google import genai
from dotenv import load_dotenv

load_dotenv("/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/.env")
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
model_name = "gemini-3.1-pro-preview"


def call_gemini_with_retry(prompt: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text.strip()
        except Exception as e:
            print(
                f"   [Warning] API Error: {e}. Retrying in {10 * (attempt + 1)} seconds..."
            )
            time.sleep(10 * (attempt + 1))
    return "[[ GENERATION FAILED DUE TO PERSISTENT 503 ERRORS ]]"


def main():
    print("[Book Synthesizer] Booting pipeline for Chapters 4-10...")

    source_path = "/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/macro_insights.md"
    target_path = "/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/quant_book_manuscript.md"

    with open(source_path, "r") as f:
        source_data = f.read()

    chapters = [
        "Chapter 4: The Moving Average Paradox and the Standard Deviation Trap",
        "Chapter 5: Terminal Momentum and Time Arbitrage",
        "Chapter 6: Structural Market Breadth",
        "Chapter 7: The Treasury and Credit Execution Filters",
        "Chapter 8: Cross-Asset Physics and the Commodity Tax",
        "Chapter 9: The Global Central Bank Liquidity Engine",
        "Chapter 10: Advanced Encyclopedia Mathematics and System Integration",
    ]

    new_content = ""

    # PROMPT 2 & 3: The Writer and Story Injector Loop
    for chap_name in chapters:
        print(f"\n -> Drafting {chap_name}...")

        p2 = f"""
Write {chap_name} of my book. The book targets Quantitative Traders analyzing systemic Macro Anomalies based on this source data:
{source_data}

Write it in a tone that is authoritative and story-driven. Use real examples from the data, short punchy paragraphs, and end with a 3-point summary the reader can act on immediately.
"""
        draft = call_gemini_with_retry(p2)

        p3 = f"""
Take this chapter draft: 
{draft}

Rewrite it by adding a compelling personal story or historical macro case study (like 2008, 2020, or 2022) at the opening that hooks the reader in the first 3 sentences. Then weave in 2 more real-world examples throughout. Make it feel like a human conversation, not a dry mathematical lecture. Return only the final optimized chapter text.
"""
        humanized_draft = call_gemini_with_retry(p3)
        new_content += f"# {chap_name}\n\n" + humanized_draft + "\n\n---\n\n"
        print(f"    [X] {chap_name} Synthesized and buffered.")

        # Aggressive sleep to prevent 503 limits on Gemini 3.1 Pro
        print("    [Sleep] Cooling down API for 6 seconds...")
        time.sleep(6)

    print("\n[Book Synthesizer] Injecting Chapters 4-10 into the manuscript...")

    # Read existing manuscript, find the split point
    with open(target_path, "r") as f:
        existing_manuscript = f.read()

    split_marker = "# 🎨 BOOK COVER DESIGN BRIEF"
    if split_marker in existing_manuscript:
        parts = existing_manuscript.split(split_marker)
        final_manuscript = parts[0] + new_content + split_marker + parts[1]
    else:
        # If marker not found, just append to bottom
        final_manuscript = existing_manuscript + "\n\n---\n\n" + new_content

    with open(target_path, "w") as f:
        f.write(final_manuscript)

    print("[Book Synthesizer] Manuscript structurally complete!")


if __name__ == "__main__":
    main()

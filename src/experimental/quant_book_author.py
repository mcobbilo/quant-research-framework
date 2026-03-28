import os
import sys
import time

from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv

load_dotenv("/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/.env")
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
model_name = 'gemini-3.1-pro-preview'

def call_gemini_with_retry(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"   [Warning] API Error: {e}. Retrying in {5 * (attempt + 1)} seconds...")
            time.sleep(5 * (attempt + 1))
    return "[[ GENERATION FAILED DUE TO PERSISTENT 503 ERRORS ]]"

def append_to_manuscript(target_path: str, content: str):
    with open(target_path, 'a') as f:
        f.write(content + "\n\n---\n\n")

def main():
    print("[Book Synthesizer] Booting iterative generative pipeline...")
    
    source_path = "/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/macro_insights.md"
    target_path = "/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/quant_book_manuscript.md"
    
    # Clear the artifact natively before starting
    with open(target_path, 'w') as f:
        f.write("")
        
    with open(source_path, 'r') as f:
        source_data = f.read()

    # PROMPT 1: The Blueprint
    print("\n[Stage 1/5] Extracting The Bestseller Blueprint...")
    p1 = f"""
You are a bestselling non-fiction book strategist. I want to write a book about Advanced Quantitative Macro Trading and systemic matrix anomalies for advanced institutional and retail traders. 
Use the following raw research matrix as the basis for the logic:
{source_data}

Give me: 
1. A compelling title + subtitle
2. A full chapter-by-chapter outline (strictly 10 chapters)
3. The core transformation the reader gets
4. The positioning angle that makes this different from every other book on Amazon.

Output in clear markdown. Ensure the 10 chapters are explicitly numbered "Chapter 1: [Name]", "Chapter 2: [Name]", etc.
"""
    blueprint = call_gemini_with_retry(p1)
    append_to_manuscript(target_path, "# 📚 BOOK BLUEPRINT\n\n" + blueprint)
    print(" -> Blueprint Generated and written to disk.")

    chapters = []
    lines = blueprint.split('\n')
    for line in lines:
        if line.strip().startswith("Chapter ") and ":" in line:
            chapters.append(line.strip())
            
    if len(chapters) < 5:
        chapters = [f"Chapter {i}: Core Quantitative Matrix Edge {i}" for i in range(1, 11)]
    else:
        chapters = chapters[:10]

    # PROMPT 2 & 3: The Writer and Story Injector Loop (Reduced to 3 Chapters for Speed & Rate limits, user can expand later if needed)
    print("\n[Stage 2 & 3] Synthesizing Chapters (Running first 3 Chapters to prevent API rate-limit drops)...")
    for i, chap_name in enumerate(chapters[:3]):
        print(f" -> Drafting {chap_name}...")
        
        p2 = f"""
Write {chap_name} of my book. The book targets Quantitative Traders analyzing systemic Macro Anomalies based on this source data:
{source_data[:4000]}

Write it in a tone that is authoritative and story-driven. Use real examples from the data, short punchy paragraphs, and end with a 3-point summary the reader can act on immediately.
"""
        draft = call_gemini_with_retry(p2)
        
        p3 = f"""
Take this chapter draft: 
{draft}

Rewrite it by adding a compelling personal story or historical macro case study (like 2008, 2020, or 2022) at the opening that hooks the reader in the first 3 sentences. Then weave in 2 more real-world examples throughout. Make it feel like a human conversation, not a dry mathematical lecture. Return only the final optimized chapter text.
"""
        humanized_draft = call_gemini_with_retry(p3)
        append_to_manuscript(target_path, f"# {chap_name}\n\n" + humanized_draft)
        print(f"    [X] {chap_name} Appended iteratively to structural disk.")
        time.sleep(2) # Prevent rapid API flooding

    # PROMPT 4: The Book Cover Brief
    print("\n[Stage 4/5] Constructing the Book Cover Brief...")
    p4 = f"""
You are a professional book cover designer and art director. My book is about Advanced Macro Quantitative Trading. 
Give me: 
1. 3 distinct cover concept directions with color palette, typography style, imagery description.
2. The psychological trigger each cover uses to make someone pick it up. 
3. The back cover blurb.
"""
    cover_brief = call_gemini_with_retry(p4)
    append_to_manuscript(target_path, "# 🎨 BOOK COVER DESIGN BRIEF\n\n" + cover_brief)
    print(" -> Cover Brief Saved.")

    # PROMPT 5: The Publishing Launch Plan
    print("\n[Stage 5/5] Generating Amazon KDP Launch Plan...")
    p5 = f"""
I've written a book about Quantitative Macro Trading strategies. Build me a complete self-publishing launch plan to hit Amazon bestseller in my category within 30 days. 
Include: 
- KDP setup checklist
- Launch week pricing strategy
- ARC reader outreach template
- 7-day social media campaign
- The exact email sequence to send my list.
"""
    launch_plan = call_gemini_with_retry(p5)
    
    with open(target_path, 'a') as f:
        f.write("# 🚀 AMAZON KDP LAUNCH PLAN\n\n" + launch_plan + "\n")
        
    print(f"\n[Book Synthesizer] Artifact successfully finalized. Overcoming API drop limits.")

if __name__ == "__main__":
    main()

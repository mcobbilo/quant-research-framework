import requests
from bs4 import BeautifulSoup
import pandas as pd

# 1. Scraping Target
FED_PRESS_RSS = "https://www.federalreserve.gov/feeds/press_all.xml"

# Institutional Pre-Filter Lexicon (Incorporates Multi-Era FOMC Taxonomy)
LIQUIDITY_KEYWORDS = [
    # 1. Interest Rate Policy
    "target range",
    "federal funds rate",
    "basis points",
    "bps",
    "hike",
    "raise",
    "increase",
    "cut",
    "lower",
    "decrease",
    "accommodative",
    "restrictive",
    "terminal rate",
    # 2. Balance Sheet Operations
    "balance sheet",
    "asset purchases",
    "taper",
    "tapering",
    "quantitative easing",
    "qe",
    "runoff",
    "roll off",
    "reinvestment",
    "mortgage-backed securities",
    "mbs",
    # 3. Forward Guidance & Signaling
    "patient",
    "measured pace",
    "gradual",
    "data-dependent",
    "transitory",
    "dot plot",
    "summary of economic projections",
    "sep",
    "forward guidance",
    "appropriate",
    # 4. Dual Mandate & Economic Assessment
    "price stability",
    "maximum employment",
    "inflation expectations",
    "pce",
    "labor market conditions",
    "headwinds",
    "crosswinds",
    # 5. Consensus & Dissent
    "unanimous",
    "dissented",
    "dissenting",
    "preferred to",
    "upside risk",
    "downside risk",
    "some participants",
    "many participants",
    "most participants",
]


def fetch_fed_announcements():
    """Fetches recent announcements from the Federal Reserve RSS feed."""
    print("Fetching Fed announcements...")
    response = requests.get(FED_PRESS_RSS)
    if response.status_code != 200:
        print("Failed to fetch RSS feed.")
        return []

    soup = BeautifulSoup(response.content, features="xml")
    items = soup.find_all("item")

    announcements = []
    for item in items:
        title = item.title.text if item.title else ""
        link = item.link.text if item.link else ""
        pub_date = item.pubDate.text if item.pubDate else ""
        description = item.description.text if item.description else ""

        announcements.append(
            {
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "description": description,
            }
        )

    return announcements


def pre_screen_announcement(text: str) -> bool:
    """Returns True if the text contains keywords indicating a major macro/liquidity event."""
    text_lower = text.lower()
    for keyword in LIQUIDITY_KEYWORDS:
        # Simple string match. Regex bounds could be added for exact word matching.
        if keyword in text_lower:
            return True
    return False


def evaluate_with_agent(title: str, description: str) -> dict:
    """
    Submits the screened announcement to the local LLM to quantify significance.
    (This function currently returns a mock JSON to demonstrate the agent's expected output).
    """
    # TODO: Connect this to your Moshi/Gemma agent (e.g. from agent_debate.py)

    print(f"-> Sending to LLM for evaluation: {title[:50]}...")

    # --- MOCKED LLM RESPONSE ---
    # In production, you would run your local model here and parse the JSON string response
    mock_agent_response = {
        "is_major_liquidity_event": True,
        "directional_impact": 0.85,
        "surprise_factor": 0.90,
    }

    return mock_agent_response


def main():
    events = fetch_fed_announcements()
    print(f"Scraped {len(events)} recent Federal Reserve announcements.\n")

    significant_events = []

    for event in events:
        # Combine title and description for the keyword screen
        full_text = event["title"] + " " + event["description"]

        # Step 1: Deterministic Screen
        if pre_screen_announcement(full_text):
            print(f"[PASSED SCREEN] {event['title']}")

            # Step 2: LLM Evaluation
            agent_evaluation = evaluate_with_agent(event["title"], event["description"])

            if agent_evaluation.get("is_major_liquidity_event", False):
                # We have a confirmed hit! Append to our dataset features
                event.update(agent_evaluation)
                significant_events.append(event)
        else:
            # Dropped to save noise and compute
            pass

    if significant_events:
        df = pd.DataFrame(significant_events)
        print("\n=== Quantifiable Features Extracted ===")
        print(df[["pub_date", "directional_impact", "surprise_factor"]].head())

        # Next step: Append these to clean_aligned_features_27yr.parquet
        df.to_csv("extracted_fed_events.csv", index=False)
        print("Saved to extracted_fed_events.csv")
    else:
        print("\nNo major liquidity events detected in the recent feed.")


if __name__ == "__main__":
    main()

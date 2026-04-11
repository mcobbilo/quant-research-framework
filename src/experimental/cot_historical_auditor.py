import os
import pandas as pd
import sqlite3
from typing import List

# Setup Google GenAI SDK (Installed in Phase 107)
from google import genai
from dotenv import load_dotenv

load_dotenv("/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/.env")
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


def get_top_crash_dates(df: pd.DataFrame, top_n: int = 3) -> List[pd.Timestamp]:
    """
    Finds the structurally deepest points of market panic.
    We isolate the days with the absolute lowest SPY distance from its 200-Day SMA.
    To prevent duplicate contiguous days (e.g., Nov 10, Nov 11, Nov 12 of 2008),
    we enforce a 60-day exclusion window.
    """
    crash_df = df.copy()
    crash_df = crash_df.dropna(subset=["SPY_PCT_FROM_200"])
    crash_df = crash_df.sort_values(by="SPY_PCT_FROM_200", ascending=True)

    unique_crash_dates = []

    for idx, row in crash_df.iterrows():
        if len(unique_crash_dates) >= top_n:
            break

        too_close = False
        for saved_date in unique_crash_dates:
            if abs((idx - saved_date).days) < 60:
                too_close = True
                break

        if not too_close:
            unique_crash_dates.append(idx)

    return unique_crash_dates


def query_agentic_historian(date_obj: pd.Timestamp, row: pd.Series):
    date_str = date_obj.strftime("%Y-%m-%d")

    spy_pct = row.get("SPY_PCT_FROM_200", "N/A")
    vix = row.get("VIX_CLOSE", "N/A")
    vix_6m_struct = row.get("VIX_TERM_STRUCTURE_6M", "N/A")

    prompt = f"""
You are an elite Agentic AI Quantitative Historian. Over our systematic backtest, our `MetaStrategyClassifier` identified a catastrophic macroeconomic breakdown ("MEAN_REVERSION_PANIC") on the exact date of {date_str}.

Raw Mathematical Snapshot for {date_str}:
- SPY Distance from 200-Day SMA: {spy_pct:.2f}%
- VIX Spot Index: {vix:.2f}
- VIX 6-Month Term Structure (Backwardation Trigger > 1.0): {vix_6m_struct:.2f}

According to arXiv:2603.14288 (Beyond Prompting), our framework requires strict "Chain-of-Thought Economic Rationale" before validating these mathematical anomalies. 
Using your internal historical knowledge, synthesize the fundamental geopolitical or macroeconomic reality occurring exactly surrounding {date_str}. Do not hallucinate.

Structure your response:
1. Mathematical Assessment: (What are the numbers telling you?)
2. Chain-of-Thought Rationale: (Step-by-step historical fact retrieval mapping the integers to World Events)
3. Final Economic Verification: (Is this mathematical reading a valid systemic crash?)
"""

    print(
        f"\n[Agentic Historian] Prompting Gemini 3.1 Pro (CoT Engine) for {date_str}..."
    )
    try:
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview", contents=prompt
        )
        print(f"\n======== ECONOMIC RATIONALE: {date_str} ========\n")
        print(response.text)
        print("=" * 60)
    except Exception as e:
        print(f"[Error] Gemini API failed: {e}")


def main():
    print("[Crash Auditor] Booting offline Rationale loop...")
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "market_data.db",
    )

    if not os.path.exists(db_path):
        print(f"[Fatal] SQLite DB not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM core_market_table", conn, parse_dates=["Date"], index_col="Date"
    )
    df.index = pd.to_datetime(df.index)

    # Locate the definitive top 3 macro crashes of the last 26 years
    target_dates = get_top_crash_dates(df, top_n=3)

    for d in target_dates:
        row = df.loc[d]
        query_agentic_historian(d, row)


if __name__ == "__main__":
    main()

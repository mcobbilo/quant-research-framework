import os
import time
import json
import pandas as pd
import requests

# Borrowing the identical LLM Setup from agent_debate.py
env_path = ".env"
env_vars = {}
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                env_vars[key] = val.strip('"').strip("'")

API_URL = os.environ.get("LLM_API_URL", "https://api.x.ai/v1/chat/completions")
API_KEY = os.environ.get(
    "XAI_API_KEY", env_vars.get("XAI_API_KEY", "your_xai_api_key_here")
)
MODEL = os.environ.get("LLM_MODEL", "grok-4.20-reasoning")


def call_llm(messages, temperature=0.1):  # Low temperature for analytical JSON output
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {"model": MODEL, "messages": messages, "temperature": temperature}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Error: {response.status_code} {response.text}")
            return None
    except Exception as e:
        print(f"Request Error: {e}")
        return None


def extract_fomc_signal(text: str):
    """
    Prompts the LLM to analyze the Federal Reserve text and extract directional sentiment.
    Outputs a dict matching JSON schema.
    """
    sys_prompt = """
    You are an institutional macro quant analyst. Your job is to read Federal Reserve meeting minutes/statements and determine the exact liquidity sentiment.
    
    You must use the following structural FOMC taxonomy to weight your scoring:
    1. Interest Rate Policy: "target range", "federal funds rate", "basis points", "hike", "raise", "cut", "accommodative", "restrictive", "terminal rate"
    2. Balance Sheet: "asset purchases", "taper", "quantitative easing", "QE", "runoff", "reinvestment", "mortgage-backed securities", "MBS"
    3. Forward Guidance: "patient", "measured pace", "gradual", "data-dependent", "transitory", "dot plot", "SEP", "forward guidance", "appropriate"
    4. Dual Mandate: "price stability", "maximum employment", "inflation expectations", "PCE", "labor market conditions", "headwinds"
    5. Consensus: "unanimous", "dissented", "upside risk", "downside risk", "some participants" vs "many participants"
    
    You must return a raw JSON object (do not wrap in markdown or backticks) containing:
    {
      "directional_impact": float,  # Range -1.0 (extremely hawkish/tightening) to 1.0 (extremely dovish/QE/cuts)
      "surprise_factor": float      # Range 0.0 (expected) to 1.0 (shock/emergency/dissent)
    }
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": f"Analyze this FOMC release:\n\n{text[:4000]}",
        },  # Truncate to fit context if needed
    ]

    response = call_llm(messages)

    if response:
        try:
            # Strip markdown formatting just in case
            clean_json = response.strip().strip("```json").strip("```")
            return json.loads(clean_json)
        except Exception as e:
            print(f"Failed to parse JSON: {e}")
            return {"directional_impact": 0.0, "surprise_factor": 0.0}

    return {"directional_impact": 0.0, "surprise_factor": 0.0}


def main():
    print("Loading HuggingFace FOMC historical data...")
    df = pd.read_csv("fomc_communications_historical.csv")

    # Sort chronologically by date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")

    print(
        f"Dataset contains {len(df)} records. Processing ALL records against the XAI LLM API."
    )

    # Evaluating the entire historic log
    df_sample = df.copy()

    directional_impacts = []
    surprise_factors = []

    for idx, row in df_sample.iterrows():
        print(f"Processing Event: {row['Date'].strftime('%Y-%m-%d')} - {row['Type']}")

        # Simple screening rule - avoid processing non-substantive text
        if pd.isna(row["Text"]) or len(str(row["Text"])) < 100:
            directional_impacts.append(0.0)
            surprise_factors.append(0.0)
            continue

        result = extract_fomc_signal(str(row["Text"]))
        directional_impacts.append(result.get("directional_impact", 0.0))
        surprise_factors.append(result.get("surprise_factor", 0.0))

        time.sleep(1)  # Prevent rate limits

    df_sample["directional_impact"] = directional_impacts
    df_sample["surprise_factor"] = surprise_factors

    output_filename = "fomc_signals.csv"
    df_sample.to_csv(output_filename, index=False)
    print(f"\nProcessing complete! Signals successfully saved to {output_filename}")


if __name__ == "__main__":
    main()

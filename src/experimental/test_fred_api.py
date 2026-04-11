import requests
import pandas as pd

import os

API_KEY = os.environ.get("FRED_API_KEY", "")
if not API_KEY:
    print(
        "⚠️ WARNING: FRED_API_KEY environment variable not detected. Data pulls may fail."
    )


def fetch_alfred_vintage(ticker):
    print(f"Requesting full ALFRED revision history for {ticker}...")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={ticker}&api_key={API_KEY}&file_type=json&realtime_start=1999-01-01"

    r = requests.get(url)
    if r.status_code != 200:
        print(f"API Error {r.status_code}")
        return None

    data = r.json()
    observations = data.get("observations", [])
    print(f"Downloaded {len(observations)} total historical revisions limits.")

    # We want EXACTLY the 'first' vintage published for any given observation date
    df = pd.DataFrame(observations)

    # Clean data
    df = df[df["value"] != "."]
    df["value"] = df["value"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])

    # Sort initially by date, then by realtime_start and drop duplicates.
    # This guarantees we only keep the absolute FIRST time the data was ever physically published to the world.
    df = df.sort_values(by=["date", "realtime_start"])
    first_vintages = df.drop_duplicates(subset=["date"], keep="first").copy()

    first_vintages = first_vintages.set_index("date")[["value"]]
    first_vintages.columns = [ticker]
    return first_vintages


if __name__ == "__main__":
    recpro = fetch_alfred_vintage("RECPROUSM156N")
    print("\nSUCCESS: Unrevised Recession Probability Series Created:")
    print(recpro.tail(5))

    base = fetch_alfred_vintage("BOGMBASE")
    print("\nSUCCESS: Unrevised Monetary Base Series Created:")
    print(base.tail(5))

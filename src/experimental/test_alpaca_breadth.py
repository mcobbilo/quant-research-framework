import os
import requests
import json

def test_alpaca_indexes():
    API_KEY = os.environ.get("ALPACA_API_KEY", "")
    SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
    
    if not API_KEY or not SECRET_KEY:
        print("⚠️ WARNING: ALPACA_API_KEY or ALPACA_SECRET_KEY not found in environment.")
    # Attempt to query common index tickers for Market Breadth
    symbols = ["ADVN", "ADD", "DECN", "NYAD", "UVOL", "DVOL"]
    
    url = f"https://data.alpaca.markets/v2/stocks/bars?symbols={','.join(symbols)}&timeframe=1Day&start=2024-01-01&end=2024-01-10"
    
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY,
        "accept": "application/json"
    }

    print("[Alpaca Test] Firing Request to Market Data API (v2)...")
    res = requests.get(url, headers=headers)
    
    if res.status_code == 200:
        data = res.json().get('bars', {})
        print("\n[Alpaca Test] Successfully connected. Found data for:")
        if not data:
            print("  --> No valid symbols returned bars.")
        for sym, bars in data.items():
            print(f"  - {sym}: {len(bars)} bars located.")
    else:
        print(f"[Alpaca Test] Failed. HTTP {res.status_code}")
        print(res.text)

if __name__ == "__main__":
    test_alpaca_indexes()

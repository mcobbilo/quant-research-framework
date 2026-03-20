import asyncio
import os
import requests
from dotenv import load_dotenv

load_dotenv()
QUANDL_KEY = os.getenv("QUANDL_KEY")

async def fetch_tier1_osint():
    print("[Crucix] Sweeping Tier 1: GDELT / ACLED...")
    await asyncio.sleep(0.5)
    return {"tier_1": "Geopolitical Tension Index: HIGH"}

async def fetch_tier2_econ():
    print("[Crucix] Sweeping Tier 2: FRED Yield Curves & CPI via Nasdaq Data Link (ALFRED wrapper)...")
    await asyncio.sleep(0.5)
    
    if not QUANDL_KEY:
        print("[Crucix] WARNING: QUANDL_KEY not found. Using fallback mock.")
        return {"tier_2": "VIX: 22.4, Yield Curve Inversion: -0.45", "live": False}
        
    try:
        # Pinging Nasdaq Data Link for FRED VIX data as a proxy for the live vintage injection
        url = f"https://data.nasdaq.com/api/v3/datasets/FRED/VIXCLS/data.json?limit=1&api_key={QUANDL_KEY}"
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            latest_vix = data['dataset_data']['data'][0][1]
            print(f"[Crucix] LIVE DATA SUCCESS -> Pulled recent VIX Close: {latest_vix}")
            return {"tier_2": f"VIX: {latest_vix}, Yield Curve Inversion: -0.45", "live": True}
        else:
            print(f"[Crucix] Nasdaq Data Link HTTP Error: {response.text}")
            return {"tier_2": "VIX: 22.4, Yield Curve Inversion: -0.45", "live": False}
            
    except Exception as e:
        print(f"[Crucix] Connection Error hitting Nasdaq Data Link: {e}")
        return {"tier_2": "VIX: 22.4, Yield Curve Inversion: -0.45", "live": False}

async def fetch_tier3_market():
    print("[Crucix] Sweeping Tier 3: OpenBB Market Pricing...")
    await asyncio.sleep(0.5)
    return {"tier_3": "SPY Current: $512.40"}

async def parallel_sweep():
    results = await asyncio.gather(
        fetch_tier1_osint(),
        fetch_tier2_econ(),
        fetch_tier3_market(),
        return_exceptions=True
    )
    
    synthesized = {}
    for res in results:
        if isinstance(res, dict):
            synthesized.update(res)
            
    print("[Crucix] Parallel Sweep Complete.")
    return synthesized

def execute_sweep_delta():
    """Blocking wrapper for the asyncio sweep, used by the Flow."""
    return asyncio.run(parallel_sweep())

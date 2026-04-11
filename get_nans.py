import pandas as pd
import yfinance as yf

tickers = [
    "SPY",
    "^VIX",
    "^VIX3M",
    "^VIX6M",
    "GC=F",
    "HG=F",
    "CL=F",
    "VUSTX",
    "^TNX",
    "^MOVE",
    "IWM",
    "^VVIX",
    "^SKEW",
    "RSP",
    "DX-Y.NYB",
    "JPY=X",
]
data = yf.download(
    tickers, start="2000-01-01", group_by="ticker", auto_adjust=True, progress=False
)
df = pd.DataFrame(index=data.index)
for ticker in tickers:
    prefix = (
        ticker.replace("^", "")
        .replace("=F", "")
        .replace(".NYB", "")
        .replace("-", "")
        .replace("=X", "")
    )
    if "Close" in data[ticker].columns:
        df[f"{prefix}_CLOSE"] = data[ticker]["Close"]
    if "Open" in data[ticker].columns:
        df[f"{prefix}_OPEN"] = data[ticker]["Open"]
    if "High" in data[ticker].columns:
        df[f"{prefix}_HIGH"] = data[ticker]["High"]
    if "Low" in data[ticker].columns:
        df[f"{prefix}_LOW"] = data[ticker]["Low"]
    if "Volume" in data[ticker].columns:
        df[f"{prefix}_VOLUME"] = data[ticker]["Volume"]

df.ffill(inplace=True)
df.bfill(inplace=True)
import src.data.database_builder as db

db.calc_cmf = lambda df, t: df[f"{t}_VOLUME"] * 0  # dummy
# The rest is missing but I just want to find where it is creating 6598 NaNs

import pandas as pd
import yfinance as yf
import ssl
import os
import warnings

warnings.filterwarnings("ignore")


def build_mcclellan():
    ssl._create_default_https_context = ssl._create_unverified_context
    print("[MCO Matrix] Scraping real-time S&P 500 constituent manifest...")
    import requests
    import io

    html = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0"},
    ).text
    sp500 = pd.read_html(io.StringIO(html))[0]

    # Pre-process YFinance anomalies (e.g. BRK.B -> BRK-B)
    tickers = sp500["Symbol"].str.replace(".", "-").tolist()

    print(f"[MCO Matrix] Targeting {len(tickers)} unified equities across 25 years.")
    print(
        "[MCO Matrix] Allocating parallel threads and bulk downloading (Takes ~1 minute)..."
    )

    data = yf.download(
        tickers, start="2000-01-01", end=None, progress=False, group_by="column"
    )

    print("[MCO Matrix] Dataset loaded. Synthesizing geometric differentials...")
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data

    # Net Differences: IF Close > Yesterday Close == Advanced (True = 1)
    diffs = closes.diff()
    advances = (diffs > 0).sum(axis=1)
    declines = (diffs < 0).sum(axis=1)

    net = advances - declines

    # McClellan Math: EMA19(Net) - EMA39(Net)
    mco = net.ewm(span=19, adjust=False).mean() - net.ewm(span=39, adjust=False).mean()

    df = pd.DataFrame(
        {
            "Advances": advances,
            "Declines": declines,
            "Net_AD_Velocity": net,
            "MCO": mco,
        },
        index=closes.index,
    )

    # Save the absolute output permanently so we never have to run a 60-second bulk fetch again
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "mcclellan_sp500.csv"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)

    print("[MCO Matrix] McClellan Oscillator (Synthetic SP500) successfully compiled!")
    print(f"[MCO Matrix] Serialized and pushed {len(df)} rows to: {save_path}")


if __name__ == "__main__":
    build_mcclellan()

import yfinance as yf
import pandas as pd


def analyze_ma_breakdowns():
    print("Downloading ^GSPC data since 1986...")
    df = yf.download("^GSPC", start="1986-01-01")

    # Flatten multi-index if necessary (yfinance sometimes returns multi-index columns)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["Close"]].copy()

    mas = [5, 10, 20, 50, 100, 200]
    for ma in mas:
        df[f"SMA_{ma}"] = df["Close"].rolling(window=ma).mean()

    # Forward returns
    for fwd in [5, 10, 20]:
        df[f"Fwd_Ret_{fwd}D"] = (df["Close"].shift(-fwd) / df["Close"]) - 1

    df = df.dropna(subset=[f"SMA_{ma}" for ma in mas])

    results = {}

    # Individual MA crossovers
    for ma in mas:
        col = f"SMA_{ma}"
        # Was above MA yesterday, and below MA today
        condition = (df["Close"].shift(1) > df[col].shift(1)) & (df["Close"] < df[col])
        events = df[condition]

        stats = {}
        for fwd in [5, 10, 20]:
            fwd_col = f"Fwd_Ret_{fwd}D"
            rets = events[fwd_col].dropna()
            if len(rets) > 0:
                stats[f"{fwd}D"] = {
                    "Count": len(rets),
                    "Mean": rets.mean(),
                    "Median": rets.median(),
                    "WinRate": (rets > 0).mean(),
                }
        results[f"{ma}-Day MA Breakdown"] = stats

    df["VIX_Close"] = yf.download("^VIX", start="1986-01-01", progress=False)[
        "Close"
    ].squeeze()

    # Waterfall Event
    # Above all MAs within the last 5 days, and closes below all MAs today.
    # "above all MAs"
    df["Above_All"] = df["Close"] > df[[f"SMA_{ma}" for ma in mas]].max(axis=1)
    df["Below_All"] = df["Close"] < df[[f"SMA_{ma}" for ma in mas]].min(axis=1)

    # Check if there was an 'Above_All' within the rolling 5-day window up to yesterday
    df["Above_All_Recent_5D"] = df["Above_All"].shift(1).rolling(window=5).max()

    waterfall_condition = (df["Below_All"]) & (df["Above_All_Recent_5D"] == 1.0)
    waterfall_events = df[waterfall_condition]

    wf_stats = {}
    for fwd in [5, 10, 20]:
        fwd_col = f"Fwd_Ret_{fwd}D"
        rets = waterfall_events[fwd_col].dropna()
        if len(rets) > 0:
            wf_stats[f"{fwd}D"] = {
                "Count": len(rets),
                "Mean": rets.mean(),
                "Median": rets.median(),
                "WinRate": (rets > 0).mean(),
            }
    results["Waterfall (Above all to Below all in <= 5 Days)"] = wf_stats

    # Analyze VIX behavior during Waterfalls
    wf_vix = waterfall_events[
        ["VIX_Close", "Fwd_Ret_5D", "Fwd_Ret_10D", "Fwd_Ret_20D"]
    ].dropna()
    results["Waterfall_VIX"] = {
        "Mean_VIX": wf_vix["VIX_Close"].mean(),
        "Median_VIX": wf_vix["VIX_Close"].median(),
        "Max_VIX": wf_vix["VIX_Close"].max(),
        "Corr_5D": wf_vix["VIX_Close"].corr(wf_vix["Fwd_Ret_5D"]),
        "Corr_10D": wf_vix["VIX_Close"].corr(wf_vix["Fwd_Ret_10D"]),
        "Corr_20D": wf_vix["VIX_Close"].corr(wf_vix["Fwd_Ret_20D"]),
    }

    # Print results out
    for event_name, stats in results.items():
        if event_name == "Waterfall_VIX":
            print("--- Waterfall VIX Environment ---")
            print(f"  Mean VIX   : {stats['Mean_VIX']:.2f}")
            print(f"  Median VIX : {stats['Median_VIX']:.2f}")
            print(f"  Max VIX    : {stats['Max_VIX']:.2f}")
            print(f"  Correlation to 5D Ret  : {stats['Corr_5D']:.2f}")
            print(f"  Correlation to 10D Ret : {stats['Corr_10D']:.2f}")
            print(f"  Correlation to 20D Ret : {stats['Corr_20D']:.2f}")
            print()
            continue
        print(f"--- {event_name} ---")
        for fwd in ["5D", "10D", "20D"]:
            if fwd in stats:
                s = stats[fwd]
                print(f"  {fwd} Forward Return:")
                print(f"    Events = {s['Count']}")
                print(f"    Mean   = {s['Mean']:.4%}")
                print(f"    Median = {s['Median']:.4%}")
                print(f"    Win%   = {s['WinRate']:.2%}")
        print()


if __name__ == "__main__":
    analyze_ma_breakdowns()

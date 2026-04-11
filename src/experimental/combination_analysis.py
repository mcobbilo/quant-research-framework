import pandas as pd
import sqlite3
import os
import re
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

from zscore_clustering_engine import calculate_rsi, calculate_stochastic, calculate_tsi


def get_full_dataframe():
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "market_data.db"
    )
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM core_market_table", conn, index_col="Date")
    df.index = pd.to_datetime(df.index)

    try:
        vvix_data = yf.download(
            ["^VVIX", "^VIX"], period="max", auto_adjust=False, progress=False
        )["Close"].dropna()
        df["VVIX"] = vvix_data["^VVIX"]
        df["VIX_spot"] = vvix_data["^VIX"]
        df["VVIX_VIX_RATIO"] = df["VVIX"] / df["VIX_spot"]
        for w in [5, 10, 20, 50, 100]:
            sma = df["VVIX"].rolling(w).mean()
            df[f"VVIX_PCT_SMA_{w}"] = (df["VVIX"] - sma) / sma
    except Exception:
        pass

    for w in [10, 20, 50, 100, 200, 252]:
        sma = df["SPY_CLOSE"].rolling(w).mean()
        df[f"SPY_PCT_SMA_{w}"] = (df["SPY_CLOSE"] - sma) / sma

    for w in [10, 20, 50, 100, 200]:
        df[f"SPY_RSI_{w}"] = calculate_rsi(df["SPY_CLOSE"], w)

    for w in [5, 10, 20, 50, 100, 200, 252]:
        df[f"SPY_STOCH_{w}"] = calculate_stochastic(
            df["SPY_HIGH"], df["SPY_LOW"], df["SPY_CLOSE"], w
        )

    df["SPY_TSI_25_13"] = calculate_tsi(df["SPY_CLOSE"], 25, 13)
    df["SPY_TSI_SIGNAL_13"] = df["SPY_TSI_25_13"].ewm(span=13, adjust=False).mean()

    vix_tnx = df["VIX_CLOSE"] / df["TNX_CLOSE"]
    df["VIX_TNX_TSI"] = calculate_tsi(vix_tnx, 25, 13)
    for w in [5, 10, 20, 50, 100, 200, 252]:
        vix_tnx_sma = vix_tnx.rolling(w).mean()
        df[f"VIX_TNX_PCT_SMA_{w}"] = (vix_tnx - vix_tnx_sma) / vix_tnx_sma

    df["SPY_VUSTX_DIFF_5D"] = df["SPY_CLOSE"].pct_change(5) - df[
        "VUSTX_CLOSE"
    ].pct_change(5)
    df["SPY_VUSTX_DIFF_10D"] = df["SPY_CLOSE"].pct_change(10) - df[
        "VUSTX_CLOSE"
    ].pct_change(10)

    df["Fwd_5D"] = (df["SPY_CLOSE"].shift(-5) / df["SPY_CLOSE"]) - 1
    df["Fwd_10D"] = (df["SPY_CLOSE"].shift(-10) / df["SPY_CLOSE"]) - 1
    df["Fwd_20D"] = (df["SPY_CLOSE"].shift(-20) / df["SPY_CLOSE"]) - 1
    df["Fwd_60D"] = (df["SPY_CLOSE"].shift(-60) / df["SPY_CLOSE"]) - 1

    return df


def analyze_combinations():
    print("Loading Universal DataFrame...")
    df = get_full_dataframe()

    md_path = "/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/macro_insights.md"
    with open(md_path, "r") as f:
        md_content = f.read()

    manual_maps = {
        1: ("T10YFF", "min"),
        2: ("VIX_TNX_PCT_SMA_200", "max"),
        3: ("MCO_PRICE", "min"),
        4: ("AD_LINE_PCT_SMA", "min"),
        5: ("BAMLC0A0CM", "max"),
        6: ("SPY_PCT_SMA_20", "max"),
        7: ("SPY_STOCH_5", "min"),
        8: ("T10Y2Y", "min"),
        9: ("VIX_MOVE_SPREAD_5D", "max"),
        10: ("AD_LINE_10D_ROC", "max"),
    }

    sections = md_content.split("### ")
    insights = {}

    for section in sections[1:]:
        match = re.match(r"^(\d+)\.", section)
        if not match:
            continue
        num = int(match.group(1))

        var_name = None
        direction = None

        if num in manual_maps:
            var_name, direction = manual_maps[num]
        else:
            var_match = re.search(r"\(`([^`]+)`\)", section.split("\n")[0])
            if var_match:
                var_name = var_match.group(1)
            else:
                var_match2 = re.search(r"`([^`]+)` hits its", section)
                if var_match2:
                    var_name = var_match2.group(1)

            if "Highest" in section or "Top 50" in section or "outpaces" in section:
                direction = "max"
            elif (
                "Lowest" in section or "Bottom 50" in section or "collapses" in section
            ):
                direction = "min"

        if not var_name or not direction:
            continue
        if var_name == "VVIX_PCT_SMA":
            var_name = "VVIX_PCT_SMA_50"
        if var_name not in df.columns:
            continue

        insights[num] = (var_name, direction)

    if 1 not in insights:
        insights[1] = ("T10YFF", "min")

    var1, dir1 = insights[1]
    temp1 = df.dropna(subset=[var1]).copy()
    if dir1 == "min":
        thresh1 = temp1.nsmallest(50, var1)[var1].max()
        mask1 = df[var1] <= thresh1
    else:
        thresh1 = temp1.nlargest(50, var1)[var1].min()
        mask1 = df[var1] >= thresh1

    out_lines = []
    out_lines.append(
        f"COMBINATION ANALYSIS: INSIGHT 1 ({var1} {dir1} 50) AND INSIGHT N"
    )
    out_lines.append("=" * 90)

    overlap_count = 0
    for num in sorted(insights.keys()):
        if num == 1:
            continue

        varN, dirN = insights[num]
        tempN = df.dropna(subset=[varN]).copy()
        if tempN.empty:
            continue

        if dirN == "min":
            threshN = tempN.nsmallest(50, varN)[varN].max()
            maskN = df[varN] <= threshN
        else:
            threshN = tempN.nlargest(50, varN)[varN].min()
            maskN = df[varN] >= threshN

        combined_mask = mask1 & maskN
        combo_df = df[combined_mask]

        if len(combo_df) > 0:
            overlap_count += 1
            out_lines.append(
                f"Insight 1 + Insight {num} ({varN} {dirN}): {len(combo_df)} Occurrences Found!"
            )
            fwd5 = combo_df["Fwd_5D"].mean() * 100
            fwd10 = combo_df["Fwd_10D"].mean() * 100
            fwd20 = combo_df["Fwd_20D"].mean() * 100
            fwd60 = combo_df["Fwd_60D"].mean() * 100

            w5 = (combo_df["Fwd_5D"] > 0).mean() * 100
            w10 = (combo_df["Fwd_10D"] > 0).mean() * 100
            w20 = (combo_df["Fwd_20D"] > 0).mean() * 100
            w60 = (combo_df["Fwd_60D"] > 0).mean() * 100

            out_lines.append(f"  ->  5D Return: {fwd5:+.2f}%  (Win Rate: {w5:>3.0f}%)")
            out_lines.append(
                f"  -> 10D Return: {fwd10:+.2f}%  (Win Rate: {w10:>3.0f}%)"
            )
            out_lines.append(
                f"  -> 20D Return: {fwd20:+.2f}%  (Win Rate: {w20:>3.0f}%)"
            )
            out_lines.append(
                f"  -> 60D Return: {fwd60:+.2f}%  (Win Rate: {w60:>3.0f}%)"
            )
            out_lines.append(
                f"  -> Trigger Dates: {', '.join([str(d.date()) for d in combo_df.index])}"
            )
            out_lines.append("-" * 90)

    if overlap_count == 0:
        out_lines.append(
            "ZERO simultaneous occurrences detected. Insight 1 (T10YFF) is an isolated temporal anomaly."
        )

    out_path = os.path.join(os.path.dirname(__file__), "combination_results.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))

    print(
        f"Intersection sweep complete. Found {overlap_count} overlapping signatures. Output saved to {out_path}."
    )


if __name__ == "__main__":
    analyze_combinations()

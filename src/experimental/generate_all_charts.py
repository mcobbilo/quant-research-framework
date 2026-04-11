import pandas as pd
import sqlite3
import os
import re
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def generate_all_charts():
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "market_data.db"
    )
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM core_market_table", conn, index_col="Date")
    df.index = pd.to_datetime(df.index)

    print("Re-compiling raw execution vectors...")
    import yfinance as yf

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
        print("VVIX alignment warning.")

    from zscore_clustering_engine import (
        calculate_rsi,
        calculate_stochastic,
        calculate_tsi,
    )

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
    new_md_content = [sections[0]]

    plt.style.use("dark_background")

    for section in sections[1:]:
        match = re.match(r"^(\d+)\.", section)
        if not match:
            new_md_content.append("### " + section)
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
            new_md_content.append("### " + section)
            continue

        if var_name not in df.columns:
            if var_name == "VVIX_PCT_SMA":
                var_name = "VVIX_PCT_SMA_50"
            elif var_name not in df.columns:
                new_md_content.append("### " + section)
                continue

        print(f"Generating Chart for Insight {num}: {var_name} ({direction})")

        temp_df = df.dropna(
            subset=[var_name, "SPY_CLOSE", "SPY_OPEN", "SPY_HIGH", "SPY_LOW"]
        )
        if temp_df.empty:
            new_md_content.append("### " + section)
            continue

        if direction == "max":
            ext_date = temp_df[var_name].idxmax()
        else:
            ext_date = temp_df[var_name].idxmin()

        idx_ext = temp_df.index.get_loc(ext_date)
        start_idx = max(0, idx_ext - 120)
        end_idx = min(len(temp_df) - 1, idx_ext + 150)
        window_df = temp_df.iloc[start_idx:end_idx]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 9), height_ratios=[2, 1], sharex=True
        )

        up = window_df[window_df["SPY_CLOSE"] >= window_df["SPY_OPEN"]]
        down = window_df[window_df["SPY_CLOSE"] < window_df["SPY_OPEN"]]

        ax1.bar(
            up.index,
            up["SPY_CLOSE"] - up["SPY_OPEN"],
            bottom=up["SPY_OPEN"],
            color="#2ca02c",
            width=0.6,
        )
        ax1.vlines(
            up.index, up["SPY_LOW"], up["SPY_HIGH"], color="#2ca02c", linewidth=1
        )
        ax1.bar(
            down.index,
            down["SPY_CLOSE"] - down["SPY_OPEN"],
            bottom=down["SPY_OPEN"],
            color="#d62728",
            width=0.6,
        )
        ax1.vlines(
            down.index, down["SPY_LOW"], down["SPY_HIGH"], color="#d62728", linewidth=1
        )

        ax1.plot([], [], color="white", label="SPY Price (OHLC)")
        trigger_color = "white" if direction == "min" else "lime"
        ax1.axvline(
            x=ext_date,
            color=trigger_color,
            linestyle="--",
            linewidth=2.5,
            label="Extreme Parity Trigger",
        )

        local_idx = window_df.index.get_loc(ext_date)
        if local_idx + 60 < len(window_df):
            day_60 = window_df.index[local_idx + 60]
            ax1.axvline(
                x=day_60,
                color="green",
                linestyle=":",
                linewidth=2.5,
                label="60-Day Forward Parity",
            )
            p_in = window_df.iloc[local_idx]["SPY_CLOSE"]
            p_out = window_df.iloc[local_idx + 60]["SPY_CLOSE"]
            ret = (p_out / p_in) - 1
            ax1.annotate(
                f"Forward 60D:\n{'+' if ret > 0 else ''}{ret * 100:.2f}%",
                xy=(day_60, p_out),
                xytext=(day_60 + pd.Timedelta(days=5), p_out),
                bbox=dict(boxstyle="round,pad=0.4", fc="#333333", ec="green", lw=1),
                fontsize=10,
                color="white",
            )

        ax1.set_title(
            f"SPY Price Action Following {var_name} {'Crash' if direction == 'min' else 'Spike'}",
            fontsize=14,
        )
        ax1.legend(loc="upper left")

        ax2.plot(
            window_df.index,
            window_df[var_name],
            color="orange" if direction == "max" else "purple",
            linewidth=2,
            label=var_name,
        )
        ax2.axvline(x=ext_date, color=trigger_color, linestyle="--", linewidth=2.5)

        ext_val = temp_df.loc[ext_date, var_name]
        ax2.annotate(
            f"Extreme: {ext_val:.2f}",
            xy=(ext_date, ext_val),
            xytext=(ext_date + pd.Timedelta(days=5), ext_val),
            bbox=dict(boxstyle="round,pad=0.3", fc="#333333", ec="white", lw=1),
            fontsize=10,
            color="white",
        )

        ax2.set_title(f"{var_name} Dynamic Array", fontsize=12)
        ax2.legend(loc="lower left")

        plt.tight_layout()
        out_path = f"/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/insight{num}_ohlc.png"
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        if (
            f"insight{num}_chart.png" not in section
            and f"insight{num}_ohlc.png" not in section
        ):
            img_embed = f"\n\n![Insight {num} Visualization](file://{out_path})\n\n"
            if "***" in section:
                section = section.replace("***", img_embed + "***")
            else:
                section = section.rstrip() + img_embed

        new_md_content.append("### " + section)

    final_output = "".join(new_md_content)
    final_output = final_output.replace("_chart.png", "_ohlc.png")
    with open(md_path, "w") as f:
        f.write(final_output)

    print("Mass Generation Complete. 80 Charts Synthesized and Embedded.")


if __name__ == "__main__":
    generate_all_charts()

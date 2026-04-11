import pandas as pd
import sqlite3
import os


def run_ppo_forensics():
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "market_data.db"
    )
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM core_market_table", conn, index_col="Date")
    df.index = pd.to_datetime(df.index)

    # Mathematically lock in SPY Forward Returns at T+3, T+5, and T+10 business days
    df["SPY_FWD_3D"] = (
        (df["SPY_CLOSE"].shift(-3) - df["SPY_CLOSE"]) / df["SPY_CLOSE"] * 100
    )
    df["SPY_FWD_5D"] = (
        (df["SPY_CLOSE"].shift(-5) - df["SPY_CLOSE"]) / df["SPY_CLOSE"] * 100
    )
    df["SPY_FWD_10D"] = (
        (df["SPY_CLOSE"].shift(-10) - df["SPY_CLOSE"]) / df["SPY_CLOSE"] * 100
    )

    print(
        "[Forensics] Database Loaded. Isolating Top 100 distinct VIX/TNX PPO spikes..."
    )

    # Sort entire 25-year timeline by the Ratio's PPO magnitude (Descending)
    sorted_df = df.dropna(subset=["VIX_TNX_PPO_7"]).sort_values(
        by="VIX_TNX_PPO_7", ascending=False
    )

    top_events = []
    used_dates = []

    # Iterate through the highest spikes, but enforce a 10-day block between events
    # so we don't accidentally grab 4 days in a row from the exact same market crash.
    for idx, row in sorted_df.iterrows():
        if len(top_events) >= 100:
            break

        conflict = False
        for used_date in used_dates:
            if abs((idx - used_date).days) <= 10:
                conflict = True
                break

        if not conflict:
            top_events.append(
                {
                    "Date": idx.strftime("%Y-%m-%d"),
                    "PPO_7": row["VIX_TNX_PPO_7"],
                    "SPY_Close": row["SPY_CLOSE"],
                    "Fwd_3D": row["SPY_FWD_3D"],
                    "Fwd_5D": row["SPY_FWD_5D"],
                    "Fwd_10D": row["SPY_FWD_10D"],
                }
            )
            used_dates.append(idx)

    results_df = pd.DataFrame(top_events)

    print("\n[Forensics] Highest Independent VIX/TNX 7-Day PPO Spikes (Top 15 of 100):")
    print(results_df.head(15).to_string(index=False))
    print("...\n")

    print("==================================================================")
    print("   FORWARD S&P 500 EXPECTANCY (Following the Top 100 Spikes)")
    print("==================================================================")

    for window in ["3D", "5D", "10D"]:
        col = f"Fwd_{window}"
        avg_ret = results_df[col].mean()
        med_ret = results_df[col].median()
        win_rate = (results_df[col] > 0).mean() * 100
        print(
            f" [SPY T+{window}] Average: {avg_ret:+.2f}% | Median: {med_ret:+.2f}% | Win Rate: {win_rate:.1f}%"
        )


if __name__ == "__main__":
    run_ppo_forensics()

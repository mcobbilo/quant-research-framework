import pandas as pd
import numpy as np
import sqlite3
import os


def run_strategy_j():
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "market_data.db"
    )
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM core_market_table", conn, index_col="Date")
    df.index = pd.to_datetime(df.index)

    print("[Strategy J] Localizing Top 100 Mathematical Extrema Anomalies...")

    # Identify the Top 100 independent trigger dates
    sorted_df = df.dropna(subset=["VIX_TNX_PPO_7"]).sort_values(
        by="VIX_TNX_PPO_7", ascending=False
    )
    top_dates = []
    used_dates = []
    for idx, row in sorted_df.iterrows():
        if len(top_dates) >= 100:
            break
        conflict = False
        for used in used_dates:
            if abs((idx - used).days) <= 10:
                conflict = True
                break
        if not conflict:
            top_dates.append(idx)
            used_dates.append(idx)

    # Matrix tag the 100 primary entry nodes
    df["Is_Top_100_Trigger"] = False
    df.loc[top_dates, "Is_Top_100_Trigger"] = True

    positions = np.zeros(len(df))  # Baseline 0.0x Cash

    in_trade = False
    trade_count = 0
    days_held = []
    current_hold = 0

    # -----------------------------
    # THE DYNAMIC EXECUTION LOOP
    # -----------------------------
    for i in range(len(df)):
        # Trigger Condition (Entry)
        if df["Is_Top_100_Trigger"].iloc[i]:
            if not in_trade:
                trade_count += 1
                current_hold = 0
            in_trade = True

        # Exit Protocol: Algorithm sells organically when the Volatility PPO momentum completely crosses below zero (mathematical stabilization)
        if in_trade and df["VIX_TNX_PPO_7"].iloc[i] <= 0.0:
            in_trade = False
            if current_hold > 0:
                days_held.append(current_hold)
            current_hold = 0

        if in_trade:
            positions[i] = 1.0  # 100% Invested (Zero Leverage)
            current_hold += 1

    df["Position"] = positions
    df["Raw_Returns"] = df["SPY_CLOSE"].pct_change()

    # Gross Return Calculation
    df["Strategy_Returns"] = df["Position"].shift(1) * df["Raw_Returns"]

    # Standard Execution Friction Models
    trades = df["Position"].diff().abs()
    margin_borrowed = np.maximum(0, df["Position"].shift(1) - 1.0)

    cost_per_trade = 0.001
    daily_margin_rate = 0.05 / 252.0

    df["Net_Returns"] = (
        df["Strategy_Returns"]
        - (trades * cost_per_trade)
        - (margin_borrowed * daily_margin_rate)
    )

    # Add Risk-Free Cash Yield during non-trading days
    risk_free_rate = 0.03 / 252.0
    df["Cash_Yield"] = np.where(df["Position"] == 0, risk_free_rate, 0.0)

    df["SPY_Growth"] = (1 + df["Raw_Returns"].fillna(0)).cumprod()
    df["Strat_Growth"] = (
        1 + df["Net_Returns"].fillna(0) + df["Cash_Yield"].fillna(0)
    ).cumprod()

    spy_final = df["SPY_Growth"].iloc[-1]
    strat_final = df["Strat_Growth"].iloc[-1]

    # Calculate Max Drawdowns
    spy_cummax = df["SPY_Growth"].cummax()
    spy_max_dd = ((df["SPY_Growth"] - spy_cummax) / spy_cummax).min() * 100

    strat_cummax = df["Strat_Growth"].cummax()
    strat_max_dd = ((df["Strat_Growth"] - strat_cummax) / strat_cummax).min() * 100

    avg_hold = sum(days_held) / len(days_held) if days_held else 0

    print("\n=======================================================")
    print("   STRATEGY J: TOP 100 DYNAMIC 'MODEL-EXIT' EXECUTION  ")
    print("=======================================================")
    print(" Entry Protocol:  Buy identically on Top 100 PPO Extrema dates.")
    print(" Exit Protocol:   Sell when VIX_TNX_PPO_7 structurally recovers (<= 0.0)")
    print(
        f" Timeline:        {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
    )
    print(" Starting Cash:   $10,000")
    print(f" Total Trades:    {trade_count}")
    print(f" Average Hold:    {avg_hold:.1f} Trading Days")
    print("")
    print(
        f" [BENCHMARK] SPY Buy/Hold:  ${(10000 * spy_final):,.2f}  Return: {spy_final * 100 - 100:>6.1f}% | Max DD: {spy_max_dd:>6.1f}%"
    )
    print(
        f" [ALGORITHM] Strategy J:    ${(10000 * strat_final):,.2f}  Return: {strat_final * 100 - 100:>6.1f}% | Max DD: {strat_max_dd:>6.1f}%"
    )
    print("=======================================================\n")


if __name__ == "__main__":
    run_strategy_j()

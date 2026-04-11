import pandas as pd
import sqlite3
import os
from zscore_clustering_engine import calculate_rsi, calculate_stochastic, calculate_tsi


def export_full_matrix():
    print("Exporting full Z-Score matrix to CSV for Google Sheets...")
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "market_data.db"
    )
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM core_market_table", conn, index_col="Date")
    df.index = pd.to_datetime(df.index)

    # Analyze multiple timeframes explicitly as requested
    feature_cols = []

    # 1. SPY SMA Percent Displacement [10, 20, 50, 100, 200, 252]
    for w in [10, 20, 50, 100, 200, 252]:
        sma = df["SPY_CLOSE"].rolling(w).mean()
        df[f"SPY_PCT_SMA_{w}"] = (df["SPY_CLOSE"] - sma) / sma
        feature_cols.append(f"SPY_PCT_SMA_{w}")

    # 2. SPY RSI [10, 20, 50, 100, 200]
    for w in [10, 20, 50, 100, 200]:
        df[f"SPY_RSI_{w}"] = calculate_rsi(df["SPY_CLOSE"], w)
        feature_cols.append(f"SPY_RSI_{w}")

    # 3. SPY Stochastics [5, 10, 20, 50, 100, 200, 252]
    for w in [5, 10, 20, 50, 100, 200, 252]:
        df[f"SPY_STOCH_{w}"] = calculate_stochastic(
            df["SPY_HIGH"], df["SPY_LOW"], df["SPY_CLOSE"], w
        )
        feature_cols.append(f"SPY_STOCH_{w}")

    # 4. Normal SPY TSI + Signal Line
    df["SPY_TSI_25_13"] = calculate_tsi(df["SPY_CLOSE"], 25, 13)
    df["SPY_TSI_SIGNAL_13"] = df["SPY_TSI_25_13"].ewm(span=13, adjust=False).mean()
    feature_cols.extend(["SPY_TSI_25_13", "SPY_TSI_SIGNAL_13"])

    # 5. VIX/TNX SMA Percent Displacement [5, 10, 20, 50, 100, 200, 252]
    vix_tnx = df["VIX_CLOSE"] / df["TNX_CLOSE"]
    for w in [5, 10, 20, 50, 100, 200, 252]:
        vix_tnx_sma = vix_tnx.rolling(w).mean()
        df[f"VIX_TNX_PCT_SMA_{w}"] = (vix_tnx - vix_tnx_sma) / vix_tnx_sma
        feature_cols.append(f"VIX_TNX_PCT_SMA_{w}")

    # 6. High Yield Corporate Credit Spreads (BAMLC0A0CM) moving averages
    if "BAMLC0A0CM" in df.columns:
        df["BAMLC0A0CM_SMA_50"] = df["BAMLC0A0CM"].rolling(50).mean()
        df["BAMLC0A0CM_SMA_200"] = df["BAMLC0A0CM"].rolling(200).mean()

    macro_cols = [
        "BAMLC0A0CM",
        "BAMLC0A0CM_SMA_50",
        "BAMLC0A0CM_SMA_200",
        "T10Y2Y",
        "T10YFF",
        "VIX_MOVE_SPREAD_5D",
        "VIX_MOVE_SPREAD_10D",
        "NYA200R",
        "CPC",
        "CPCE",
        "MCO_PRICE",
        "MCO_VOLUME",
        "AD_LINE_PCT_SMA",
        "AD_LINE_5D_ROC",
        "AD_LINE_10D_ROC",
        "AD_LINE_20D_ROC",
        "CPC_5D_ROC",
        "CPCE_5D_ROC",
    ]
    for col in macro_cols:
        if col in df.columns:
            feature_cols.append(col)

    z_score_cols = []
    for col in feature_cols:
        roll_mean = df[col].rolling(252).mean().shift(1)
        roll_std = df[col].rolling(252).std().shift(1)
        z_col = f"Z_{col}"
        df[z_col] = (df[col] - roll_mean) / roll_std
        z_score_cols.append(z_col)

    df = df.dropna(subset=z_score_cols).copy()

    panic_conditions = []
    euphoria_conditions = []

    upside_panic_keywords = ["VIX", "BAMLC0A0CM", "CPC", "CPCE"]

    for col in z_score_cols:
        is_upside = any(keyword in col for keyword in upside_panic_keywords)
        if is_upside:
            panic_conditions.append(df[col] > 2.5)
            euphoria_conditions.append(df[col] < -2.5)
        else:
            panic_conditions.append(df[col] < -2.5)
            euphoria_conditions.append(df[col] > 2.5)

    extreme_matrix = pd.concat(panic_conditions, axis=1)
    df["Panic_Cluster_Score"] = extreme_matrix.sum(axis=1)

    euphoria_matrix = pd.concat(euphoria_conditions, axis=1)
    df["Euphoria_Cluster_Score"] = euphoria_matrix.sum(axis=1)

    # Export to Desktop
    export_path = "/Users/milocobb/Desktop/zscore_feature_matrix.csv"
    df.to_csv(export_path)
    print(f"Successfully exported {len(df.columns)} columns to: {export_path}")


if __name__ == "__main__":
    export_full_matrix()

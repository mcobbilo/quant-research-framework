import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

# Dynamically link the existing feature extractor pipeline
from xgboost_allocation_engine import get_ml_dataframe


def train_crash_oracle():
    print("Initializing Phase 18 Isolation Forest Crash Engine...")
    df = get_ml_dataframe()

    ml_df = df.dropna(subset=["Fwd_20D_Return"]).copy()
    ml_df["Is_Crash"] = (ml_df["Fwd_20D_Return"] <= -0.10).astype(int)

    crash_count = ml_df["Is_Crash"].sum()
    total_count = len(ml_df)
    print(f"Total Rows: {total_count}")
    print(
        f"Total True CRASH states (>10% drop over 20D): {crash_count} ({(crash_count / total_count) * 100:.2f}%)"
    )

    if crash_count == 0:
        print("Mathematical anomaly: No crashes found in dataset. Halting.")
        return

    excluded_cols = [
        "Fwd_20D_Return",
        "Fwd_20D_Max_Drawdown",
        "Fwd_20D_Min_Price",
        "SPY_Daily_Ret",
        "Is_Crash",
    ]
    excluded_cols += [
        c
        for c in ml_df.columns
        if any(x in c for x in ["OPEN", "HIGH", "LOW", "CLOSE", "PRICE", "VOLUME"])
    ]
    excluded_cols += [
        "VVIX",
        "VIX_spot",
        "NYADV",
        "NYDEC",
        "NYUPV",
        "NYDNV",
        "NYADU",
        "AD_LINE",
    ]
    excluded_cols += [
        "RECPROUSM156N",
        "BOGMBASE",
        "WALCL",
        "TREAST",
        "TSIFRGHT",
        "JPNASSETS",
        "ECBASSETSW",
        "DEXJPUS",
        "DEXUSEU",
        "World_CentralBank_BalSh",
        "MonetaryBase_50dMA",
        "FederalReserveRecessionProbability_50dMA",
    ]
    excluded_cols += [
        "FederalReserveTreasuryHoldings_45d%Chg",
        "FederalReserveBalanceSheetSize_45d%Chg",
        "FederalReserveBalanceSheetSize_20d%Chg",
    ]

    features = []
    for c in ml_df.columns:
        if c in excluded_cols:
            continue
        if "SPY_SMA" in c or "VUSTX_SMA" in c or "AD_LINE_SMA" in c:
            continue
        if "Diff" in c and (
            "MonetaryBase" in c
            or "TreasuryHoldings" in c
            or "RecessionProbability" in c
        ):
            continue
        if c in [
            "FederalReserveTreasuryHoldings_20dDiff",
            "MonetaryBase_50dMA_20dDiff",
            "MonetaryBase_50dMA_20dDiff_10dDiff",
            "FederalReserveRecessionProbability_50dMA_5dDiff",
        ]:
            continue
        if "VIX_TNX_SMA" in c or "VIX_TNX_BB" in c or "VIX_TNX_STD" in c:
            continue
        features.append(c)

    ml_df = ml_df.dropna(subset=features)
    X = ml_df[features]
    y = ml_df["Is_Crash"]

    # 80/20 out-of-sample block split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    _y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("Training Unsupervised Isolation Forest...")
    # Contamination set slightly above 2.02% base rate to allow early warnings
    # Setting at 0.05 (5%) based on Council recommendation for safe net
    model = IsolationForest(
        n_estimators=300, contamination=0.05, random_state=42, n_jobs=-1
    )

    # Fit strictly on features; IGNORE target label entirely (Unsupervised)
    model.fit(X_train)

    # Inference: Output is 1 for Geometric Normality, -1 for Extreme Anomaly
    y_pred_if = model.predict(X_test)

    # Filter the pure binary Oracle Flags (If Anomaly = True)
    y_pred_bool = (y_pred_if == -1).astype(int)

    print("\n[ Crash Oracle Integrity Report (Unsupervised) ]")
    print(confusion_matrix(y_test, y_pred_bool))
    print("\n", classification_report(y_test, y_pred_bool))

    out_dates = X_test.index
    spy_close = df.loc[out_dates, "SPY_CLOSE"]

    plt.style.use("dark_background")
    fig, ax1 = plt.subplots(figsize=(15, 8))

    ax1.plot(out_dates, spy_close, color="#999999", linewidth=2, label="SPY Index")
    ax1.set_ylabel("SPY Absolute Level", color="white", fontsize=12)

    # Overlay an aggressive red bar for every day the geometric Oracle yelled panic
    ax2 = ax1.twinx()
    ax2.fill_between(
        out_dates,
        0,
        y_pred_bool,
        color="#ff0055",
        alpha=0.3,
        step="post",
        label="Oracle Isolation Flag",
    )
    ax2.set_yticks([])  # Hide the 1/0 axis to keep it clean
    ax2.set_ylim(0, 5)  # Compress the red bars downward slightly

    plt.title(
        "Phase 18 Isolation Forest Oracle (Unsupervised Structural Detection)",
        color="white",
        fontsize=16,
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2, labels_1 + labels_2, loc="upper left", facecolor="black"
    )

    out_path = "/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/crash_oracle_isolation.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Visual Oracle geometry rendered to: {out_path}")


if __name__ == "__main__":
    train_crash_oracle()

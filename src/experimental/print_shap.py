import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import sys

sys.path.append(
    "/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/src/experimental"
)
from xgboost_allocation_engine import get_ml_dataframe


def read_shap():
    df = get_ml_dataframe()
    ml_df = df.dropna(subset=["Fwd_60D_Return"]).copy()

    excluded_cols = ["Fwd_60D_Return", "SPY_Daily_Ret"]
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

    features = []
    for c in ml_df.columns:
        if c in excluded_cols:
            continue
        if "SPY_SMA" in c or "VUSTX_SMA" in c or "AD_LINE_SMA" in c:
            continue
        features.append(c)

    ml_df = ml_df.dropna(subset=features)
    X = ml_df[features]
    y = ml_df["Fwd_60D_Return"]

    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.DataFrame(
        {"feature": features, "shap_importance": mean_abs_shap}
    )
    feat_importance = feat_importance.sort_values(by="shap_importance", ascending=False)

    print("\n--- EXACT TOP 15 MATHEMATICAL SHAP FEATURES ---")
    for idx, row in feat_importance.head(20).iterrows():
        print(f"{row['feature']}  (Importance: {row['shap_importance']:.6f})")


if __name__ == "__main__":
    read_shap()

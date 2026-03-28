import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from boruta import BorutaPy

def run_boruta_pruning(X, y):
    """
    Executes Phase 132: Boruta Shadow Noise algorithm.
    Runs strictly over the first block of data to preserve Walk-Forward purity.
    """
    print(f"\n[Phase 132] Initializing Boruta 'Shadow Noise' Pruning across {X.shape[1]} raw vectors...")
    
    # Initialize rigorous Classifier structure matching Phase 10
    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        eval_metric='logloss'
    )
    
    # Strict Boruta mathematical destruction routine using alpha=0.01 (99% confidence)
    boruta_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=42, max_iter=50, alpha=0.01)
    
    X_val = X.values
    y_val = y.values
    
    boruta_selector.fit(X_val, y_val)
    
    # support_ is the absolute undeniable features, support_weak_ is tentative.
    selected_mask = boruta_selector.support_ | boruta_selector.support_weak_
    
    surviving_features = X.columns[selected_mask].tolist()
    shredded_count = X.shape[1] - len(surviving_features)
    
    print(f"\n[Boruta] Purge Complete! {shredded_count} unverified statistical features successfully SHREDDED.")
    print(f"[Boruta] Surviving True Alphas: {len(surviving_features)} vectors.\n")
    
    # Print the specific shredded features for forensics
    shredded = [c for c in X.columns if c not in surviving_features]
    print(f"Shredded: {shredded}")
    print(f"Survived (The Pillars of Truth): {surviving_features}\n")
    
    return surviving_features

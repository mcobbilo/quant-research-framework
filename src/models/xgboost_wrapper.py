import numpy as np
import pandas as pd
import xgboost as xgb

class WalkForwardXGBoost:
    def __init__(self):
        print("[XGBoost] Initialized Dynamic Walk-Forward Classifier Pipeline.")

    def train_and_predict(self, history_df, current_vix):
        """
        Dynamically trains an XGBoost classifier on an expanding historical window.
        Returns the specific probability of SPY achieving a positive 21-day forward return.
        """
        df = history_df.copy()
        
        # Synthesize standard quantitative features
        df['Mom_5'] = df['Returns'].rolling(5).sum()
        df['Mom_21'] = df['Returns'].rolling(21).sum()
        df['Vol_21'] = df['Returns'].rolling(21).std() * np.sqrt(252)
        
        # Target: Positive Forward 21-Day Return
        # We negatively shift the target so that each historical day maps to what happened 21 days later!
        df['Forward_21d_Return'] = df['SPY'].shift(-21) / df['SPY'] - 1.0
        df['Target'] = (df['Forward_21d_Return'] > 0).astype(int)
        
        # Crucial Filter to prevent Look-ahead bias:
        # The final 21 days of history have NaNs in the Target column because their 21-day forward returns haven't happened yet!
        # By dropping NaNs, we strictly prune out the future and only train on concretely closed time-windows.
        train_df = df.dropna().copy()
        
        if len(train_df) < 50:
            # Not enough data to map a tree, return neutral edge
            return 0.50
            
        features = ['VIX', 'Mom_5', 'Mom_21', 'Vol_21']
        
        X_train = train_df[features]
        y_train = train_df['Target']
        
        # Initialize shallow bounds to prevent tree overfitting
        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1
        )
        
        # Iterative re-training per timestep mapping
        model.fit(X_train, y_train)
        
        # Generate Out-Of-Sample prediction for the Current Day Step
        current_features = pd.DataFrame({
            'VIX': [current_vix],
            'Mom_5': [df['Returns'].iloc[-5:].sum()],
            'Mom_21': [df['Returns'].iloc[-21:].sum()],
            'Vol_21': [df['Returns'].iloc[-21:].std() * np.sqrt(252)]
        })
        
        # XGBoost outputs probability scalar for Target == 1 class
        prob_up = model.predict_proba(current_features)[0][1]
        
        return prob_up

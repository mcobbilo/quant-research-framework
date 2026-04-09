import pandas as pd
import numpy as np

def generate_degenerate_mocks():
    """
    Generates a suite of 'Adversarial' DataFrames to stress-test 
    new quantitative strategies for stability. Inspired by codedb.
    """
    assets = ['SPY', 'IWM', 'RSP', 'GLD', 'CL', 'HG', 'VUSTX']
    cols = ['DATE', 'target_returns'] + [a + '_CLOSE' for a in assets]
    
    # 1. Empty DataFrame (with schema)
    df_empty = pd.DataFrame(columns=cols)
    
    # 2. Single-Row DataFrame
    single_data = {c: [100.0] for c in cols}
    single_data['DATE'] = ['2026-01-01']
    single_data['target_returns'] = [0.02]
    df_single = pd.DataFrame(single_data)
    
    # 3. All-NaN DataFrame
    nan_data = {c: [np.nan]*5 for c in cols}
    nan_data['DATE'] = pd.date_range('2026-01-01', periods=5)
    df_nans = pd.DataFrame(nan_data)
    
    # 4. Zero-Variance (Static Price)
    static_data = {c: [100.0]*10 for c in cols}
    static_data['DATE'] = pd.date_range('2026-01-01', periods=10)
    static_data['target_returns'] = [0.0]*10
    df_static = pd.DataFrame(static_data)

    return {
        "Empty": df_empty,
        "SingleRow": df_single,
        "AllNaN": df_nans,
        "StaticPrice": df_static
    }

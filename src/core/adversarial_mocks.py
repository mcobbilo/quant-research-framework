import os
import sqlite3
import pandas as pd
import numpy as np

def get_base_data():
    """Fetches the actual historical data for synthetic adversarial generation."""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "market_data.db")
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM core_market_table ORDER BY Date ASC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

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

def generate_adversarial_paths():
    """
    Synthesizes mathematically rigorous out-of-sample stress environments based on real historical data.
    """
    base_df = get_base_data()
    if base_df.empty:
        print("[WARNING] Could not load base market data, falling back to degenerate mocks.")
        return generate_degenerate_mocks()
        
    synthetics = {}

    # Identify numeric columns
    numeric_cols = base_df.select_dtypes(include=[np.number]).columns

    # 1. Adversarial_Block_Bootstrap (Block size 21 days)
    block_size = 21
    n_blocks = len(base_df) // block_size
    if n_blocks > 0:
        blocks = [base_df.iloc[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
        np.random.seed(42) # Deterministic for consistent testing
        shuffled_idx = np.random.permutation(len(blocks))
        bootstrapped_df = pd.concat([blocks[i] for i in shuffled_idx]).reset_index(drop=True)
        # Keep Date strictly monotonically increasing
        bootstrapped_df['Date'] = base_df['Date'][:len(bootstrapped_df)].reset_index(drop=True)
        synthetics['Adversarial_Block_Bootstrap'] = bootstrapped_df

    # 2. Adversarial_Noise_Injected
    noise_df = base_df.copy()
    np.random.seed(42)
    for col in numeric_cols:
        std_val = noise_df[col].std()
        if pd.notnull(std_val) and std_val > 0:
            # Inject 0.5-sigma scaled Gaussian noise
            noise = np.random.normal(0, std_val * 0.5, size=len(noise_df))
            noise_df[col] = noise_df[col] + noise
    synthetics['Adversarial_Noise_Injected'] = noise_df

    # 3. Adversarial_Fat_Tail_Jumps
    fat_tail_df = base_df.copy()
    np.random.seed(42)
    # Pick 5 random epochs (days) to inject a correlated flash crash
    num_jumps = 5
    if len(fat_tail_df) > num_jumps:
        jump_indices = np.random.choice(fat_tail_df.index[1:-1], num_jumps, replace=False)
        for idx in jump_indices:
            # Simulate a 10% SPY drop, 200% VIX spike, etc.
            if 'SPY_CLOSE' in fat_tail_df.columns:
                fat_tail_df.at[idx, 'SPY_CLOSE'] *= 0.90
            if 'VIX_CLOSE' in fat_tail_df.columns:
                fat_tail_df.at[idx, 'VIX_CLOSE'] *= 3.0
            if 'IWM_CLOSE' in fat_tail_df.columns:
                fat_tail_df.at[idx, 'IWM_CLOSE'] *= 0.85
            if 'GLD_CLOSE' in fat_tail_df.columns:
                fat_tail_df.at[idx, 'GLD_CLOSE'] *= 1.05 # Flight to safety
            if 'VUSTX_CLOSE' in fat_tail_df.columns:
                fat_tail_df.at[idx, 'VUSTX_CLOSE'] *= 1.08 # Treasury rally
    synthetics['Adversarial_Fat_Tail_Jumps'] = fat_tail_df
    
    return synthetics

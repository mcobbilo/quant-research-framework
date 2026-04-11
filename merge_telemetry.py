import pandas as pd
import os

def main():
    parquet_path = '/Users/milocobb/tft_model/clean_aligned_features_27yr.parquet'
    csv_path = '/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/druckenmiller_features.csv'
    
    print(f"Loading master matrix from: {parquet_path}")
    master_df = pd.read_parquet(parquet_path)
    
    # Store old shape
    old_shape = master_df.shape
    
    print(f"Loading telemetry from: {csv_path}")
    telemetry_df = pd.read_csv(csv_path)
    telemetry_df['Date'] = pd.to_datetime(telemetry_df['Date'])
    
    # Master matrix presumably has Date as index. Let's merge properly.
    if master_df.index.name == 'Date':
        master_df = master_df.reset_index()
    
    # Merge left on Date
    # Drop existing macros to prevent _x and _y suffix collisions on re-runs
    new_cols = ['taylor_policy_spread', 'us2y_ffr_divergence', 'net_liquidity_momentum', 'hy_spread_velocity', 't_bill_drain_ratio']
    existing_cols = [c for c in new_cols if c in master_df.columns]
    if existing_cols:
        master_df.drop(columns=existing_cols, inplace=True)
        
    print("Fusing matrices...")
    merged_df = pd.merge(master_df, telemetry_df, on='Date', how='left')
    
    # Handle NaNs from the merge
    # We forward fill because macro variables persist until the next reading.
    # 💥 RED TEAM HARDENING: REMOVED .bfill() 💥
    # Backfilling projects the future into the void. We replace missing historical boundaries with 0.
    merged_df[new_cols] = merged_df[new_cols].ffill().fillna(0)
    
    # Re-set index
    merged_df.set_index('Date', inplace=True)
    
    print(f"Old Matrix Shape: {old_shape}")
    print(f"New Matrix Shape: {merged_df.shape}")
    
    print("Overwriting master parquet file...")
    merged_df.to_parquet(parquet_path)
    print("✅ Integration successful.")

if __name__ == '__main__':
    main()

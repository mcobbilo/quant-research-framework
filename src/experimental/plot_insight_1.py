import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt

def generate_chart():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_data.db')
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT Date, SPY_CLOSE, T10YFF FROM core_market_table', conn, index_col='Date')
    df.index = pd.to_datetime(df.index)
    
    # Drop NaNs
    df = df.dropna(subset=['T10YFF', 'SPY_CLOSE'])

    # Find the absolute minimum T10YFF date (Maximum Inversion)
    min_date = df['T10YFF'].idxmin()
    print(f"Absolute Minimum T10YFF found on: {min_date}")
    
    # Create window: 120 trading days before, 150 trading days after
    idx_min = df.index.get_loc(min_date)
    start_idx = max(0, idx_min - 120)
    end_idx = min(len(df) - 1, idx_min + 150)
    
    window_df = df.iloc[start_idx:end_idx]
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1], sharex=True)
    
    # Plot SPY
    ax1.plot(window_df.index, window_df['SPY_CLOSE'], color='#1f77b4', linewidth=2, label='SPY Price')
    ax1.axvline(x=min_date, color='red', linestyle='--', linewidth=2.5, label='Maximum Yield Inversion Trigger')
    
    # Find the exact 60-trading-day forward index relative to the sliced window
    local_idx_min = window_df.index.get_loc(min_date)
    if local_idx_min + 60 < len(window_df):
        day_60_actual = window_df.index[local_idx_min + 60]
        ax1.axvline(x=day_60_actual, color='green', linestyle=':', linewidth=2.5, label='60 Trading-Day Forward Horizon')
        
        # Calculate precise return
        price_entry = window_df.iloc[local_idx_min]['SPY_CLOSE']
        price_exit = window_df.iloc[local_idx_min + 60]['SPY_CLOSE']
        ret = (price_exit / price_entry) - 1
        
        # Annotate
        ax1.annotate(f"Forward 60D Return:\n+{ret*100:.2f}%", 
                     xy=(day_60_actual, price_exit), 
                     xytext=(day_60_actual + pd.Timedelta(days=5), price_exit),
                     bbox=dict(boxstyle="round,pad=0.4", fc="#d4edda", ec="green", lw=2),
                     fontsize=11, fontweight='bold', color='darkgreen')

    ax1.set_title('SPY Price Action Following Absolute T10YFF Maximum Inversion', fontsize=15, fontweight='bold')
    ax1.set_ylabel('SPY Price ($)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)

    # Plot T10YFF
    ax2.plot(window_df.index, window_df['T10YFF'], color='#d62728', linewidth=2, label='T10YFF (10Y Yield - Fed Funds)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, label='Inversion Parity Limit (0%)')
    ax2.axvline(x=min_date, color='red', linestyle='--', linewidth=2.5)
    
    # T10YFF Minimum Annotation
    min_val = window_df.loc[min_date, 'T10YFF']
    ax2.annotate(f"Max Inversion Penalty: {min_val:.2f}%", 
                 xy=(min_date, min_val), 
                 xytext=(min_date + pd.Timedelta(days=5), min_val + 0.5),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f8d7da", ec="red", lw=1),
                 fontsize=10, fontweight='bold', color='darkred')
                 
    ax2.set_title('T10YFF Yield Spread Dynamics', fontsize=13)
    ax2.set_ylabel('Spread (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    
    out_path = '/Users/milocobb/.gemini/antigravity/brain/86f8d6d6-545f-43de-8268-7b50b6d1c47a/insight1_t10yff.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved chart to {out_path}")

if __name__ == '__main__':
    generate_chart()

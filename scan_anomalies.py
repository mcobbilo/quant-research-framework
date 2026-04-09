import pandas as pd
import numpy as np

df = pd.read_csv('market_data_inspection.csv')
num_rows = len(df)

suspicious = []

for c in df.columns:
    if c == 'Date': continue
    
    col_data = df[c]
    
    # Check for NaNs
    nans = col_data.isna().sum()
    if nans > 0:
        suspicious.append((c, f"Contains {nans} NaNs ({nans/num_rows*100:.2f}%)"))
        continue
        
    # Check for Infinite max/min
    if np.isinf(col_data).any():
        suspicious.append((c, "Contains infinite (inf) values"))
        continue
        
    # Check for exact zero columns
    if (col_data == 0).all():
        suspicious.append((c, "Column is entirely 0.0 (Dead data)"))
        continue
        
    # Check for highly constant data (excluding binary signals)
    val_counts = col_data.value_counts(normalize=True)
    most_common_pct = val_counts.iloc[0]
    unique_vals = col_data.nunique()
    
    # Is it a binary signal?
    is_binary = unique_vals == 2 and set(col_data.unique()).issubset({0, 1, 0.0, 1.0})
    
    if not is_binary and most_common_pct > 0.5:
        most_common_val = val_counts.index[0]
        # Allow it if it's an integer sequence or something, but financial float data shouldn't be constant >50%
        suspicious.append((c, f"Highly invariant: {most_common_pct*100:.1f}% of data is exactly {most_common_val}"))
        
    # Check for near-zero columns that aren't binary
    zero_pct = (col_data == 0).mean()
    if not is_binary and zero_pct > 0.2:
        suspicious.append((c, f"High sparsity: {zero_pct*100:.1f}% of data is exactly 0.0"))
        
print("SUSPICIOUS COLUMNS:")
for col, reason in suspicious:
    print(f"- {col}: {reason}")
    

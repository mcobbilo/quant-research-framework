import pandas as pd
import sqlite3
import os
import sys

# Change to src directory to grab the engine properly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'experimental'))
from xgboost_allocation_engine import get_ml_dataframe

df = get_ml_dataframe()
excluded_cols = ['Fwd_20D_Return', 'SPY_Daily_Ret']
excluded_cols += [c for c in df.columns if any(x in c for x in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'PRICE', 'VOLUME'])]
excluded_cols += ['VVIX', 'VIX_spot', 'NYADV', 'NYDEC', 'NYUPV', 'NYDNV', 'NYADU', 'AD_LINE']
excluded_cols += ['RECPROUSM156N', 'BOGMBASE', 'WALCL', 'TREAST', 'TSIFRGHT', 'JPNASSETS', 
                  'ECBASSETSW', 'DEXJPUS', 'DEXUSEU', 'World_CentralBank_BalSh', 
                  'MonetaryBase_50dMA', 'FederalReserveRecessionProbability_50dMA']
excluded_cols += ['FederalReserveTreasuryHoldings_45d%Chg', 'FederalReserveBalanceSheetSize_45d%Chg', 'FederalReserveBalanceSheetSize_20d%Chg']

features = []
for c in df.columns:
    if c in excluded_cols: continue
    if 'SPY_SMA' in c or 'VUSTX_SMA' in c or 'AD_LINE_SMA' in c: continue
    if 'Diff' in c and ('MonetaryBase' in c or 'TreasuryHoldings' in c or 'RecessionProbability' in c): continue
    if c in ['FederalReserveTreasuryHoldings_20dDiff', 'MonetaryBase_50dMA_20dDiff', 'MonetaryBase_50dMA_20dDiff_10dDiff', 'FederalReserveRecessionProbability_50dMA_5dDiff']: continue
    if 'VIX_TNX_SMA' in c or 'VIX_TNX_BB' in c or 'VIX_TNX_STD' in c: continue
    features.append(c)

print("TOTAL FEATURES:", len(features))
for i, f in enumerate(features):
    print(f"{i}: {f}")

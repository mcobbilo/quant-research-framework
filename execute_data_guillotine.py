import sqlite3
import pandas as pd
import os

db_path = os.path.join(os.path.dirname(__file__), "src", "data", "market_data.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 1. Row Cut: Delete all rows prior to 2008-01-03
try:
    cursor.execute("DELETE FROM core_market_table WHERE Date < '2008-01-03 00:00:00'")
    deleted_rows = cursor.rowcount
    print(f"Successfully deleted {deleted_rows} rows from prior to Jan 3, 2008.")
except Exception as e:
    print(f"Failed to delete rows: {e}")

# 2. Column Cut: Drop DXY_VOLUME and VIX3M_VOLUME (if it still exists)
cols_to_drop = ['DXY_VOLUME', 'VIX3M_VOLUME']
for col in cols_to_drop:
    try:
        cursor.execute(f"ALTER TABLE core_market_table DROP COLUMN {col}")
        print(f"Successfully dropped {col}")
    except Exception as e:
        print(f"Note on {col}: {e}")

conn.commit()

# 3. Export new cleaned CSV for user inspection
print("Refreshing market_data_inspection.csv...")
df = pd.read_sql_query("SELECT * FROM core_market_table ORDER BY Date ASC", conn)
df.to_csv('market_data_inspection.csv', index=False)
print(f"Data re-exported: {len(df)} rows and {len(df.columns)} columns remaining.")
conn.close()

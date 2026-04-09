import sqlite3
import pandas as pd
import os

db_path = os.path.join(os.path.dirname(__file__), "src", "data", "market_data.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Column Cut: Drop remaining absolute zero volume columns
cols_to_drop = ['VVIX_VOLUME', 'SKEW_VOLUME', 'JPY_VOLUME']
for col in cols_to_drop:
    try:
        cursor.execute(f"ALTER TABLE core_market_table DROP COLUMN {col}")
        print(f"Successfully dropped {col}")
    except Exception as e:
        print(f"Note on {col}: {e}")

conn.commit()

print("Refreshing market_data_inspection.csv...")
df = pd.read_sql_query("SELECT * FROM core_market_table ORDER BY Date ASC", conn)
df.to_csv('market_data_inspection.csv', index=False)
print(f"Data re-exported: {len(df)} rows and {len(df.columns)} columns remaining.")
conn.close()

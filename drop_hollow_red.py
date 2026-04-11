import sqlite3
import pandas as pd
import os

db_path = os.path.join(os.path.dirname(__file__), "src", "data", "market_data.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE core_market_table DROP COLUMN SPY_Hollow_Red")
    print("Successfully dropped SPY_Hollow_Red")
except Exception as e:
    print(f"Note: {e}")

conn.commit()
df = pd.read_sql_query("SELECT * FROM core_market_table ORDER BY Date ASC", conn)
df.to_csv("market_data_inspection.csv", index=False)
print(f"Data re-exported: {len(df.columns)} columns remaining.")
conn.close()

import sqlite3
import pandas as pd
import os

db_path = os.path.join(os.path.dirname(__file__), "src", "data", "market_data.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE core_market_table DROP COLUMN CVR3_BUY_SIGNAL")
    print("Successfully dropped CVR3_BUY_SIGNAL")
except Exception as e:
    print(f"Note on cvr3_buy: {e}")

try:
    cursor.execute("ALTER TABLE core_market_table DROP COLUMN CVR3_SELL_SIGNAL")
    print("Successfully dropped CVR3_SELL_SIGNAL")
except Exception as e:
    print(f"Note on cvr3_sell: {e}")

conn.commit()

df = pd.read_sql_query("SELECT * FROM core_market_table ORDER BY Date ASC", conn)
df.to_csv("market_data_inspection.csv", index=False)
print(f"Data re-exported: {len(df.columns)} columns remaining.")
conn.close()

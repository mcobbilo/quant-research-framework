import sqlite3
import pandas as pd
import os

db_path = os.path.join(os.path.dirname(__file__), "src", "data", "market_data.db")
conn = sqlite3.connect(db_path)

print("Extracting full market datatable...")
# Fetch everything, ensuring Date is at the start
df = pd.read_sql_query("SELECT * FROM core_market_table ORDER BY Date ASC", conn)
conn.close()

output_file = "market_data_inspection.csv"
df.to_csv(output_file, index=False)
print(f"Exported {len(df)} rows and {len(df.columns)} columns to {output_file}")

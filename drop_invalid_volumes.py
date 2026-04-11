import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), "src", "data", "market_data.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

columns_to_drop = [
    "VIX_VOLUME",
    "VIX3M_VOLUME",
    "VIX6M_VOLUME",
    "GC_VOLUME",
    "HG_VOLUME",
    "VUSTX_VOLUME",
    "TNX_VOLUME",
    "MOVE_VOLUME",
]

for col in columns_to_drop:
    try:
        # SQLite 3.35.0+ supports ALTER TABLE DROP COLUMN
        cursor.execute(f"ALTER TABLE core_market_table DROP COLUMN {col};")
        print(f"Successfully dropped {col}")
    except Exception as e:
        print(f"Failed to drop {col}: {e}")

conn.commit()
conn.close()

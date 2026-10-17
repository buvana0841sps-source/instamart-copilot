
import sqlite3
from pathlib import Path
import pandas as pd

# Define file locations
DATA_FOLDER = Path("data")
CLEAN_FOLDER = DATA_FOLDER / "clean"
DB_PATH = DATA_FOLDER / "instamart.db"

# Read the cleaned CSV files
customers = pd.read_csv(CLEAN_FOLDER / "customers.csv")
products = pd.read_csv(CLEAN_FOLDER / "products.csv")
orders = pd.read_csv(CLEAN_FOLDER / "orders.csv")

# Keep order timestamps as text in the required format
orders["order_ts"] = orders["order_ts"].astype(str)

# Connect to SQLite and create the database
with sqlite3.connect(DB_PATH) as conn:
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)

    # Check the number of rows in each table
    print("DATABASE CHECK")
    for table in ["customers", "products", "orders"]:
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS total FROM {table}", conn
        ).iloc[0]["total"]
        print(f"{table}: {count} rows")

    # Check that all three required tables exist
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        conn
    )["name"].tolist()

    assert all(
        name in tables for name in ["customers", "products", "orders"]
    )

    # Verify the required order timestamp format
    bad_timestamps = pd.read_sql_query(
        """
        SELECT COUNT(*) AS total
        FROM orders
        WHERE length(order_ts) != 19
           OR substr(order_ts, 5, 1) != '-'
           OR substr(order_ts, 8, 1) != '-'
           OR substr(order_ts, 11, 1) != ' '
        """,
        conn
    ).iloc[0]["total"]

    assert bad_timestamps == 0, "Some order timestamps have an incorrect format."

    assert len(customers) == 1200
    assert len(products) == 40
    assert len(orders) == 12309

print(f"\nDatabase created: {DB_PATH}")
print("Database validation passed.")

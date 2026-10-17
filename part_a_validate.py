
import sqlite3
from pathlib import Path

import pandas as pd

# File locations
DATA = Path("data")
CLEAN = DATA / "clean"
DB_PATH = DATA / "instamart.db"

# Load clean CSV files
customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

# Convert numeric columns for reliable checking
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="raise"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)
customers["age"] = pd.to_numeric(
    customers["age"], errors="raise"
)

# Separate completed and cancelled orders
completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

# 1. Check the number of records
assert len(customers) == 1200
assert len(products) == 40
assert len(orders) == 12309

# 2. Check unique order IDs
assert orders["order_id"].is_unique

# 3. Check that no pending orders remain
assert not orders["status"].eq("pending").any()

# 4. Check order status counts
assert len(completed) == 11110
assert len(cancelled) == 1199

# 5. Check customer cities
expected_cities = {
    "Pune": 227,
    "Delhi": 210,
    "Bengaluru": 197,
    "Chennai": 191,
    "Hyderabad": 191,
    "Mumbai": 184,
}
assert customers["city"].value_counts().to_dict() == expected_cities

# 6. Check gender counts
expected_genders = {
    "female": 610,
    "male": 560,
    "unknown": 30,
}
assert customers["gender"].value_counts().to_dict() == expected_genders

# 7. Check ages
assert customers["age"].notna().all()
assert customers["age"].between(18, 60).all()

# 8. Check order amounts
assert orders["order_amount"].notna().all()
assert round(completed["order_amount"].sum(), 2) == 2998668.80

# 9. Check delivery times
assert completed["delivery_minutes"].notna().all()
assert cancelled["delivery_minutes"].isna().all()

# 10. Check that every order has a final status
assert set(orders["status"].unique()) == {"completed", "cancelled"}

# 11. Check SQLite database and required tables
assert DB_PATH.exists(), "SQLite database was not found."

with sqlite3.connect(DB_PATH) as conn:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        conn
    )["name"].tolist()

    assert all(
        table in tables
        for table in ["customers", "products", "orders"]
    )

    db_customers = pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM customers", conn
    ).iloc[0]["n"]

    db_products = pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM products", conn
    ).iloc[0]["n"]

    db_orders = pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM orders", conn
    ).iloc[0]["n"]

    assert db_customers == 1200
    assert db_products == 40
    assert db_orders == 12309

# Print the required success message only after all checks pass
print("ALL CHECKPOINTS PASSED")

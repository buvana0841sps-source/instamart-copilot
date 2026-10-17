
import pandas as pd
from pathlib import Path

# Project folders
RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. LOAD THE ORIGINAL DATA
# --------------------------------------------------
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")
orders = pd.read_csv(RAW / "orders.csv")

print("Original data loaded successfully.")

# --------------------------------------------------
# 2. CLEAN CUSTOMERS
# --------------------------------------------------

# Standardise city names
customers["city"] = (
    customers["city"].astype(str).str.strip().str.title()
)

# Standardise gender
gender_map = {
    "male": "male",
    "m": "male",
    "female": "female",
    "f": "female"
}

customers["gender"] = (
    customers["gender"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .map(gender_map)
    .fillna("unknown")
)

# Convert age to numeric
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

# Treat ages outside 10-100 as missing
invalid_age = (
    (customers["age"] < 10)
    | (customers["age"] > 100)
)
customers.loc[invalid_age, "age"] = float("nan")

# Fill missing ages with the median age of their city
city_medians = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_medians)

# Convert signup date to datetime
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="raise"
)

print("\nCUSTOMER CLEANING")
print("City counts:")
print(customers["city"].value_counts().sort_index())

print("\nGender counts:")
print(customers["gender"].value_counts().sort_index())

print("Missing ages:", customers["age"].isna().sum())
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())

# --------------------------------------------------
# 3. CLEAN PRODUCTS
# --------------------------------------------------

products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="raise"
)

# --------------------------------------------------
# 4. CLEAN ORDERS
# --------------------------------------------------

# Convert order timestamp to datetime
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="raise"
)

# Remove exact duplicate rows
before_duplicates = len(orders)
orders = orders.drop_duplicates().copy()
exact_duplicates_removed = before_duplicates - len(orders)

# Keep the latest record for each order_id
orders = orders.sort_values(
    "order_ts", kind="stable"
)
before_latest = len(orders)
orders = orders.drop_duplicates(
    subset=["order_id"], keep="last"
).copy()
older_records_removed = before_latest - len(orders)

# Convert numeric columns
orders["quantity"] = pd.to_numeric(
    orders["quantity"], errors="raise"
)
orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="raise"
)
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# Join product prices and categories
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

# Fill missing or unparseable amounts using the given formula
missing_amount = orders["order_amount"].isna()
orders.loc[missing_amount, "order_amount"] = (
    orders.loc[missing_amount, "price_inr"]
    * orders.loc[missing_amount, "quantity"]
    * (
        1
        - orders.loc[missing_amount, "discount_pct"] / 100
    )
).round(2)

# Remove impossible negative delivery times
negative_delivery = orders["delivery_minutes"] < 0
negative_delivery_count = int(negative_delivery.sum())
orders.loc[negative_delivery, "delivery_minutes"] = float("nan")

# Fill missing delivery times for completed orders only.
# The median is calculated within each product category.
completed_mask = orders["status"].eq("completed")

category_medians = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"]
    .fillna(category_medians)
)

# Cancelled orders must have no delivery time.
orders.loc[
    orders["status"].eq("cancelled"), "delivery_minutes"
] = float("nan")

# Keep the required clean order columns
orders = orders[
    [
        "order_id", "customer_id", "product_id", "quantity",
        "order_ts", "status", "discount_pct",
        "order_amount", "delivery_minutes"
    ]
].copy()

# --------------------------------------------------
# 5. SAVE CLEAN CSV FILES
# --------------------------------------------------

# Store timestamps in the required text format
orders["order_ts"] = orders["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)
customers["signup_date"] = customers["signup_date"].dt.strftime(
    "%Y-%m-%d"
)

customers.to_csv(CLEAN / "customers.csv", index=False)
products.to_csv(CLEAN / "products.csv", index=False)
orders.to_csv(CLEAN / "orders.csv", index=False)

# --------------------------------------------------
# 6. VALIDATE IMPORTANT CHECKPOINTS
# --------------------------------------------------

print("\nORDER CLEANING")
print("Exact duplicate rows removed:", exact_duplicates_removed)
print("Older records removed:", older_records_removed)
print("Negative delivery times changed to missing:", negative_delivery_count)
print("Clean order rows:", len(orders))
print("\nOrder status counts:")
print(orders["status"].value_counts().sort_index())

completed = orders["status"].eq("completed")
cancelled = orders["status"].eq("cancelled")

print(
    "Missing order amounts:",
    orders["order_amount"].isna().sum()
)
print(
    "Completed orders with missing delivery time:",
    orders.loc[completed, "delivery_minutes"].isna().sum()
)
print(
    "Cancelled orders with missing delivery time:",
    orders.loc[cancelled, "delivery_minutes"].isna().sum()
)
print(
    "Completed order revenue (INR):",
    f"{orders.loc[completed, 'order_amount'].sum():,.2f}"
)

# Stop with an error if an essential checkpoint fails.
assert len(customers) == 1200
assert len(products) == 40
assert len(orders) == 12309
assert orders["order_id"].is_unique
assert not orders["status"].eq("pending").any()
assert orders["order_amount"].notna().all()
assert orders.loc[completed, "delivery_minutes"].notna().all()
assert orders.loc[cancelled, "delivery_minutes"].isna().all()
assert round(orders.loc[completed, "order_amount"].sum(), 2) == 2998668.80
assert customers["age"].notna().all()
assert customers["city"].nunique() == 6
assert customers["gender"].value_counts().to_dict() == {
    "female": 610, "male": 560, "unknown": 30
}

print("\nClean CSV files saved in data/clean/")
print("Cleaning validation passed.")


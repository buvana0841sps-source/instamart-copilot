
import pandas as pd
from pathlib import Path

# Find the raw data folder
DATA_FOLDER = Path("data/raw")

# Load the original CSV files
customers = pd.read_csv(DATA_FOLDER / "customers.csv")
products = pd.read_csv(DATA_FOLDER / "products.csv")
orders = pd.read_csv(DATA_FOLDER / "orders.csv")

# Store the tables together so we can inspect them
tables = {
    "CUSTOMERS": customers,
    "PRODUCTS": products,
    "ORDERS": orders
}

# Display the shape, data types and missing values
for name, df in tables.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Shape (rows, columns):", df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values by column:")
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)

    missing_report = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percent
    })

    print(missing_report.to_string())

# Audit the raw orders
print("\n" + "=" * 60)
print("RAW ORDERS AUDIT")
print("=" * 60)

numeric_amount = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

numeric_delivery = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

print("Raw order rows:", len(orders))
print("Exact duplicate rows:", orders.duplicated().sum())
print("Pending rows:", (orders["status"] == "pending").sum())
print("Distinct order IDs:", orders["order_id"].nunique())
print("Non-numeric order amounts:", numeric_amount.isna().sum())
print("Negative delivery times:", (numeric_delivery < 0).sum())
print(
    "Blank delivery times:",
    orders["delivery_minutes"].isna().sum()
    + (orders["delivery_minutes"].astype(str).str.strip() == "").sum()
)

# Audit the raw customers
print("\n" + "=" * 60)
print("RAW CUSTOMERS AUDIT")
print("=" * 60)

numeric_age = pd.to_numeric(customers["age"], errors="coerce")
gender = customers["gender"].fillna("").astype(str).str.strip()
city = customers["city"].fillna("").astype(str)

print("Blank ages:", customers["age"].isna().sum()
      + (customers["age"].astype(str).str.strip() == "").sum())
print(
    "Ages outside 10-100:",
    ((numeric_age < 10) | (numeric_age > 100)).sum()
)
print("Distinct raw city spellings:", city.nunique())
print(
    "Distinct non-blank raw gender spellings:",
    gender[gender != ""].nunique()
)
print("Blank genders:", (gender == "").sum())

print("\nRaw data audit completed.")

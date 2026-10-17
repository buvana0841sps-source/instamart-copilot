
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. LOAD CLEAN DATA
# --------------------------------------------------
DATA = Path("data/clean")

customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv")

# Keep completed orders only for revenue analysis
completed = orders.loc[
    orders["status"].eq("completed")
].copy()

completed["order_amount"] = pd.to_numeric(
    completed["order_amount"], errors="raise"
)
completed["discount_pct"] = pd.to_numeric(
    completed["discount_pct"], errors="raise"
)
completed["quantity"] = pd.to_numeric(
    completed["quantity"], errors="raise"
)
completed["delivery_minutes"] = pd.to_numeric(
    completed["delivery_minutes"], errors="raise"
)

# Join city and category information
analysis = (
    completed
    .merge(
        customers[["customer_id", "city"]],
        on="customer_id",
        how="left",
        validate="many_to_one"
    )
    .merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )
)

# --------------------------------------------------
# 2. PANDAS PIVOT TABLE CROSS-CHECK OF SQL Q2
# --------------------------------------------------
print("=" * 65)
print("PANDAS PIVOT TABLE: REVENUE BY CITY AND CATEGORY")
print("=" * 65)

pivot = pd.pivot_table(
    analysis,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="All"
)

print(pivot.round(2).to_string())

pivot_total = pivot.loc["All", "All"]
completed_total = analysis["order_amount"].sum()

print("\nPivot-table grand total (INR):", round(pivot_total, 2))
print("Completed-order total (INR):", round(completed_total, 2))

assert round(pivot_total, 2) == 2998668.80
assert round(completed_total, 2) == 2998668.80

# --------------------------------------------------
# 3. CITY-LEVEL SUMMARY USING NAMED AGGREGATION
# --------------------------------------------------
print("\n" + "=" * 65)
print("COMPLETED ORDER STATISTICS BY CITY")
print("=" * 65)

city_summary = analysis.groupby("city").agg(
    total_revenue_inr=("order_amount", "sum"),
    mean_order_amount_inr=("order_amount", "mean"),
    completed_order_count=("order_amount", "count")
).sort_index()

print(city_summary.round(2).to_string())

# --------------------------------------------------
# 4. DISTRIBUTION AND OUTLIER ANALYSIS
# --------------------------------------------------
print("\n" + "=" * 65)
print("COMPLETED ORDER AMOUNT DISTRIBUTION")
print("=" * 65)

amount = analysis["order_amount"]

mean_amount = amount.mean()
median_amount = amount.median()
skewness = amount.skew()

q1 = amount.quantile(0.25)
q2 = amount.quantile(0.50)
q3 = amount.quantile(0.75)

iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

below_outliers = int((amount < lower_fence).sum())
above_outliers = int((amount > upper_fence).sum())

print(f"Mean (INR): {mean_amount:.2f}")
print(f"Median (INR): {median_amount:.2f}")
print(f"Skewness: {skewness:.4f}")
print(f"Q1 (INR): {q1:.2f}")
print(f"Q2 / Median (INR): {q2:.2f}")
print(f"Q3 (INR): {q3:.2f}")
print(f"IQR (INR): {iqr:.2f}")
print(f"Lower 1.5 x IQR fence (INR): {lower_fence:.2f}")
print(f"Upper 1.5 x IQR fence (INR): {upper_fence:.2f}")
print("Outliers below lower fence:", below_outliers)
print("Outliers above upper fence:", above_outliers)

# --------------------------------------------------
# 5. CORRELATION ANALYSIS
# --------------------------------------------------
print("\n" + "=" * 65)
print("CORRELATION MATRIX")
print("=" * 65)

correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation = analysis[correlation_columns].corr()

print(correlation.round(4).to_string())

# Find the pair with the largest absolute correlation,
# excluding the diagonal and duplicate pairs.
abs_corr = correlation.abs()
upper_triangle = abs_corr.where(
    pd.DataFrame(
        [[i < j for j in range(len(abs_corr.columns))]
         for i in range(len(abs_corr.columns))],
        index=abs_corr.index,
        columns=abs_corr.columns
    )
)

largest_pair = upper_triangle.stack().idxmax()
largest_value = correlation.loc[largest_pair[0], largest_pair[1]]

print("\nLargest absolute correlation pair:")
print(f"{largest_pair[0]} and {largest_pair[1]}: {largest_value:.4f}")

# --------------------------------------------------
# 6. BASIC DATA VALIDATION
# --------------------------------------------------
assert len(customers) == 1200
assert len(products) == 40
assert len(orders) == 12309
assert len(completed) == 11110

assert round(mean_amount, 2) == 269.91
assert round(median_amount, 2) == 135.00
assert round(q1, 2) == 64.80
assert round(q3, 2) == 297.00
assert round(upper_fence, 2) == 645.30
assert above_outliers == 897
assert below_outliers == 0

print("\nALL PANDAS ANALYSIS CHECKPOINTS PASSED")

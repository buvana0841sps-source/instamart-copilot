
import sqlite3
from pathlib import Path
import pandas as pd

# File locations
DB_PATH = Path("data/instamart.db")
SQL_PATH = Path("sql/queries.sql")

# Read the seven queries from the SQL file
sql_text = SQL_PATH.read_text(encoding="utf-8")
queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

results = []

# Connect to the database and run every query
with sqlite3.connect(DB_PATH) as conn:
    print("=" * 60)
    print("INSTAMART SQL ANALYSIS")
    print("=" * 60)

    for number, query in enumerate(queries, start=1):
        result = pd.read_sql_query(query, conn)
        results.append(result)

        print(f"\nQ{number}")
        print("-" * 40)
        print("Number of rows:", len(result))
        print(result.head(5).to_string(index=False))

        # Show the last cumulative revenue row for Q7
        if number == 7:
            print("\nFinal daily revenue row:")
            print(result.tail(1).to_string(index=False))

# Check the reference results
assert len(results) == 7, "Expected seven SQL queries."

q1, q2, q3, q4, q5, q6, q7 = results

print("\n" + "=" * 60)
print("SQL REFERENCE CHECKS")
print("=" * 60)

print("Q1 customers:", len(q1))
print("Q2 city-category rows:", len(q2))
print("Q2 total revenue (INR):", round(q2["revenue_inr"].sum(), 2))
print("Q3 products:", len(q3))
print("Q4 customers without completed orders:", len(q4))
print("Q5 customers above average spend:", len(q5))
print("Q5 average spend (INR):",
      round(q5["average_spend_inr"].iloc[0], 2)
      if len(q5) else "No rows")

print("Q6 rows:", len(q6))
print(
    "Q6 rows where RANK differs from ROW_NUMBER:",
    (q6["spend_rank"] != q6["spend_row_number"]).sum()
)

print("Q7 daily rows:", len(q7))
print("Q7 final cumulative revenue (INR):",
      q7["cumulative_revenue_inr"].iloc[-1]
      if len(q7) else "No rows")

# Required reference assertions
assert len(q1) == 30, "Q1 checkpoint failed."
assert len(q2) == 30, "Q2 checkpoint failed."
assert round(q2["revenue_inr"].sum(), 2) == 2998668.80, \
    "Q2 revenue checkpoint failed."
assert len(q3) == 11, "Q3 checkpoint failed."
assert len(q4) == 25, "Q4 checkpoint failed."
assert len(q5) == 550, "Q5 checkpoint failed."
assert round(q5["average_spend_inr"].iloc[0], 2) == 2552.06, \
    "Q5 average checkpoint failed."
assert len(q7) == 183, "Q7 daily row checkpoint failed."
assert round(q7["cumulative_revenue_inr"].iloc[-1], 2) == 2998668.80, \
    "Q7 cumulative revenue checkpoint failed."

print("\nALL SQL CHECKPOINTS PASSED")

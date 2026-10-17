
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. LOAD CLEAN DATA
# --------------------------------------------------
DATA = Path("data/clean")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(DATA / "customers.csv")
orders = pd.read_csv(DATA / "orders.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="raise"
)

completed = orders[orders["status"] == "completed"].copy()

# RFM snapshot: 1 October 2025 at midnight
snapshot = pd.Timestamp("2025-10-01 00:00:00")

# --------------------------------------------------
# 2. CALCULATE RFM
# --------------------------------------------------
rfm = completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
)

rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days

rfm = rfm[["recency", "frequency", "monetary"]].reset_index()

print("=" * 60)
print("RFM SUMMARY")
print("=" * 60)
print("Customers with completed orders:", len(rfm))
print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

# --------------------------------------------------
# 3. STANDARDISE RFM AND TEST K=2 TO K=8
# --------------------------------------------------
features = ["recency", "frequency", "monetary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

scores = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )
    labels = model.fit_predict(X_scaled)

    scores.append({
        "K": k,
        "Inertia": model.inertia_,
        "Silhouette": silhouette_score(X_scaled, labels)
    })

scores_df = pd.DataFrame(scores)

print("\n" + "=" * 60)
print("K-MEANS COMPARISON")
print("=" * 60)
print(scores_df.round(3).to_string(index=False))

# Choose K=4 for business interpretability and the elbow.
# This is the selected number of segments for the project.
FINAL_K = 4

kmeans = KMeans(
    n_clusters=FINAL_K,
    n_init=10,
    random_state=42
)
rfm["segment_id"] = kmeans.fit_predict(X_scaled)

# --------------------------------------------------
# 4. PROFILE THE SEGMENTS
# --------------------------------------------------
profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).round(2)

print("\n" + "=" * 60)
print("SEGMENT PROFILES")
print("=" * 60)
print(profile.to_string())

# The segment with the highest mean monetary value is Premium.
premium_id = profile["mean_monetary"].idxmax()

# Give each cluster a business name based on its observed profile.
# Cluster numbers themselves do not indicate a ranking.
segment_names = {}

remaining_ids = [sid for sid in profile.index if sid != premium_id]
segment_names[premium_id] = "Premium Spenders"

# Among the remaining groups, the highest-frequency group is
# named Frequent Loyalists.
if remaining_ids:
    frequent_id = profile.loc[remaining_ids, "mean_frequency"].idxmax()
    segment_names[frequent_id] = "Frequent Loyalists"
    remaining_ids.remove(frequent_id)

# Among remaining groups, the least recently active group has
# the largest recency and is named At-Risk Shoppers.
if remaining_ids:
    at_risk_id = profile.loc[remaining_ids, "mean_recency"].idxmax()
    segment_names[at_risk_id] = "At-Risk Shoppers"
    remaining_ids.remove(at_risk_id)

# Name any remaining cluster as steady shoppers.
for sid in remaining_ids:
    segment_names[sid] = "Steady Shoppers"

rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

print("\nSEGMENT NAMES AND ACTIONS")
actions = {
    "Premium Spenders": "Offer personalised high-value rewards.",
    "Frequent Loyalists": "Encourage repeat orders with loyalty benefits.",
    "At-Risk Shoppers": "Send a targeted win-back offer.",
    "Steady Shoppers": "Use relevant recommendations to increase frequency."
}

for sid in sorted(segment_names):
    name = segment_names[sid]
    print(f"{sid}: {name} — {actions[name]}")

# --------------------------------------------------
# 5. PCA WITH TWO COMPONENTS
# --------------------------------------------------
pca = PCA(n_components=2)
pca.fit(X_scaled)

print("\n" + "=" * 60)
print("PCA EXPLAINED VARIANCE")
print("=" * 60)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())

# --------------------------------------------------
# 6. SAVE CUSTOMER SEGMENTS
# --------------------------------------------------
final_columns = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium"
]

rfm[final_columns].to_csv(
    OUTPUTS / "customer_segments.csv",
    index=False
)

assert len(rfm) == 1175
assert rfm["customer_id"].is_unique
assert rfm["is_premium"].sum() == profile.loc[
    premium_id, "customer_count"
]

print("\nSaved:", OUTPUTS / "customer_segments.csv")
print("Premium segment:", segment_names[premium_id])
print("Customer segmentation completed successfully.")

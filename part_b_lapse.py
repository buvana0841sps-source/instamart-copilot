
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# 1. LOAD THE CLEAN DATA
# =========================================================

DATA = Path("data/clean")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv")

customers["signup_date"] = pd.to_datetime(customers["signup_date"])
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="raise"
)
orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="raise"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="raise"
)

# The cutoff is 1 September 2025.
# Only orders strictly BEFORE this date enter the model features.
cutoff = pd.Timestamp("2025-09-01")
september_end = pd.Timestamp("2025-10-01")

pre_cutoff = orders[
    (orders["order_ts"] < cutoff)
    & orders["status"].isin(["completed", "cancelled"])
].copy()

pre_completed = pre_cutoff[
    pre_cutoff["status"] == "completed"
].copy()

# Customers must have at least one completed order before the cutoff.
cohort_ids = sorted(pre_completed["customer_id"].unique())

print("=" * 65)
print("LAPSE PREDICTION: DATA PREPARATION")
print("=" * 65)
print("Cohort customers:", len(cohort_ids))


# =========================================================
# 2. BUILD FEATURES USING PRE-CUTOFF DATA ONLY
# =========================================================

# Aggregate completed orders only.
behaviour = pre_completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount_pct=("discount_pct", "mean"),
    avg_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

behaviour["recency"] = (
    cutoff - behaviour["last_order_ts"]
).dt.days

# Find each customer's favourite product category before cutoff.
pre_category_orders = pre_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

def favourite_category(series):
    counts = series.value_counts()
    # Sorting the category names makes ties deterministic.
    return counts[counts == counts.max()].index.sort_values()[0]

favourites = (
    pre_category_orders.groupby("customer_id")["category"]
    .agg(favourite_category)
    .rename("favourite_category")
    .reset_index()
)

# Cancellation rate uses final-status orders before cutoff.
final_counts = pre_cutoff.groupby("customer_id").agg(
    total_final_orders=("order_id", "nunique"),
    cancelled_orders=(
        "status",
        lambda values: int((values == "cancelled").sum())
    )
).reset_index()

final_counts["cancellation_rate"] = (
    final_counts["cancelled_orders"]
    / final_counts["total_final_orders"]
)

# Combine customer behaviour with customer demographics.
features = pd.DataFrame({"customer_id": cohort_ids})

features = features.merge(
    behaviour.drop(columns=["last_order_ts"]),
    on="customer_id",
    how="left",
    validate="one_to_one"
)
features = features.merge(
    favourites,
    on="customer_id",
    how="left",
    validate="one_to_one"
)
features = features.merge(
    final_counts[["customer_id", "cancellation_rate"]],
    on="customer_id",
    how="left",
    validate="one_to_one"
)
features = features.merge(
    customers[
        ["customer_id", "city", "age", "gender", "signup_date"]
    ],
    on="customer_id",
    how="left",
    validate="one_to_one"
)

features["tenure_days"] = (
    cutoff - features["signup_date"]
).dt.days

features = features.drop(columns=["signup_date"])

# =========================================================
# 3. CREATE THE LAPSE LABEL
# =========================================================
# This label uses completed orders in September.
# September information is NOT used in the feature calculations.

september_completed = orders[
    (orders["status"] == "completed")
    & (orders["order_ts"] >= cutoff)
    & (orders["order_ts"] < september_end)
]

september_active_ids = set(
    september_completed["customer_id"].unique()
)

features["lapsed"] = (
    ~features["customer_id"].isin(september_active_ids)
).astype(int)

print("\nCohort size:", len(features))
print("Lapsed customers:", int(features["lapsed"].sum()))
print(
    "Lapse percentage:",
    round(features["lapsed"].mean() * 100, 2), "%"
)

assert len(features) == 1156, "Unexpected cohort size."
assert int(features["lapsed"].sum()) == 443, "Unexpected lapse count."


# =========================================================
# 4. CHECK FEATURES AND SPLIT THE DATA
# =========================================================

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

feature_columns = numeric_features + categorical_features

X = features[feature_columns].copy()
y = features["lapsed"].copy()
customer_ids = features["customer_id"].copy()

# Check all feature columns BEFORE encoding.
assert not X.isna().any().any(), (
    "Missing feature values detected. Inspect X.isna().sum()."
)

# Split before fitting any encoder, scaler or model.
X_train, X_test, y_train, y_test, ids_train, ids_test = (
    train_test_split(
        X,
        y,
        customer_ids,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
)

print("\nTraining rows:", len(X_train))
print("Test rows:", len(X_test))
print("Training class counts:")
print(y_train.value_counts().sort_index())
print("Test class counts:")
print(y_test.value_counts().sort_index())


# =========================================================
# 5. BASELINE: ALWAYS PREDICT NOT LAPSED
# =========================================================

baseline_predictions = np.zeros(len(y_test), dtype=int)

print("\n" + "=" * 65)
print("BASELINE: ALWAYS PREDICT NOT LAPSED")
print("=" * 65)
print("Accuracy:", round(
    accuracy_score(y_test, baseline_predictions), 4
))
print("Lapsed precision:", round(
    precision_score(
        y_test, baseline_predictions, zero_division=0
    ), 4
))
print("Lapsed recall:", round(
    recall_score(
        y_test, baseline_predictions, zero_division=0
    ), 4
))
print("Lapsed F1:", round(
    f1_score(
        y_test, baseline_predictions, zero_division=0
    ), 4
))


# =========================================================
# 6. PREPROCESSING PIPELINE
# =========================================================

numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# =========================================================
# 7. TUNE LOGISTIC REGRESSION
# =========================================================

logistic_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_search = GridSearchCV(
    logistic_pipeline,
    param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1",
    n_jobs=1
)

print("\nTuning Logistic Regression...")
logistic_search.fit(X_train, y_train)

print("Logistic best parameters:", logistic_search.best_params_)
print(
    "Logistic best CV F1:",
    round(logistic_search.best_score_, 4)
)


# =========================================================
# 8. TUNE RANDOM FOREST
# =========================================================

forest_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

forest_search = GridSearchCV(
    forest_pipeline,
    param_grid={
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    },
    cv=5,
    scoring="f1",
    n_jobs=1
)

print("\nTuning Random Forest...")
forest_search.fit(X_train, y_train)

print("Random Forest best parameters:", forest_search.best_params_)
print(
    "Random Forest best CV F1:",
    round(forest_search.best_score_, 4)
)


# =========================================================
# 9. EVALUATE BOTH MODELS ON THE UNTOUCHED TEST SET
# =========================================================

def report_metrics(name, model, X_eval, y_eval):
    predictions = model.predict(X_eval)

    print("\n" + "-" * 55)
    print(name)
    print("Confusion matrix (labels: 0=active, 1=lapsed):")
    print(confusion_matrix(y_eval, predictions, labels=[0, 1]))
    print("Accuracy:", round(
        accuracy_score(y_eval, predictions), 4
    ))
    print("Lapsed precision:", round(
        precision_score(y_eval, predictions, zero_division=0), 4
    ))
    print("Lapsed recall:", round(
        recall_score(y_eval, predictions, zero_division=0), 4
    ))
    print("Lapsed F1:", round(
        f1_score(y_eval, predictions, zero_division=0), 4
    ))

    return predictions

logistic_predictions = report_metrics(
    "Tuned Logistic Regression",
    logistic_search.best_estimator_,
    X_test,
    y_test
)

forest_predictions = report_metrics(
    "Tuned Random Forest",
    forest_search.best_estimator_,
    X_test,
    y_test
)


# =========================================================
# 10. PRINT THE LEARNED TRAINING SCALER MEAN
# =========================================================

fitted_numeric_pipeline = (
    logistic_search.best_estimator_
    .named_steps["preprocessor"]
    .named_transformers_["numeric"]
)

scaler = fitted_numeric_pipeline.named_steps["scaler"]
frequency_position = numeric_features.index("frequency")

print("\n" + "=" * 65)
print("TRAINING SCALER CHECK")
print("=" * 65)
print(
    "Frequency mean learned by training scaler:",
    round(scaler.mean_[frequency_position], 4)
)
print(
    "Full cohort frequency mean:",
    round(X["frequency"].mean(), 4)
)
print(
    "These means can differ because the scaler was fitted only "
    "on training rows, while the cohort mean also includes test rows."
)


# =========================================================
# 11. RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

best_forest = forest_search.best_estimator_
forest_preprocessor = best_forest.named_steps["preprocessor"]
forest_model = best_forest.named_steps["model"]

transformed_names = forest_preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": transformed_names,
    "importance": forest_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n" + "=" * 65)
print("TOP FIVE RANDOM FOREST FEATURES")
print("=" * 65)
print(importance_df.head(5).to_string(index=False))


# =========================================================
# 12. LOGISTIC REGRESSION COEFFICIENT SIGNS
# =========================================================

best_logistic = logistic_search.best_estimator_
lr_names = best_logistic.named_steps[
    "preprocessor"
].get_feature_names_out()
lr_coefficients = best_logistic.named_steps["model"].coef_[0]

coefficient_df = pd.DataFrame({
    "feature": lr_names,
    "coefficient": lr_coefficients
})

print("\n" + "=" * 65)
print("LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 65)

for target_feature in ["recency", "frequency", "monetary"]:
    matching = coefficient_df[
        coefficient_df["feature"].str.endswith(
            "__" + target_feature
        )
    ]

    if not matching.empty:
        row = matching.iloc[0]
        sign = "positive" if row["coefficient"] > 0 else (
            "negative" if row["coefficient"] < 0 else "zero"
        )
        print(
            f"{target_feature}: {row['coefficient']:.4f} "
            f"({sign})"
        )


# =========================================================
# 13. OVER-SAMPLE LAPSED TRAINING ROWS ONLY
# =========================================================

print("\n" + "=" * 65)
print("CLASS-BALANCE EXPERIMENT")
print("=" * 65)

train_table = X_train.copy()
train_table["lapsed"] = y_train.to_numpy()

counts = train_table["lapsed"].value_counts()
target_count = int(counts.max())

minority_rows = train_table[
    train_table["lapsed"] == 1
]

number_to_add = target_count - len(minority_rows)

rng = np.random.default_rng(42)

if number_to_add > 0:
    chosen_positions = rng.choice(
        len(minority_rows),
        size=number_to_add,
        replace=True
    )
    extra_rows = minority_rows.iloc[chosen_positions].copy()
    balanced_table = pd.concat(
        [train_table, extra_rows],
        ignore_index=True
    )
else:
    balanced_table = train_table.reset_index(drop=True)

X_balanced = balanced_table[feature_columns]
y_balanced = balanced_table["lapsed"].astype(int)

print("Balanced training class counts:")
print(y_balanced.value_counts().sort_index())

# Fit only on resampled training rows; never resample the test data.
balanced_logistic = clone(
    logistic_search.best_estimator_
)
balanced_logistic.fit(X_balanced, y_balanced)

balanced_predictions = balanced_logistic.predict(X_test)

print("\nOriginal Logistic Regression:")
print("Precision:", round(
    precision_score(y_test, logistic_predictions, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, logistic_predictions, zero_division=0), 4
))
print("F1:", round(
    f1_score(y_test, logistic_predictions, zero_division=0), 4
))

print("\nBalanced Logistic Regression:")
print("Precision:", round(
    precision_score(y_test, balanced_predictions, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, balanced_predictions, zero_division=0), 4
))
print("F1:", round(
    f1_score(y_test, balanced_predictions, zero_division=0), 4
))


# =========================================================
# 14. SELECT A MODEL USING CROSS-VALIDATION RESULTS
# =========================================================

# Choose using training cross-validation, not test-set performance.
if forest_search.best_score_ >= logistic_search.best_score_:
    selected_name = "Random Forest"
    selected_model = forest_search.best_estimator_
else:
    selected_name = "Logistic Regression"
    selected_model = logistic_search.best_estimator_

selected_predictions = selected_model.predict(X_test)
selected_f1 = f1_score(
    y_test, selected_predictions, zero_division=0
)

print("\n" + "=" * 65)
print("SELECTED MODEL")
print("=" * 65)
print("Selected using best cross-validated F1:", selected_name)
print("Test lapsed-class F1:", round(selected_f1, 4))

if selected_f1 < 0.55:
    print(
        "WARNING: The test F1 is below the assignment target of 0.55. "
        "Review the results before submission."
    )


# =========================================================
# 15. SAVE AND RELOAD THE SELECTED MODEL
# =========================================================

model_path = OUTPUTS / "lapse_model.joblib"
joblib.dump(selected_model, model_path)

reloaded_model = joblib.load(model_path)

predictions_after_reload = reloaded_model.predict(X_test)

assert np.array_equal(
    selected_predictions, predictions_after_reload
)

print("Model saved and reloaded:", model_path)
print("Reloaded model predictions match exactly.")


# =========================================================
# 16. CREATE THE PREMIUM RETENTION LIST
# =========================================================

segments = pd.read_csv(OUTPUTS / "customer_segments.csv")

if segments["is_premium"].dtype != bool:
    segments["is_premium"] = (
        segments["is_premium"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

test_customers = pd.DataFrame({
    "customer_id": ids_test.to_numpy(),
    "actually_lapsed": y_test.to_numpy()
})

test_customers = test_customers.merge(
    segments[["customer_id", "is_premium"]],
    on="customer_id",
    how="left",
    validate="one_to_one"
)

test_customers["is_premium"] = (
    test_customers["is_premium"].fillna(False)
)

test_customers["lapse_probability"] = (
    reloaded_model.predict_proba(X_test)[:, 
        list(reloaded_model.classes_).index(1)
    ]
)

premium_test = test_customers[
    test_customers["is_premium"]
].copy()

retention_list = premium_test.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

retention_list = retention_list[
    ["customer_id", "lapse_probability", "actually_lapsed"]
].copy()

retention_list.to_csv(
    OUTPUTS / "retention_list.csv",
    index=False
)

print("\n" + "=" * 65)
print("PREMIUM RETENTION LIST")
print("=" * 65)
print(retention_list.to_string(index=False))
print("Customers in retention list:", len(retention_list))
print(
    "Actually lapsed in retention list:",
    int(retention_list["actually_lapsed"].sum())
)
print("Saved:", OUTPUTS / "retention_list.csv")

print("\nLapse modelling completed.")

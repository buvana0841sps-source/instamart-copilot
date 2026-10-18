# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds an end-to-end customer intelligence and support copilot using Python, Pandas, NumPy, scikit-learn, SQLite, TF-IDF retrieval, and the supplied offline `mock_llm.py` model. All project data is synthetic and fictional. All monetary values are expressed in Indian Rupees (INR).

The project has four connected parts:

- **Part A — Data Foundation:** Audit and clean raw customer, product, and order data; create a SQLite database; answer SQL and Pandas analysis questions.
- **Part B — Customer Intelligence:** Create RFM customer segments, train and evaluate a lapse prediction model, and produce an illustrative retention list.
- **Part C — Support Ticket Triage:** Redact personal information, classify support tickets, validate model outputs, retry transient failures, apply a token budget, and route tickets using Part B segments.
- **Part D — Policy and Refund Agent:** Retrieve policy passages, answer questions from policy context, use refund tools, apply safety limits, and demonstrate checkpoint recovery.

All model calls use the supplied `mock_llm.py`. The project requires no API key, paid service, or network-based model call.

### Project structure

```text
instamart-copilot/
├── data/
│   ├── raw/
│   ├── clean/
│   ├── policies/
│   └── instamart.db
├── outputs/
├── sql/
│   └── queries.sql
├── state/
├── generate_data.py
├── mock_llm.py
├── part_a_audit.py
├── part_a_clean.py
├── part_a_store.py
├── part_a_sql.py
├── part_a_analysis.py
├── part_a_validate.py
├── part_b_segments.py
├── part_b_lapse.py
├── part_c_triage.py
├── part_d_policy.py
├── part_d_eval.py
├── part_d_agent.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproduce

Run these numbered steps from the repository root in Windows PowerShell, in order from environment setup through the last script.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` to run the commands instead.

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Generate the synthetic data

```powershell
python generate_data.py
```

### 4. Run Part A

```powershell
python part_a_audit.py
python part_a_clean.py
python part_a_store.py
python part_a_sql.py
python part_a_analysis.py
python part_a_validate.py
```

The Part A validation script should print `ALL CHECKPOINTS PASSED`.

### 5. Run Part B

```powershell
python part_b_segments.py
python part_b_lapse.py
```

### 6. Run Part C

```powershell
python part_c_triage.py
```

### 7. Run Part D

```powershell
python part_d_policy.py
python part_d_eval.py
python part_d_agent.py
```

Run the commands from the repository root so relative paths such as `data/raw/`, `data/clean/`, `data/policies/`, `outputs/`, and `state/` resolve correctly.

## Part A

### Raw-data audit

Missing percentages use the row count of each respective raw table as the denominator.

**Customers — 1,200 rows × 5 columns**

| Column | Data type | Missing count | Missing % |
|---|---|---:|---:|
| `customer_id` | `str` | 0 | 0.00% |
| `city` | `str` | 0 | 0.00% |
| `signup_date` | `str` | 0 | 0.00% |
| `age` | `float64` | 61 | 5.08% |
| `gender` | `str` | 30 | 2.50% |

**Products — 40 rows × 4 columns**

| Column | Data type | Missing count | Missing % |
|---|---|---:|---:|
| `product_id` | `str` | 0 | 0.00% |
| `product_name` | `str` | 0 | 0.00% |
| `category` | `str` | 0 | 0.00% |
| `price_inr` | `int64` | 0 | 0.00% |

**Orders — 13,084 rows × 9 columns**

| Column | Data type | Missing count | Missing % |
|---|---|---:|---:|
| `order_id` | `str` | 0 | 0.00% |
| `customer_id` | `str` | 0 | 0.00% |
| `product_id` | `str` | 0 | 0.00% |
| `quantity` | `int64` | 0 | 0.00% |
| `order_ts` | `str` | 0 | 0.00% |
| `status` | `str` | 0 | 0.00% |
| `discount_pct` | `int64` | 0 | 0.00% |
| `order_amount` | `str` | 53 | 0.41% |
| `delivery_minutes` | `float64` | 2,541 | 19.42% |

The raw audit identified 120 order amounts that could not be converted to numbers, including values that were present but non-numeric.

### Raw-data audit checkpoints

| Audit item | Result |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Pending-status rows | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery times in raw data | 106 |
| Blank delivery times in raw data | 2,541 |
| Blank customer ages | 61 |
| Ages outside the 10–100 range | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank gender values | 30 |

### Cleaning rules and results

**Customers**

- City names were trimmed and converted to title case, standardising 24 raw spellings to six cities.
- The raw gender spellings were mapped to `male`, `female`, or `unknown`; 30 blank gender values were represented as `unknown`.
- The final gender counts were 610 female, 560 male, and 30 unknown.
- Ages were converted to numeric values. Missing ages and ages outside the 10–100 range were treated as missing, then imputed using the median age within the customer's city.
- The clean customer table has no missing ages, and its final age range is 18–60.
- `signup_date` was parsed as a date.
- Final customer rows: 1,200.

| City | Customer count |
|---|---:|
| Bengaluru | 197 |
| Chennai | 191 |
| Delhi | 210 |
| Hyderabad | 191 |
| Mumbai | 184 |
| Pune | 227 |

**Products**

- Product prices were converted to numeric values.
- Final product rows: 40.

**Orders**

- Order timestamps were parsed as dates and times.
- The 119 exact duplicate rows were removed.
- Records were ordered by timestamp, and 656 older records were removed by keeping the latest record for each order ID.
- The clean order table contains 12,309 unique orders and no pending records.
- Quantity, discount, order amount, and delivery duration were converted to numeric values.
- Missing or invalid order amounts were recalculated using `price_inr × quantity × (1 − discount_pct / 100)`, rounded to two decimal places.
- Negative delivery durations were changed to missing. The raw audit found 106 negative delivery times; after resolving duplicate and older records, the cleaning run reported 105 surviving negative values changed to missing.
- Missing delivery times for completed orders were filled using the median delivery time within the product category.
- Delivery time remains missing for cancelled orders.
- Final clean order rows: 12,309, comprising 11,110 completed orders and 1,199 cancelled orders.

### Final cleaning checkpoints

| Check | Result |
|---|---:|
| Clean customers | 1,200 |
| Clean products | 40 |
| Clean orders | 12,309 |
| Unique order IDs | Passed |
| Pending status in clean orders | None |
| Missing order amounts | None |
| Completed orders with missing delivery times | None |
| Cancelled orders with missing delivery times | 1,199 |
| Distinct customer cities | 6 |
| Customers with missing ages | None |
| Completed-order revenue | INR 2,998,668.80 |

The clean CSV files are saved under `data/clean/`. The SQLite database `data/instamart.db` contains the `customers`, `products`, and `orders` tables.

### SQL analysis

Seven SQL queries are saved in `sql/queries.sql` and run against `data/instamart.db`.

| Query | Finding | Result |
|---|---|---:|
| Q1 — `WHERE` then `HAVING` | Customers with qualifying completed orders and total qualifying spend above the threshold | 30 customers |
| Q2 — Three-table join | Revenue by city and product category | 30 city-category rows; total INR 2,998,668.80 |
| Q2 — Highest revenue cell | Bengaluru / Electronics | INR 233,909.45 |
| Q3 — Scalar subquery | Products priced above average catalogue price | 11 products |
| Q4 — `LEFT JOIN` | Customers without a completed order | 25 customers |
| Q5 — CTE | Customers with total completed spend above average | 550 customers; average INR 2,552.06 |
| Q6 — Ranking window functions | Customers with completed spend | 1,175 result rows; eight positions differed between `RANK()` and `ROW_NUMBER()` |
| Q7 — Running total | Daily completed revenue | 183 days; final cumulative revenue INR 2,998,668.80 |

The top-spending customer in each city was:

| City | Customer ID | Total completed spend |
|---|---|---:|
| Bengaluru | C0082 | INR 12,856.20 |
| Chennai | C0930 | INR 10,151.40 |
| Delhi | C0291 | INR 13,742.60 |
| Hyderabad | C0743 | INR 12,730.05 |
| Mumbai | C0693 | INR 14,766.55 |
| Pune | C0426 | INR 10,377.10 |

`RANK()` assigns tied values the same rank and leaves gaps in later ranks, while `ROW_NUMBER()` assigns a unique sequential number to each row. This explains why ties can cause the ranking positions to differ.

The Part A validation script printed `ALL CHECKPOINTS PASSED`.

### Pandas cross-check and exploratory analysis

The Pandas pivot table reconstructed revenue by city and product category. Its grand total was INR 2,998,668.80, matching completed-order revenue and SQL Q2.

**Completed-order statistics by city**

| City | Total revenue (INR) | Mean order amount (INR) | Completed orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

**Completed-order amount distribution**

| Statistic | Result |
|---|---:|
| Mean | INR 269.91 |
| Median | INR 135.00 |
| Skewness | 4.6088 |
| First quartile (Q1) | INR 64.80 |
| Third quartile (Q3) | INR 297.00 |
| Interquartile range (IQR) | INR 232.20 |
| Lower 1.5 × IQR fence | INR -283.50 |
| Upper 1.5 × IQR fence | INR 645.30 |
| Outliers below lower fence | 0 |
| Outliers above upper fence | 897 |

**Correlation analysis**

The largest absolute correlation was between `order_amount` and `discount_pct`, at -0.0951. The correlation between `order_amount` and `quantity` was +0.0873.

**Interpretation**

- **What is the data about?** The dataset contains completed customer orders, their amounts, discounts, quantities, and delivery durations across six cities.
- **What does the distribution look like?** The mean order amount (INR 269.91) is higher than the median (INR 135.00), and skewness of 4.6088 indicates a strongly right-skewed distribution.
- **Are there outliers?** There are 897 order amounts above the upper IQR fence of INR 645.30 and none below the lower fence. These are statistical outliers and are not automatically errors.
- **What relationships exist?** Order amount and discount percentage have the largest absolute correlation, at -0.0951. This is a weak negative relationship; correlation of this size does not establish causation.

The Pandas analysis printed `ALL PANDAS ANALYSIS CHECKPOINTS PASSED`.

## Part B

### RFM segmentation

RFM values were computed for customers with at least one completed order using the snapshot moment 2025-10-01 00:00:00.

| Metric | Result |
|---|---:|
| Customers in the RFM analysis | 1,175 |
| Mean purchase frequency | 9.46 |
| Maximum purchase frequency | 33 |
| Mean monetary value | INR 2,552.06 |
| Maximum monetary value | INR 14,766.55 |
| Median recency | 19 days |
| Minimum recency | 0 days |
| Maximum recency | 180 days |

### K-means comparison

The three RFM features were standardised before fitting K-means. Models with K = 2 through K = 8 were compared using `n_init=10` and `random_state=42`.

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1,820.761 | 0.465 |
| 3 | 1,268.235 | 0.445 |
| 4 | 743.071 | 0.526 |
| 5 | 617.367 | 0.480 |
| 6 | 503.171 | 0.482 |
| 7 | 408.460 | 0.416 |
| 8 | 366.503 | 0.412 |

K = 4 had the highest silhouette score among the tested configurations and was
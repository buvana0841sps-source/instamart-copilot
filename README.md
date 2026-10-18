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
- Ages were converted to numeric values. The 61 blank ages and 6 out-of-range ages (67 values in total) were treated as missing, then imputed using the median age within the customer's city.
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
- Records were ordered by timestamp, and older records were removed by keeping the latest record for each order ID.
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
| Q1 — `WHERE` then `HAVING` | Completed orders with order_amount > INR 500; retain customers whose total qualifying spend exceeds INR 8,000. | 30 customers |
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

K = 4 had the highest silhouette score among the tested configurations and was selected for four interpretable customer groups. Inertia naturally falls as more clusters are added because each cluster can fit a smaller share of the data; therefore, K = 8 having lower inertia than K = 2 does not by itself make K = 8 the best business choice.

### Four customer segments

| Segment ID | Customers | Mean recency (days) | Mean frequency | Mean monetary value | Segment name |
|---:|---:|---:|---:|---:|---|
| 0 | 436 | 28.86 | 3.77 | INR 1,057.62 | Steady Shoppers |
| 1 | 138 | 24.12 | 6.77 | INR 7,129.70 | Premium Spenders |
| 2 | 419 | 8.77 | 19.40 | INR 3,410.02 | Frequent Loyalists |
| 3 | 182 | 109.04 | 2.20 | INR 685.99 | At-Risk Shoppers |

Cluster IDs are labels only; they do not imply a ranking from best to worst. Premium is assigned to the segment with the highest mean monetary value, Premium Spenders.

| Segment | Recommended action |
|---|---|
| Steady Shoppers | Encourage repeat purchases with relevant product bundles and occasional targeted offers. |
| Premium Spenders | Provide personalised recommendations and loyalty rewards to encourage continued high-value purchases. |
| Frequent Loyalists | Reward repeat purchases with loyalty benefits and relevant complementary-product recommendations. |
| At-Risk Shoppers | Run a win-back campaign with relevant offers and monitor whether purchasing activity resumes. |

The first two principal components explained approximately 85.23% of the variance in the scaled RFM matrix: PC1 = 63.43%, PC2 = 21.80%.

`outputs/customer_segments.csv` contains `customer_id`, `recency`, `frequency`, `monetary`, `segment_id`, `segment_name`, and `is_premium`.

### Customer lapse prediction

The lapse cohort used the 2025-09-01 cutoff. Customers had at least one completed order before the cutoff, and the label was based on whether they had a completed order from 2025-09-01 through 2025-09-30.

| Cohort metric | Result |
|---|---:|
| Lapse cohort | 1,156 customers |
| Lapsed customers | 443 |
| Lapse rate | 38.32% |
| Training rows | 924 |
| Test rows | 232 |

The baseline that always predicts “not lapsed” achieved accuracy 0.6164 and lapsed-class F1 = 0.

**Tuning configuration**

- Logistic Regression: `max_iter=1000`; grid `C ∈ {0.01, 0.1, 1, 10, 100}`; 5-fold cross-validation; scoring by F1.
- Random Forest: `n_estimators=100`, `random_state=42`; grid `max_depth ∈ {3, 5, 8}` and `min_samples_leaf ∈ {5, 10, 20}`; 5-fold cross-validation; scoring by F1.

**Logistic Regression**

| Metric | Result |
|---|---:|
| Best C | 100 |
| Best cross-validation F1 | 0.6506 |
| Test precision for lapsed class | 0.6444 |
| Test recall for lapsed class | 0.6517 |
| Test F1 for lapsed class | 0.6480 |
| Test confusion matrix | `[[111, 32], [31, 58]]` |

**Random Forest**

| Metric | Result |
|---|---:|
| Best maximum depth | 8 |
| Best minimum leaf size | 10 |
| Best cross-validation F1 | 0.6811 |
| Test precision for lapsed class | 0.6591 |
| Test recall for lapsed class | 0.6517 |
| Test F1 for lapsed class | 0.6554 |
| Test confusion matrix | `[[113, 30], [31, 58]]` |

The Random Forest was selected for the persisted model because it had the better cross-validation and test F1 scores. A missed lapsing customer can mean losing a retention opportunity, while wrongly targeting a loyal customer consumes campaign budget and can reduce the value of offers. The selected model balances these costs with its measured precision and recall.

**Preprocessing and leakage prevention**

The model features are recency, frequency, monetary value, average discount percentage, cancellation rate, average delivery time, most frequent product category, city, age, gender, and tenure in days. They use only orders dated before 2025-09-01; orders from 2025-09-01 onwards were excluded from the features. Segment labels were not used as model features. Numeric features are standardised with `StandardScaler`, and categorical features (city, gender, and favourite category) are one-hot encoded using `OneHotEncoder(drop="first", handle_unknown="ignore")`. The transformations are included in a `ColumnTransformer` and `Pipeline`. The train/test split happens before fitting preprocessing or model steps, so preprocessing is fitted on training data and applied to test data. Oversampling was applied only to training data, and the original test set remained unchanged. The feature table contained no missing values.

The fitted scaler's training-set mean for frequency was 8.1147, compared with the full-cohort mean of 8.0285. These differ because the scaler was fitted on the training subset, not on the full cohort.

**Leading Random Forest feature importances**

| Feature | Importance |
|---|---:|
| Frequency | 0.279589 |
| Monetary value | 0.271176 |
| Recency | 0.115149 |
| Average delivery time | 0.061286 |
| Cancellation rate | 0.052325 |

**Logistic Regression coefficient interpretation**

| Feature | Coefficient | Business interpretation |
|---|---:|---|
| Recency | +0.0396 | Higher recency is associated with higher predicted lapse probability, so customers who have not ordered recently may need retention attention. |
| Frequency | -1.6982 | Higher purchase frequency is associated with lower predicted lapse probability, indicating that repeat purchasing is a useful retention signal. |
| Monetary value | +0.2846 | Holding the other model features constant, higher monetary value is associated with higher predicted lapse probability in this fitted model. This association should be investigated rather than interpreted as causal. |

**Class-balance experiment**

Oversampling the lapsed class in the training data changed Logistic Regression metrics as follows:

| Metric | Original Logistic Regression | Oversampled Logistic Regression |
|---|---:|---:|
| Precision | 0.6444 | 0.6000 |
| Recall | 0.6517 | 0.8427 |
| F1 | 0.6480 | 0.7009 |

Recall moved the most, increasing by about 0.19, while precision decreased slightly. This means the model identifies more lapsing customers but also risks sending more loyal customers into retention campaigns.

The selected model was saved to `outputs/lapse_model.joblib`, reloaded with `joblib.load`, and reproduced the original test predictions exactly. Among test-set customers who are in the Premium segment, the top 10 by predicted lapse probability were written to `outputs/retention_list.csv` with columns `customer_id`, `lapse_probability`, and `actually_lapsed`; one of the ten was labelled lapsed.

**Retention-list caveat:** The Premium flag uses whole-window monetary value, including September. The retention list is illustrative and is not a leakage-free prospective evaluation of Premium customer targeting.

## Part C

### Pipeline and privacy safeguards

The triage pipeline uses the supplied `mock_llm.py`, processes tickets in source-file order, and applies the following email, phone, and order-ID regular expressions in order:

- Email: `[\w.+-]+@[\w-]+\.[\w.]+` → `[EMAIL]`
- Phone: `\b[6-9]\d{9}\b` → `[PHONE]`
- Order ID: `ORD-\d{6}` → `[ORDER_ID]`

The redaction audit found 55 email matches, 91 phone-number matches, and 132 order-ID matches, for 278 replacements total. Reapplying the patterns found zero remaining matches.

Empty or whitespace-only tickets were skipped, and tickets whose estimated length exceeded 400 tokens were flagged as long. Four empty tickets and four long tickets were not sent to the model. The long tickets estimated to 734 tokens.

### Full-run status and reliability results

| Metric | Result |
|---|---:|
| Input tickets | 164 |
| Valid JSON classifications | 135 |
| Keyword fallback classifications | 12 |
| Invalid model outputs | 9 |
| Empty tickets skipped | 4 |
| Long tickets flagged | 4 |
| Successfully classified tickets | 147 |
| Transient errors recovered | 19 |
| Model call attempts, including retries | 175 |

Both zero-shot and few-shot runs recovered all 19 transient errors. Retry waits followed the configured exponential backoff with jitter; the observed waits were within the required range for retry number 0.

### Zero-shot and few-shot comparison

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Tickets processed | 164 | 164 |
| Successfully classified tickets | 147 | 147 |
| Valid JSON classifications | 135 | 135 |
| Keyword fallback classifications | 12 | 12 |
| Invalid outputs | 9 | 9 |
| Accuracy on classified tickets | 85.71% | 100.00% |
| Total response tokens | 17,044 | 34,672 |
| Tokens per classified ticket | 115.95 | 235.86 |
| Model call attempts, including retries | 175 | 175 |
| Transient errors recovered | 19 | 19 |

**Zero-shot confusion matrix** — rows are true categories; columns are predicted categories.

| True category | Billing | Delivery | Technical | Fraud |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Technical | 10 | 0 | 29 | 0 |
| Fraud | 11 | 0 | 0 | 23 |

Few-shot classification used 17,628 more response tokens than zero-shot, an increase of approximately 119.91 tokens per classified ticket. In this run, the additional tokens increased accuracy from 85.71% (126 of 147) to 100% (147 of 147). The zero-shot errors were 11 fraud tickets and 10 technical tickets predicted as billing. The extra few-shot cost is worthwhile when avoiding such misclassification matters for safe, accurate support routing.

### Token-budget demonstration

The zero-shot token-budget demonstration used a token budget of 6,000.

| Metric | Result |
|---|---:|
| Configured token budget | 6,000 |
| Total response tokens recorded | 6,009 |
| Tickets marked `budget_exhausted` | 101 |
| Tickets classified before exhaustion | 52 |

The last permitted call can take the running total beyond 6,000; later calls are blocked when the running total has reached or exceeded the configured limit. The output was saved as `outputs/triage_budget_demo.csv`.

### Routing, validation, and human review

The final `outputs/triage_results.csv` was generated with the few-shot prompt and a token budget of 100,000. The result has 164 rows and columns `ticket_id`, `customer_id`, `status`, `category`, `segment_name`, `priority`, `route`, and `tokens_used`.

- Tickets with status `skipped_empty`, `flagged_long`, `flagged_invalid_output`, or `budget_exhausted` are routed to `human_review` with NORMAL priority.
- A valid fraud classification is routed to `security_team` with HIGH priority.
- Otherwise, a premium customer is routed to `premium_desk` with HIGH priority.
- All other eligible tickets are routed to `support_queue` with NORMAL priority.
- A customer without a row in `outputs/customer_segments.csv` is treated as not premium.
- A JSON classification is accepted only if its category is one of `billing`, `delivery`, `technical`, or `fraud`. A non-JSON response is accepted as `classified_keyword` only when exactly one allowed category word can be extracted. Invalid outputs are flagged and sent for human review instead of guessing.

Thus, `classified_json` outputs were JSON/category validated, `classified_keyword` outputs passed the allowed-category word check, and empty, long, and invalid-output tickets were held for human review. The budget demo also records tickets stopped due to budget exhaustion.

Outputs: `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv`.

The redaction counts represent regex matches, not a guarantee that every possible form of personal information has been removed.

## Part D

### Policy chunking and retrieval

The five policy text files were split into 40-word chunks with 10 words of overlap, producing 15 chunks.

| Policy file | Chunks |
|---|---:|
| `refund_policy.txt` | 4 |
| `delivery_policy.txt` | 3 |
| `escalation_policy.txt` | 3 |
| `payment_policy.txt` | 3 |
| `privacy_policy.txt` | 2 |
| **Total** | **15** |

The assistant uses `TfidfVectorizer` and cosine similarity to rank the three most relevant chunks. Term frequency (TF) reflects how often a term appears in a chunk, while inverse document frequency (IDF) gives less weight to terms that occur across many chunks. TF-IDF is lexical, so it can miss relevant passages when a question and policy use different words for the same idea.

For the query `refund policy late delivery`, all top-three chunks came from `refund_policy.txt`.

### Policy evaluation and zero-overlap comparison

Fourteen questions from `data/raw/policy_eval.csv` were evaluated.

| Evaluation checkpoint | Normal 10-word overlap | Zero overlap |
|---|---:|---:|
| Answerable questions with expected source in top 3 | 12/12 | 12/12 |
| Answerable questions with expected fact in answer | 12/12 | 10/12 |
| Unanswerable questions answered with the required fallback | 2/2 | 2/2 |
| Answerable questions missing the expected fact | 0/12 | 2/12 |

With zero overlap, two answerable questions failed the expected-fact check:

1. **Question:** “What delivery time makes a completed order eligible for a full refund?” The zero-overlap answer omitted the required phrase “more than 45 minutes.”
2. **Question:** “How soon is money returned when a payment fails but is deducted?” The zero-overlap answer omitted the expected fact “48 hours.”

With zero overlap, the relevant source document was still present in the top three retrieved chunks, but the selected chunk context did not provide the complete expected fact to the answer generator. Without overlapping words between adjacent chunks, a relevant detail near a chunk boundary may be omitted from the selected context. This explains why the expected-fact check failed for the delivery-time and payment-return questions even though their expected source documents were retrieved.

Both unanswerable questions — the mascot's favourite colour and the cricket World Cup final winner — received the correct fallback answer: `I could not find that information in the policy documents.`

The evaluation results are stored in `outputs/policy_eval_results.csv`.

### Refund-agent tools and rules

The agent has five tools:

1. `get_order` — looks up an order in `data/instamart.db`.
2. `search_policy` — retrieves relevant policy chunks.
3. `calculate_refund` — calculates a refund under the policy.
4. `issue_refund` — records an eligible automatic refund.
5. `escalate_to_human` — records an eligible refund requiring human approval.

Only completed orders can be refunded. A completed order delivered in more than 45 minutes is eligible for a full refund; delivery in 45 minutes or less results in no late-delivery refund. Refunds above INR 1,000 are escalated to a human and are not issued automatically. The simulated requests are treated as being raised within the seven-day policy window.

Refunds and escalations are local simulations and do not move real money or contact a real support lead.

### Seven agent scenarios

| Scenario | Expected outcome | Actual status | Refund / amount considered | Model calls | Tokens used |
|---|---|---|---:|---:|---:|
| S1 | REFUND | REFUND | INR 128.25 | 5 | 1,396 |
| S2 | REFUND | REFUND | INR 135.00 | 5 | 1,396 |
| S3 | ESCALATE | ESCALATE | INR 1,999.00 | 5 | 1,383 |
| S4 | DENIED | DENIED | INR 0.00 | 4 | 991 |
| S5 | DENIED | DENIED | INR 0.00 | 3 | 612 |
| S6 | NOT_FOUND | NOT_FOUND | INR 0.00 | 2 | 246 |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | INR 0.00 | 6 | 3,576 |

All seven scenarios matched their expected status and refund checks. S7 completed all six model calls and stopped with `FALLBACK_MAX_ITERATIONS`; it did not hit the default token limit of 6,000.

### Checkpoint recovery demonstration

Scenario S1 was deliberately interrupted after the second tool execution, after its checkpoint was saved. The agent was resumed from that checkpoint and completed the refund of INR 128.25.

The tools executed across the original run and resume, in order, were:

`get_order → search_policy → calculate_refund → issue_refund`

The four tool executions were not repeated, and the completed S1 run used five model calls. Checkpoints are stored in `state/`.

### Safety-limit demonstrations

| Demonstration | Expected status | Recorded status | Model calls | Tokens recorded |
|---|---|---|---:|---:|
| S1 with `token_limit=200` | `FALLBACK_TOKEN_LIMIT` | `FALLBACK_TOKEN_LIMIT` | 2 | 279 |
| S1 with `time_limit_seconds=0` | `FALLBACK_TIMEOUT` | `FALLBACK_TIMEOUT` | 0 | 0 |

The token-limit run recorded 279 response tokens over two model calls, exceeding the configured limit of 200. The agent returned FALLBACK_TOKEN_LIMIT before proceeding with another model call or tool execution. With `time_limit_seconds=0`, the agent returned FALLBACK_TIMEOUT without making a model call.

Agent outputs include `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `outputs/agent_scenario_results.csv`, `outputs/agent_safety_demos.csv`, and checkpoint files under `state/`.

### ReAct versus plan-and-execute

A ReAct agent alternates model calls, tool execution, and observations, allowing its next action to depend on actual results such as order status, delivery time, calculated refund, or a human-approval threshold.

In the measured runs, S1 took five model calls and 1,396 tokens; S5 took three calls and 612 tokens; S6 took two calls and 246 tokens. A plan-and-execute design could ask the model for a sequence once and replan only when an observation differs from expectations, potentially reducing repeated planning calls for routine cases. Its token use was not separately measured in this project.

An unexpected result such as `NOT_FOUND` or a cancelled order must still stop or change the plan safely. For this refund workflow, ReAct offers useful step-by-step adaptation, provided iteration, token, time, order-status, refund-amount, and human-approval checks remain enforced. This comparison is conceptual; no separate plan-and-execute benchmark was run.

## Limitations and Next Steps

- **Mock model:** The project uses a deterministic offline mock rather than a production LLM. Results do not establish real-world model quality.
- **Synthetic data:** The data is fictional, and results may not generalise to real customer behaviour or policies.
- **Data analysis:** Outliers and correlations describe the synthetic dataset; correlation does not establish causation.
- **Segmentation:** Cluster IDs are arbitrary labels, and segment actions are recommendations rather than experimentally validated interventions.
- **Lapse prediction:** Model metrics and thresholds should be tested with time-based validation, drift monitoring, and business-cost measures.
- **Premium targeting:** The Premium flag uses whole-window monetary value, including September. The retention list is illustrative and is not a leakage-free prospective evaluation of Premium targeting.
- **PII redaction:** Regex rules can miss unusual formats and sensitive information outside the tested patterns.
- **Policy retrieval:** TF-IDF is lexical, and the policy evaluation contains only 14 questions. Broader tests for paraphrases, ambiguity, and conflicting documents are needed.
- **Refund execution:** Refunds and escalations are local simulations. No real payment is issued, and no real support lead is contacted.
- **State and persistence:** Checkpoints and JSON Lines logs are local files. Production use would require stronger access controls, durable storage, concurrency protection, auditing, and recovery testing.
- **Production safeguards:** A deployed system would need authentication, authorization, human-approval workflows, monitoring, rate limits, and comprehensive security testing.
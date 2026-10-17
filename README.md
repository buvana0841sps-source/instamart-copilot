
# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a customer intelligence and support copilot for Instamart using Python, Pandas, NumPy, scikit-learn, SQLite, TF-IDF retrieval, and an offline mock language model.

The project is divided into four parts:

- **Part A — Data Quality and Analytics:** Audit, clean, validate, and store customer, product, and order data. Run SQL and Pandas analyses.
- **Part B — Customer Intelligence:** Build RFM customer segments, train a customer lapse prediction model, and prepare a retention list.
- **Part C — Support Ticket Triage:** Redact personally identifiable information, classify support tickets, apply validation and fallback logic, and track token usage and errors.
- **Part D — Policy and Refund Agent:** Retrieve policy documents, answer policy questions, execute refund tools, apply safety limits, and demonstrate resumable agent execution.

The project uses the supplied `generate_data.py` and `mock_llm.py` files. The mock model runs locally without an API key or network-based language-model service.

## Project Structure

```text
instamart-copilot/
├── data/
│   ├── raw/
│   ├── clean/
│   ├── policies/
│   └── instamart.db
├── outputs/
├── state/
├── sql/
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

Generated CSVs, JSON Lines logs, model files, reports, and agent checkpoints are stored under `outputs/` and `state/`.

## Reproduce the Project

Run these commands from the project root in Windows PowerShell.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the Python executable inside `.venv` to run the commands instead.

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Generate the data

```powershell
python generate_data.py
```

### 4. Part A — Data preparation and analytics

```powershell
python part_a_audit.py
python part_a_clean.py
python part_a_store.py
python part_a_sql.py
python part_a_analysis.py
python part_a_validate.py
```

### 5. Part B — Customer intelligence

```powershell
python part_b_segments.py
python part_b_lapse.py
```

### 6. Part C — Support ticket triage

```powershell
python part_c_triage.py
```

### 7. Part D — Policy retrieval and agent

```powershell
python part_d_policy.py
python part_d_eval.py
python part_d_agent.py
```

Run the commands from the project root so that relative paths such as `data/raw/` and `outputs/` resolve correctly.

## Part A — Data Quality and Analytics

### Raw-data audit

The initial audit reported:

| Audit item | Result |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Raw order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Pending-status records | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery times | 106 |
| Missing delivery times | 2,541 |
| Missing customer ages | 61 |
| Ages outside the 10–100 range | 6 |
| Different city spellings | 24 |
| Different gender spellings | 10 |
| Blank gender values | 30 |

### Cleaning rules

**Customers**
- Standardize city names using whitespace trimming and title case.
- Map gender variants to `male` or `female`; use `unknown` for unrecognized or blank values.
- Convert ages to numeric values.
- Treat ages below 10 or above 100 as missing.
- Impute missing ages using the median age within each city.
- Parse signup dates as dates.

**Products**
- Convert product prices to numeric values.

**Orders**
- Parse order timestamps.
- Remove exact duplicate rows.
- Sort by order timestamp and keep the latest record for each order ID.
- Convert quantity, discount, order amount, and delivery duration to numeric values.
- Recalculate missing or invalid order amounts using:

  `price_inr × quantity × (1 − discount_pct / 100)`

  Round recalculated amounts to two decimal places.
- Treat negative delivery durations as missing.
- Fill missing delivery times for completed orders using the median delivery time within the product category.
- Keep delivery time missing for cancelled orders.
- Save only the required clean order columns.

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
| Cancelled orders with missing delivery times | None |
| Distinct customer cities | 6 |
| Customers with missing ages | None |
| Completed-order revenue | INR 2,998,668.80 |

Final gender counts:

| Gender | Customers |
|---|---:|
| Female | 610 |
| Male | 560 |
| Unknown | 30 |

### SQL and Pandas checkpoints

The SQL script passed the following checks:

- Q1: 30 customers.
- Q2: 30 city-category result rows, with total revenue of INR 2,998,668.80.
- Q3: 11 products.
- Q4: 25 customers without a completed order.
- Q5: 550 customers above average spend; average spend approximately INR 2,552.06.
- Q6: 1,175 result rows; eight ranking positions differed between `RANK` and `ROW_NUMBER`.
- Q7: 183 days, ending with cumulative revenue of INR 2,998,668.80.

The Pandas analysis printed `ALL PANDAS ANALYSIS CHECKPOINTS PASSED`.

The validation script printed `ALL CHECKPOINTS PASSED`.

## Part B — Customer Intelligence

### RFM segmentation

The customer segmentation script created `outputs/customer_segments.csv`.

| Metric | Result |
|---|---:|
| Customers in the RFM analysis | 1,175 |
| Mean purchase frequency | 9.46 |
| Mean monetary value | INR 2,552.06 |
| Median recency | 19 days |
| Minimum recency | 0 days |
| Maximum recency | 180 days |

K-means models with K = 2 through K = 8 were compared.

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1,820.761 | 0.465 |
| 3 | 1,268.235 | 0.445 |
| 4 | 743.071 | 0.526 |
| 5 | 617.367 | 0.480 |
| 6 | 503.171 | 0.482 |
| 7 | 408.460 | 0.416 |
| 8 | 366.503 | 0.412 |

K = 4 produced the highest silhouette score among the tested configurations.

### Four customer segments

| Segment | Customers | Mean recency | Mean frequency | Mean monetary value | Label |
|---:|---:|---:|---:|---:|---|
| 0 | 436 | 28.86 | 3.77 | INR 1,057.62 | Steady Shoppers |
| 1 | 138 | 24.12 | 6.77 | INR 7,129.70 | Premium Spenders |
| 2 | 419 | 8.77 | 19.40 | INR 3,410.02 | Frequent Loyalists |
| 3 | 182 | 109.04 | 2.20 | INR 685.99 | At-Risk Shoppers |

The first two principal components explained approximately 85.23% of the variance:

- PC1: 63.43%
- PC2: 21.80%

### Customer lapse prediction

The lapse-model script created `outputs/lapse_model.joblib` and `outputs/retention_list.csv`.

| Metric | Result |
|---|---:|
| Lapse cohort | 1,156 customers |
| Lapsed customers | 443 |
| Lapse rate | 38.32% |
| Training set | 924 |
| Test set | 232 |

The baseline model achieved an accuracy of 0.6164 and an F1 score of 0.

**Logistic Regression**
- Best C: 100
- Best cross-validation F1: 0.6506
- Test precision: 0.6444
- Test recall: 0.6517
- Test F1: 0.6480
- Confusion matrix: `[[111, 32], [31, 58]]`

**Random Forest**
- Best maximum depth: 8
- Best minimum leaf size: 10
- Best cross-validation F1: 0.6811
- Test precision: 0.6591
- Test recall: 0.6517
- Test F1: 0.6554
- Confusion matrix: `[[113, 30], [31, 58]]`

The Random Forest was selected for the saved model.

The leading Random Forest feature importances were:

| Feature | Importance |
|---|---:|
| Frequency | 0.279589 |
| Monetary value | 0.271176 |
| Recency | 0.115149 |
| Average delivery time | 0.061286 |
| Cancellation rate | 0.052325 |

Oversampling improved the Logistic Regression model's recall to 0.8427 and F1 to 0.7009, with precision of 0.6000. The result illustrates the trade-off between finding more at-risk customers and increasing false positives.

**Retention-list caveat:** The premium flag was assigned using monetary value over the entire observed window, including September. The resulting retention list is illustrative rather than a leakage-free, prospective validation of premium customer targeting. Only one of the top ten listed customers was labelled lapsed in that analysis.

## Part C — Support Ticket Triage

The triage script uses `mock_llm.py` to test an offline support-ticket classification workflow.

Features include:

- Email, phone-number, and order-ID redaction.
- Zero-shot and few-shot classification.
- JSON output validation.
- Keyword fallback handling.
- Empty-input and long-input guardrails.
- Retry logic for simulated transient model failures.
- Token and model-call accounting.
- CSV outputs preserving source-ticket order.

### Observed checkpoints

| Metric | Result |
|---|---:|
| Input tickets | 164 |
| Email-pattern matches redacted | 55 |
| Phone-pattern matches redacted | 91 |
| Order-ID matches redacted | 132 |
| Total redaction matches | 278 |
| Valid JSON classifications | 135 |
| Keyword fallback classifications | 12 |
| Invalid outputs | 9 |
| Empty tickets | 4 |
| Long tickets | 4 |
| Successfully classified tickets | 147 |

The few-shot evaluation reported 100% accuracy on its evaluated batch.

The script generated:

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

The budget demo exercised the configured model-token budget, and the triage workflow handled malformed output, empty input, and oversized tickets through validation and fallback paths.

The redaction counts represent regex matches, not a guarantee that all possible personally identifiable information was removed.

## Part D — Policy Retrieval and Refund Agent

### Policy chunking and retrieval

Policy documents are split into 40-word chunks, using an overlap of 10 words by default. TF-IDF vectors and cosine similarity rank the most relevant chunks.

| Policy | Number of chunks |
|---|---:|
| `refund_policy.txt` | 4 |
| `delivery_policy.txt` | 3 |
| `escalation_policy.txt` | 3 |
| `payment_policy.txt` | 3 |
| `privacy_policy.txt` | 2 |
| **Total** | **15** |

The query `refund policy late delivery` retrieved three chunks from `refund_policy.txt`.

### Policy evaluation

Fourteen questions were evaluated using `data/raw/policy_eval.csv`.

| Evaluation checkpoint | Result |
|---|---:|
| Total questions | 14 |
| Answerable questions with correct source in top 3 | 12/12 |
| Answerable questions with expected fact in answer | 12/12 |
| Unanswerable questions handled with fallback | 2/2 |
| Zero-overlap retrieval failures | 2/14 |

The two unanswerable questions concern the company mascot's favourite colour and the winner of a cricket World Cup final. The policy assistant correctly declined to invent answers.

With zero chunk overlap, these same two unanswerable questions were recorded as retrieval failures. The detailed evaluation was saved to `outputs/policy_eval_results.csv`.

### ReAct agent tools

The agent registers five tools:

1. `get_order` — look up an order.
2. `search_policy` — retrieve relevant policy chunks.
3. `calculate_refund` — calculate a full refund under the late-delivery rule.
4. `issue_refund` — issue a validated refund of INR 1,000 or less.
5. `escalate_to_human` — send an eligible refund above INR 1,000 to a human support lead.

The tool implementations validate order existence, order status, delivery duration, refund amount, and the human-approval threshold.

### Agent scenario results

| Scenario | Expected outcome | Actual outcome | Refund | Model calls |
|---|---|---|---:|---:|
| S1 | REFUND | REFUND | INR 128.25 | 5 |
| S2 | REFUND | REFUND | INR 135.00 | 5 |
| S3 | ESCALATE | ESCALATE | INR 1,999.00 | 5 |
| S4 | DENIED | DENIED | INR 0.00 | 4 |
| S5 | DENIED | DENIED | INR 0.00 | 3 |
| S6 | NOT_FOUND | NOT_FOUND | INR 0.00 | 2 |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | INR 0.00 | 6 |

All seven scenarios passed their outcome and refund-amount checks.

### Resumable checkpoint demonstration

Scenario S1 was deliberately interrupted after the second tool execution. The agent saved its checkpoint, resumed from the saved state, and completed the refund flow.

The expected tool sequence was:

`get_order → search_policy → calculate_refund → issue_refund`

The resumed scenario completed with a refund of INR 128.25 in five model calls. The earlier tool executions were not repeated.

Checkpoint files were saved under `state/`.

### Safety demonstrations

| Demo | Expected result | Actual result |
|---|---|---|
| Token limit of 200 | `FALLBACK_TOKEN_LIMIT` | `FALLBACK_TOKEN_LIMIT` |
| Time limit of 0 seconds | `FALLBACK_TIMEOUT` | `FALLBACK_TIMEOUT` |

Both safety demonstrations passed.

### Agent output files

- `outputs/agent_scenario_results.csv` — expected and actual scenario outcomes.
- `outputs/agent_safety_demos.csv` — safety-limit results.
- `outputs/refund_ledger.jsonl` — refund ledger entries.
- `outputs/agent_runs.jsonl` — model/tool execution and fallback logs.
- `state/` — agent checkpoints for resuming runs.

### ReAct versus Plan-Execute

A ReAct agent alternates between reasoning, tool execution, and observing tool results. This allows the next step to depend on the order status, delivery duration, calculated refund, or need for human approval. It is useful when different requests require different tool sequences, as in these seven scenarios.

A Plan-Execute agent would create a plan first and execute its steps afterward. A fixed plan can be easier to audit and may reduce repeated model decisions, but it can be less adaptive when a tool result changes the next required action.

The ReAct implementation completed S1 in five model calls and used checkpoint recovery to resume after an intentional interruption. The added flexibility comes with model-call overhead, latency, and the risk of loops. Iteration, token, and time limits are therefore important safeguards. Both approaches would still need reliable tool validation and explicit refund authorization rules.

## Limitations and Next Steps

- **Mock model:** The project uses a deterministic offline mock rather than a production LLM. Results do not establish real-world LLM quality.
- **Synthetic/local data:** Findings apply to the provided project dataset and may not generalize to real customers.
- **Retrieval:** TF-IDF and cosine similarity are lexical retrieval methods. A production system could compare embedding-based retrieval and reranking.
- **Evaluation size:** The policy evaluation contains 14 questions. Broader tests with paraphrases, ambiguous requests, and conflicting documents are needed.
- **Lapse prediction:** The model's performance and thresholds should be validated with time-based splits, drift monitoring, and business-cost measures.
- **Premium targeting:** The whole-window premium label introduces look-ahead information into the illustrative retention list. A prospective evaluation should define premium status using information available before the prediction date.
- **PII redaction:** Regex redaction may miss unusual formats and non-text sensitive information. Redaction should be tested against a broader privacy test set.
- **Refund execution:** Refunds and escalations are local simulations. The code does not connect to a payment service, issue real money, or obtain approval from an actual support lead.
- **State and persistence:** The checkpoint files and JSON Lines logs are local files. Production use would require stronger access controls, durable storage, concurrency protection, audit policies, and recovery testing.
- **Production safeguards:** A deployed system would need authentication, authorization, human-approval workflows, monitoring, rate limits, and comprehensive security testing.

## Conclusion

The project demonstrates an end-to-end customer intelligence and support workflow: data cleaning and validation, SQL and Pandas analysis, customer segmentation and lapse prediction, ticket triage with privacy controls, policy retrieval, and a tool-using refund agent with resumable checkpoints and safety limits.

The results documented here are the observed outputs from the local project runs described above.


import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import mock_llm


# =========================================================
# 1. FILES AND SETTINGS
# =========================================================

RAW = Path("data/raw")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3

ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")

PII_PATTERNS = [EMAIL_RE, PHONE_RE, ORDER_RE]


# =========================================================
# 2. READ TICKETS AND CUSTOMER SEGMENTS
# =========================================================

# keep_default_na=False preserves empty ticket text as "".
tickets = pd.read_csv(
    RAW / "tickets.csv",
    keep_default_na=False
)

segments = pd.read_csv(
    OUTPUTS / "customer_segments.csv"
)

# Standardise the Premium flag after reading the CSV.
segments["is_premium"] = (
    segments["is_premium"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)

segment_lookup = (
    segments.set_index("customer_id")[
        ["segment_name", "is_premium"]
    ].to_dict("index")
)

print("=" * 65)
print("INSTAMART SUPPORT TICKET TRIAGE")
print("=" * 65)
print("Tickets loaded:", len(tickets))
print("Customer segments loaded:", len(segments))


# =========================================================
# 3. REDACT PERSONAL INFORMATION
# =========================================================

def redact_text(text):
    """Redact email, phone, then order ID in the required order."""

    text = str(text)

    text, email_count = EMAIL_RE.subn("[EMAIL]", text)
    text, phone_count = PHONE_RE.subn("[PHONE]", text)
    text, order_count = ORDER_RE.subn("[ORDER_ID]", text)

    return text, {
        "emails": email_count,
        "phones": phone_count,
        "order_ids": order_count
    }


redacted_texts = []
redaction_totals = {
    "emails": 0,
    "phones": 0,
    "order_ids": 0
}

for ticket_text in tickets["text"]:
    redacted, counts = redact_text(ticket_text)
    redacted_texts.append(redacted)

    for key in redaction_totals:
        redaction_totals[key] += counts[key]

# Verify that the required patterns find nothing after redaction.
for redacted in redacted_texts:
    for pattern in PII_PATTERNS:
        assert pattern.search(redacted) is None, (
            "Personal information remains after redaction."
        )

tickets["redacted_text"] = redacted_texts

print("\n" + "=" * 65)
print("REDACTION AUDIT")
print("=" * 65)
print("Email substitutions:", redaction_totals["emails"])
print("Phone substitutions:", redaction_totals["phones"])
print("Order ID substitutions:", redaction_totals["order_ids"])
print(
    "Total substitutions:",
    sum(redaction_totals.values())
)
print("PII matches remaining after redaction: 0")

assert redaction_totals == {
    "emails": 55,
    "phones": 91,
    "order_ids": 132
}
assert sum(redaction_totals.values()) == 278


# =========================================================
# 4. CREATE THE TWO SYSTEM PROMPTS
# =========================================================

ZERO_SHOT_PROMPT = """
CLASSIFY

You are a support ticket classification assistant.
Classify the ticket into exactly one of these allowed categories:
billing, delivery, technical, fraud.

Return JSON only, using exactly this structure:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
Give a short reason grounded in the ticket text.
Do not include Markdown, code fences, or any other text.
""".strip()


FEW_SHOT_PROMPT = """
CLASSIFY

You are a support ticket classification assistant.
Classify the ticket into exactly one of these allowed categories:
billing, delivery, technical, fraud.

Return JSON only, using exactly this structure:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
Give a short reason grounded in the ticket text.
Do not include Markdown, code fences, or any other text.

EXAMPLES:
Example 1 - Fraud:
Ticket: "I found an unauthorized transaction on my account."
Reply: {"category": "fraud", "reason": "unauthorized transaction"}

Example 2 - Technical:
Ticket: "The app crashes and shows an error when I open the cart."
Reply: {"category": "technical", "reason": "app crash and error"}

Example 3 - Billing:
Ticket: "I was charged twice and need the extra payment refunded."
Reply: {"category": "billing", "reason": "duplicate charge"}

Example 4 - Delivery:
Ticket: "My delivery is late and the rider has not arrived."
Reply: {"category": "delivery", "reason": "late delivery"}
""".strip()

assert "CLASSIFY" in ZERO_SHOT_PROMPT
assert "EXAMPLES:" not in ZERO_SHOT_PROMPT
assert "CLASSIFY" in FEW_SHOT_PROMPT
assert "EXAMPLES:" in FEW_SHOT_PROMPT

for prompt in [ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT]:
    for category in ALLOWED_CATEGORIES:
        assert category in prompt
    assert '{"category": "...", "reason": "..."}' in prompt


# =========================================================
# 5. VALIDATE THE MODEL'S REPLY
# =========================================================

def validate_reply(reply_text):
    """
    Return (status, category).

    Valid JSON with a disallowed category is flagged invalid.
    Keyword extraction is used only if JSON parsing fails.
    """

    try:
        parsed = json.loads(reply_text)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(reply_text).lower())

        found_categories = {
            word for word in words
            if word in ALLOWED_CATEGORIES
        }

        if len(found_categories) == 1:
            category = next(iter(found_categories))
            return "classified_keyword", category

        return "flagged_invalid_output", ""

    # JSON parsed successfully. Never guess if its category is invalid.
    if not isinstance(parsed, dict):
        return "flagged_invalid_output", ""

    category = parsed.get("category")

    if category in ALLOWED_CATEGORIES:
        return "classified_json", category

    return "flagged_invalid_output", ""


# =========================================================
# 6. ROUTING RULES
# =========================================================

HUMAN_REVIEW_STATUSES = {
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted"
}


def assign_route(status, category, customer_id):
    """Assign a route and priority using the capstone rules."""

    segment = segment_lookup.get(customer_id, {})
    segment_name = segment.get("segment_name", "")
    is_premium = bool(segment.get("is_premium", False))

    if status in HUMAN_REVIEW_STATUSES:
        return segment_name, "NORMAL", "human_review"

    if category == "fraud":
        return segment_name, "HIGH", "security_team"

    if is_premium:
        return segment_name, "HIGH", "premium_desk"

    return segment_name, "NORMAL", "support_queue"


# =========================================================
# 7. RUN THE PIPELINE
# =========================================================

def run_pipeline(system_prompt, token_budget=None, run_name="run"):
    """
    Process tickets in source-file order.

    Running token usage increases by response.usage.total_tokens
    after each successful model response. A transient error has
    no response usage to add.

    The user's model message contains the redacted ticket text
    and nothing else.
    """

    # Reset immediately before each independent full run.
    mock_llm.reset()

    retry_rng = random.Random(42)

    total_tokens = 0
    total_calls = 0
    transient_errors = 0
    records = []

    print("\n" + "=" * 65)
    print("STARTING:", run_name)
    print("Token budget:", token_budget if token_budget is not None else "None")
    print("=" * 65)

    for row_number, ticket in tickets.iterrows():
        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]
        raw_text = ticket["text"]
        redacted_text = ticket["redacted_text"]
        true_category = ticket["true_category"]

        status = ""
        category = ""
        ticket_tokens = 0

        # Guardrail 1: empty text is never sent to the model.
        if not raw_text.strip():
            status = "skipped_empty"

        # Guardrail 2: long text is never sent to the model.
        elif mock_llm.est_tokens(raw_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"

        else:
            # A prompt must contain only the redacted text as user content.
            assert all(
                pattern.search(redacted_text) is None
                for pattern in PII_PATTERNS
            )

            # One model call per ticket, with retries if necessary.
            retry_number = 0

            while True:
                # Check the budget before EVERY actual model call.
                if (
                    token_budget is not None
                    and total_tokens >= token_budget
                ):
                    status = "budget_exhausted"
                    break

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": redacted_text
                    }
                ]

                # Confirm that the model receives the exact redacted text,
                # and not the original personal information.
                assert messages[-1]["content"] == redacted_text
                assert len(messages) == 2

                for pattern in PII_PATTERNS:
                    assert pattern.search(messages[-1]["content"]) is None

                total_calls += 1

                try:
                    response = mock_llm.chat(messages)

                    call_tokens = response.usage.total_tokens
                    total_tokens += call_tokens
                    ticket_tokens += call_tokens

                    reply_text = response.choices[0].message.content
                    status, category = validate_reply(reply_text)

                    break

                except mock_llm.TransientLLMError as error:
                    transient_errors += 1

                    if retry_number >= MAX_RETRIES:
                        # The supplied mock should recover within retries.
                        # If a permanent failure occurs, send it to review.
                        status = "flagged_invalid_output"
                        category = ""
                        print(
                            f"Permanent model failure for {ticket_id}: "
                            f"{error}"
                        )
                        break

                    wait = (
                        BASE_DELAY
                        * (2 ** retry_number)
                        * retry_rng.uniform(0.9, 1.15)
                    )

                    print(
                        f"RETRY ticket_id={ticket_id} "
                        f"retry={retry_number} wait={wait:.4f}s"
                    )

                    time.sleep(wait)
                    retry_number += 1

        segment_name, priority, route = assign_route(
            status, category, customer_id
        )

        records.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": ticket_tokens,
            "true_category": true_category
        })

    result = pd.DataFrame(records)

    # The original order must be preserved.
    assert result["ticket_id"].tolist() == tickets["ticket_id"].tolist()
    assert len(result) == len(tickets)

    classified_mask = result["status"].isin(
        ["classified_json", "classified_keyword"]
    )
    classified = result[classified_mask]

    if len(classified) > 0:
        accuracy = (
            classified["category"] == classified["true_category"]
        ).mean()
    else:
        accuracy = 0.0

    print("\nRUN SUMMARY:", run_name)
    print("Tickets processed:", len(result))
    print("Status counts:")
    print(result["status"].value_counts().sort_index().to_string())
    print("Tickets classified:", len(classified))
    print("Accuracy over classified tickets:", f"{accuracy:.4%}")
    print("Total response tokens:", total_tokens)
    print(
        "Tokens per classified ticket:",
        round(total_tokens / len(classified), 2)
        if len(classified) else 0
    )
    print("Model call attempts, including retries:", total_calls)
    print("Transient errors recovered/encountered:", transient_errors)

    return result, {
        "classified": len(classified),
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_classified": (
            total_tokens / len(classified) if len(classified) else 0
        ),
        "total_calls": total_calls,
        "transient_errors": transient_errors
    }


# =========================================================
# 8. SAVE OUTPUTS WITH THE REQUIRED COLUMNS
# =========================================================

OUTPUT_COLUMNS = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]


def save_triage_csv(result, filename):
    output_path = OUTPUTS / filename
    result[OUTPUT_COLUMNS].to_csv(output_path, index=False)
    print("Saved:", output_path)
    assert len(pd.read_csv(output_path, keep_default_na=False)) == 164


# =========================================================
# 9. ZERO-SHOT AND FEW-SHOT COMPARISON (NO BUDGET)
# =========================================================

zero_shot_result, zero_stats = run_pipeline(
    ZERO_SHOT_PROMPT,
    token_budget=None,
    run_name="Zero-shot comparison (no token budget)"
)

few_shot_result, few_stats = run_pipeline(
    FEW_SHOT_PROMPT,
    token_budget=None,
    run_name="Few-shot comparison (no token budget)"
)

# Expected output validation counts for the supplied mock.
for name, result in [
    ("Zero-shot", zero_shot_result),
    ("Few-shot", few_shot_result)
]:
    counts = result["status"].value_counts().to_dict()

    print(f"\n{name} validation counts:")
    print(counts)

    assert counts.get("classified_json", 0) == 135
    assert counts.get("classified_keyword", 0) == 12
    assert counts.get("flagged_invalid_output", 0) == 9
    assert counts.get("skipped_empty", 0) == 4
    assert counts.get("flagged_long", 0) == 4

    assert result["status"].isin(
        ["classified_json", "classified_keyword",
         "flagged_invalid_output", "skipped_empty", "flagged_long"]
    ).all()

assert zero_stats["classified"] == 147
assert few_stats["classified"] == 147
assert zero_stats["transient_errors"] == 19
assert few_stats["transient_errors"] == 19


# =========================================================
# 10. ZERO-SHOT CONFUSION TABLE
# =========================================================

zero_classified = zero_shot_result[
    zero_shot_result["status"].isin(
        ["classified_json", "classified_keyword"]
    )
]

confusion = pd.crosstab(
    zero_classified["true_category"],
    zero_classified["category"]
).reindex(
    index=ALLOWED_CATEGORIES,
    columns=ALLOWED_CATEGORIES,
    fill_value=0
)

print("\n" + "=" * 65)
print("ZERO-SHOT CONFUSION TABLE")
print("Rows = true category; columns = predicted category")
print("=" * 65)
print(confusion.to_string())

print("\nCOMPARISON")
print(
    "Zero-shot accuracy:",
    f"{zero_stats['accuracy']:.4%}"
)
print(
    "Few-shot accuracy:",
    f"{few_stats['accuracy']:.4%}"
)
print("Zero-shot total tokens:", zero_stats["total_tokens"])
print("Few-shot total tokens:", few_stats["total_tokens"])
print(
    "Zero-shot tokens/classified ticket:",
    round(zero_stats["tokens_per_classified"], 2)
)
print(
    "Few-shot tokens/classified ticket:",
    round(few_stats["tokens_per_classified"], 2)
)


# =========================================================
# 11. TOKEN-BUDGET DEMONSTRATION
# =========================================================

# The assignment asks for a zero-shot budget demo.
# Use 6000 tokens first; report the actual exhausted count.
BUDGET_DEMO = 6000

budget_result, budget_stats = run_pipeline(
    ZERO_SHOT_PROMPT,
    token_budget=BUDGET_DEMO,
    run_name="Zero-shot token-budget demonstration"
)

save_triage_csv(
    budget_result,
    "triage_budget_demo.csv"
)

budget_exhausted_count = int(
    (budget_result["status"] == "budget_exhausted").sum()
)

print("\nBudget demo limit:", BUDGET_DEMO)
print("Tickets marked budget_exhausted:", budget_exhausted_count)

if budget_exhausted_count == 0:
    print(
        "WARNING: No tickets exhausted this budget. "
        "Try a lower BUDGET_DEMO value, such as 5000, "
        "then rerun the program."
    )
else:
    print("Budget demonstration passed: at least one ticket was stopped.")

assert budget_exhausted_count > 0


# =========================================================
# 12. FINAL FEW-SHOT ROUTING RUN
# =========================================================

final_result, final_stats = run_pipeline(
    FEW_SHOT_PROMPT,
    token_budget=100000,
    run_name="Final few-shot routing run"
)

save_triage_csv(
    final_result,
    "triage_results.csv"
)

# Check final routing rules.
for _, row in final_result.iterrows():
    status = row["status"]
    category = row["category"]
    premium = segment_lookup.get(
        row["customer_id"], {}
    ).get("is_premium", False)

    if status in HUMAN_REVIEW_STATUSES:
        assert row["route"] == "human_review"
        assert row["priority"] == "NORMAL"
    elif category == "fraud":
        assert row["route"] == "security_team"
        assert row["priority"] == "HIGH"
    elif premium:
        assert row["route"] == "premium_desk"
        assert row["priority"] == "HIGH"
    else:
        assert row["route"] == "support_queue"
        assert row["priority"] == "NORMAL"

assert len(final_result) == 164
assert final_stats["classified"] == 147
assert abs(final_stats["accuracy"] - 1.0) < 1e-9

print("\n" + "=" * 65)
print("PART C TRIAGE PIPELINE COMPLETED")
print("=" * 65)
print("Final output: outputs/triage_results.csv")
print("Budget demo: outputs/triage_budget_demo.csv")
print("Both files contain 164 tickets in source-file order.")

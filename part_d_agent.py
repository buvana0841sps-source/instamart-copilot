
"""Part D: ReAct refund agent with tools, safety limits and checkpoints."""

from pathlib import Path
import contextlib
import io
import json
import time

import pandas as pd
import mock_llm

# Import retrieval without displaying the policy setup banner.
with contextlib.redirect_stdout(io.StringIO()):
    from part_d_policy import retrieve


# =========================================================
# 1. PATHS AND DATA
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
STATE_DIR = BASE_DIR / "state"

ORDERS_FILE = DATA_DIR / "clean" / "orders.csv"
SCENARIOS_FILE = DATA_DIR / "raw" / "agent_scenarios.csv"

LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"
RUN_LOG_FILE = OUTPUT_DIR / "agent_runs.jsonl"
RESULTS_FILE = OUTPUT_DIR / "agent_scenario_results.csv"
SAFETY_FILE = OUTPUT_DIR / "agent_safety_demos.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

if not ORDERS_FILE.exists():
    raise FileNotFoundError(f"Orders file not found: {ORDERS_FILE}")

orders_df = pd.read_csv(ORDERS_FILE, keep_default_na=False)
orders_df["order_id"] = orders_df["order_id"].astype(str).str.strip()

if orders_df["order_id"].duplicated().any():
    raise ValueError("Cleaned orders contain duplicate order IDs.")


def get_order_record(order_id):
    """Look up one order in the cleaned order dataset."""
    order_id = str(order_id).strip()
    matches = orders_df[orders_df["order_id"] == order_id]

    if matches.empty:
        return None

    row = matches.iloc[0]

    try:
        amount = float(row["order_amount"])
    except (TypeError, ValueError):
        amount = None

    try:
        delivery = (
            float(row["delivery_minutes"])
            if str(row["delivery_minutes"]).strip()
            else None
        )
    except (TypeError, ValueError):
        delivery = None

    return {
        "order_id": order_id,
        "customer_id": str(row["customer_id"]),
        "product_id": str(row["product_id"]),
        "status": str(row["status"]).strip().lower(),
        "order_amount": amount,
        "delivery_minutes": delivery,
    }


# =========================================================
# 2. JSONL AND CHECKPOINT HELPERS
# =========================================================

def append_jsonl(path, record):
    """Append one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    """Read a JSON Lines file."""
    records = []

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path}, line {line_number}"
                    ) from exc

    return records


def save_checkpoint(path, state):
    """Atomically save the resumable agent state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")

    temp_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def append_step(state, step):
    """Save a numbered entry in the agent's step history."""
    step["step"] = len(state["steps"]) + 1
    state["steps"].append(step)


class SimulatedCrash(Exception):
    """Intentional interruption used to test checkpoint recovery."""


# =========================================================
# 3. THE FIVE REQUIRED TOOLS
# =========================================================

def tool_get_order(args, scenario_id):
    order_id = str(args.get("order_id", "")).strip()

    if not order_id:
        return {"error": "Missing order ID."}

    order = get_order_record(order_id)

    if order is None:
        return {
            "error": "Order not found.",
            "order_id": order_id,
        }

    return order


def tool_search_policy(args, scenario_id):
    query = str(args.get("query", "")).strip()

    if not query:
        return {"error": "A policy query is required."}

    results = retrieve(query, k=3)

    return {
        "query": query,
        "results": [
            {
                "source": item["source"],
                "score": round(float(item["score"]), 4),
                "text": item["text"],
            }
            for item in results
        ],
    }


def tool_calculate_refund(args, scenario_id):
    """Calculate a full refund for an order delivered after 45 minutes."""
    try:
        amount = float(args.get("order_amount"))
        delivery = float(args.get("delivery_minutes"))
    except (TypeError, ValueError):
        return {
            "error": "A valid amount and delivery time are required.",
            "refund_inr": 0.0,
        }

    if amount < 0 or delivery < 0:
        return {
            "error": "Negative amounts or delivery times are invalid.",
            "refund_inr": 0.0,
        }

    refund = round(amount, 2) if delivery > 45 else 0.0

    return {
        "order_amount": round(amount, 2),
        "delivery_minutes": delivery,
        "refund_inr": refund,
        "reason": (
            "Eligible late-delivery refund"
            if refund > 0
            else "Delivery was within 45 minutes"
        ),
    }


def tool_issue_refund(args, scenario_id):
    """Issue a validated refund of INR 1000 or less."""
    order_id = str(args.get("order_id", "")).strip()

    try:
        refund = round(float(args.get("refund_inr")), 2)
    except (TypeError, ValueError):
        return {"error": "A valid refund amount is required."}

    order = get_order_record(order_id)

    if order is None:
        return {"error": "Order not found.", "order_id": order_id}

    if order["status"] != "completed":
        return {"error": "Only completed orders can be refunded."}

    if (
        order["delivery_minutes"] is None
        or order["delivery_minutes"] <= 45
    ):
        return {"error": "Order does not qualify for a late-delivery refund."}

    if order["order_amount"] is None:
        return {"error": "The order amount is unavailable."}

    if refund <= 0 or abs(refund - order["order_amount"]) > 0.01:
        return {"error": "Refund does not match the order amount."}

    if refund > 1000:
        return {
            "error": "Refunds above INR 1000 require human approval.",
            "requires_escalation": True,
        }

    # Avoid issuing a duplicate refund for this scenario and order.
    for entry in read_jsonl(LEDGER_FILE):
        if (
            entry.get("scenario_id") == scenario_id
            and entry.get("order_id") == order_id
            and entry.get("status") == "issued"
        ):
            return {
                "status": "already_issued",
                "order_id": order_id,
                "refund_inr": entry["refund_inr"],
                "ledger_entry_written": False,
            }

    ledger_entry = {
        "scenario_id": scenario_id,
        "order_id": order_id,
        "customer_id": order["customer_id"],
        "refund_inr": refund,
        "status": "issued",
        "reason": "Eligible late-delivery refund",
    }

    append_jsonl(LEDGER_FILE, ledger_entry)

    return {
        "status": "issued",
        "order_id": order_id,
        "refund_inr": refund,
        "ledger_entry_written": True,
    }


def tool_escalate_to_human(args, scenario_id):
    """Record eligible refunds above INR 1000 for human approval."""
    order_id = str(args.get("order_id", "")).strip()

    try:
        refund = round(float(args.get("refund_inr")), 2)
    except (TypeError, ValueError):
        return {"error": "A valid refund amount is required."}

    order = get_order_record(order_id)

    if order is None:
        return {"error": "Order not found.", "order_id": order_id}

    if order["status"] != "completed":
        return {"error": "Only completed orders can be considered."}

    if (
        order["delivery_minutes"] is None
        or order["delivery_minutes"] <= 45
    ):
        return {"error": "Order does not qualify for a late-delivery refund."}

    if order["order_amount"] is None:
        return {"error": "The order amount is unavailable."}

    if refund <= 1000 or abs(refund - order["order_amount"]) > 0.01:
        return {
            "error": "Human escalation requires an eligible refund "
                     "above INR 1000."
        }

    # Avoid duplicate escalation entries.
    for entry in read_jsonl(LEDGER_FILE):
        if (
            entry.get("scenario_id") == scenario_id
            and entry.get("order_id") == order_id
            and entry.get("status") == "escalated"
        ):
            return {
                "status": "already_escalated",
                "order_id": order_id,
                "refund_inr": refund,
                "ledger_entry_written": False,
            }

    ledger_entry = {
        "scenario_id": scenario_id,
        "order_id": order_id,
        "customer_id": order["customer_id"],
        "refund_inr": refund,
        "status": "escalated",
        "reason": "Awaiting human approval for a refund above INR 1000",
    }

    append_jsonl(LEDGER_FILE, ledger_entry)

    return {
        "status": "escalated",
        "order_id": order_id,
        "refund_inr": refund,
        "message": "Sent to a human support lead for approval.",
        "ledger_entry_written": True,
    }


TOOL_MAP = {
    "get_order": tool_get_order,
    "search_policy": tool_search_policy,
    "calculate_refund": tool_calculate_refund,
    "issue_refund": tool_issue_refund,
    "escalate_to_human": tool_escalate_to_human,
}


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order using its order ID.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Retrieve relevant policy chunks.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate a valid late-delivery refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"},
                },
                "required": ["order_amount", "delivery_minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue an eligible refund of INR 1000 or less.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Send an eligible refund above INR 1000 for review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
]


# =========================================================
# 4. REACT LOOP, STATUS, STEPS AND LOGGING
# =========================================================

SYSTEM_PROMPT = """
AGENT
You are an Instamart customer-support refund agent.
Use tools to verify orders and follow policy.
Only completed orders can receive a refund.
An order delivered in more than 45 minutes qualifies for a full refund.
An order delivered in 45 minutes or less does not qualify.
Refunds up to INR 1000 may be issued automatically.
Refunds above INR 1000 must be escalated to a human support lead.
Never invent an order or skip required checks.
"""


def estimated_message_tokens(messages):
    total = 0

    for message in messages:
        content = message.get("content") or ""

        if isinstance(content, str):
            total += mock_llm.est_tokens(content)
        else:
            total += mock_llm.est_tokens(json.dumps(content, default=str))

        if message.get("tool_calls"):
            total += mock_llm.est_tokens(
                json.dumps(message["tool_calls"], default=str)
            )

    return total


def classify_outcome(final_answer):
    text = str(final_answer).strip().upper()

    for prefix, status in [
        ("REFUND:", "REFUND"),
        ("ESCALATE:", "ESCALATE"),
        ("DENIED:", "DENIED"),
        ("NOT_FOUND:", "NOT_FOUND"),
        ("NEEDS_INFO:", "NEEDS_INFO"),
        ("FALLBACK_TOKEN_LIMIT", "FALLBACK_TOKEN_LIMIT"),
        ("FALLBACK_TIMEOUT", "FALLBACK_TIMEOUT"),
        ("FALLBACK_MAX_ITERATIONS", "FALLBACK_MAX_ITERATIONS"),
    ]:
        if text.startswith(prefix):
            return status

    return "OTHER"


def result_refund_amount(tool_executions):
    for event in reversed(tool_executions):
        if event["tool"] == "calculate_refund":
            try:
                return round(float(event["result"].get("refund_inr", 0)), 2)
            except (TypeError, ValueError):
                return 0.0

    return 0.0


def agent_result(state, final_answer):
    """Build the run record and required acceptance-criteria fields."""
    status = classify_outcome(final_answer)

    return {
        "scenario_id": state["scenario_id"],
        "status": status,
        "outcome": status,
        "final_answer": str(final_answer),
        "refund_inr": result_refund_amount(state["tool_executions"]),
        "llm_calls": state["model_calls"],
        "tokens_used": state["token_usage"],
        "model_calls": state["model_calls"],
        "token_usage": state["token_usage"],
        "steps": state["steps"],
        "tool_executions": state["tool_executions"],
        "tool_names": [
            event["tool"] for event in state["tool_executions"]
        ],
    }


def log_agent_run(result):
    """Write one complete JSONL record per completed run."""
    append_jsonl(
        RUN_LOG_FILE,
        {
            "scenario_id": result["scenario_id"],
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"],
            "tool_names": result["tool_names"],
            "tool_executions": result["tool_executions"],
            "refund_inr": result["refund_inr"],
        },
    )


def run_agent(
    scenario_id,
    user_message,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    checkpoint_path=None,
    resume=False,
    crash_after_tool=None,
):
    """Run the ReAct agent with limits and checkpoint recovery."""
    if checkpoint_path is None:
        checkpoint_path = STATE_DIR / f"{scenario_id}_checkpoint.json"

    started_at = time.monotonic()

    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint missing at {checkpoint_path}"
            )

        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        if state["scenario_id"] != scenario_id:
            raise ValueError("Checkpoint scenario ID does not match.")
    else:
        state = {
            "scenario_id": scenario_id,
            "user_message": user_message,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "tool_executions": [],
            "model_calls": 0,
            "token_usage": 0,
            "steps": [],
        }
        save_checkpoint(checkpoint_path, state)

    def finish_fallback(reason, status):
        final_answer = f"{status}: {reason}"

        append_step(
            state,
            {
                "type": "fallback",
                "status": status,
                "reason": reason,
            },
        )

        state["final_answer"] = final_answer
        state["outcome"] = status
        save_checkpoint(checkpoint_path, state)

        result = agent_result(state, final_answer)
        log_agent_run(result)
        return result

    if time_limit_seconds <= 0:
        return finish_fallback(
            "Time limit reached before the first model call.",
            "FALLBACK_TIMEOUT",
        )

    while state["model_calls"] < max_iterations:
        if time.monotonic() - started_at >= time_limit_seconds:
            return finish_fallback(
                "Agent exceeded its time limit.", "FALLBACK_TIMEOUT"
            )


        response = mock_llm.chat(state["messages"], tools=TOOL_SPECS)
        state["model_calls"] += 1

        usage = getattr(response, "usage", None)
        call_tokens = int(getattr(usage, "total_tokens", 0)) if usage else 0
        state["token_usage"] += call_tokens

        choice = response.choices[0]
        message = choice.message
        finish_reason = getattr(choice, "finish_reason", "")
        tool_calls = getattr(message, "tool_calls", None) or []

        append_step(
            state,
            {
                "type": "model_call",
                "llm_call": state["model_calls"],
                "finish_reason": finish_reason,
                "tokens_used": call_tokens,
            },
        )
        save_checkpoint(checkpoint_path, state)

        if time.monotonic() - started_at >= time_limit_seconds:
            return finish_fallback(
                "Agent exceeded its time limit.", "FALLBACK_TIMEOUT"
            )

        if state["token_usage"] > token_limit:
            return finish_fallback(
                "Agent reached its token budget.", "FALLBACK_TOKEN_LIMIT"
            )

        if tool_calls:
            serialized_calls = []

            for call in tool_calls:
                arguments = call.arguments

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                serialized_calls.append(
                    {
                        "id": call.id,
                        "name": call.name,
                        "arguments": arguments,
                    }
                )

            state["messages"].append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": serialized_calls,
                }
            )

            for call_record in serialized_calls:
                tool_name = call_record["name"]
                tool_args = call_record["arguments"]

                if tool_name not in TOOL_MAP:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    tool_result = TOOL_MAP[tool_name](
                        tool_args, scenario_id
                    )

                event = {
                    "scenario_id": scenario_id,
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result,
                }

                state["tool_executions"].append(event)

                append_step(
                    state,
                    {
                        "type": "tool_execution",
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": tool_result,
                    },
                )

                state["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": call_record["id"],
                        "name": tool_name,
                        "content": json.dumps(
                            tool_result, ensure_ascii=False
                        ),
                    }
                )

                # Save after every tool execution to allow safe resumption.
                save_checkpoint(checkpoint_path, state)

                if (
                    crash_after_tool is not None
                    and len(state["tool_executions"]) == crash_after_tool
                ):
                    append_step(
                        state,
                        {
                            "type": "simulated_crash",
                            "after_tool_execution": crash_after_tool,
                        },
                    )
                    save_checkpoint(checkpoint_path, state)

                    raise SimulatedCrash(
                        "Simulated crash after tool execution "
                        f"{crash_after_tool}."
                    )

            continue

        final_answer = message.content or ""

        append_step(
            state,
            {
                "type": "final_answer",
                "finish_reason": finish_reason,
                "text": final_answer,
            },
        )

        state["final_answer"] = final_answer
        state["outcome"] = classify_outcome(final_answer)
        save_checkpoint(checkpoint_path, state)

        result = agent_result(state, final_answer)
        log_agent_run(result)
        return result

    return finish_fallback(
        "Maximum model-call iterations reached.",
        "FALLBACK_MAX_ITERATIONS",
    )


# =========================================================
# 5. RUN ALL SEVEN SCENARIOS
# =========================================================

def main():
    if not SCENARIOS_FILE.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {SCENARIOS_FILE}"
        )

    scenarios = pd.read_csv(SCENARIOS_FILE, keep_default_na=False)

    required_columns = {
        "scenario_id",
        "customer_id",
        "message",
        "expected_outcome",
        "expected_refund_inr",
    }

    if not required_columns.issubset(scenarios.columns):
        missing = required_columns - set(scenarios.columns)
        raise ValueError(f"Missing scenario columns: {sorted(missing)}")

    # One log record per completed run; preserve the refund ledger.
    RUN_LOG_FILE.write_text("", encoding="utf-8")

    result_rows = []
    s1_tool_names = []

    for _, row in scenarios.iterrows():
        scenario_id = str(row["scenario_id"]).strip()
        user_message = str(row["message"])
        checkpoint = STATE_DIR / f"{scenario_id}_checkpoint.json"

        if checkpoint.exists():
            checkpoint.unlink()

        mock_llm.reset()

        if scenario_id == "S1":
            print("\nS1: Starting crash-and-resume test.")

            try:
                run_agent(
                    scenario_id,
                    user_message,
                    max_iterations=6,
                    token_limit=6000,
                    time_limit_seconds=10,
                    checkpoint_path=checkpoint,
                    crash_after_tool=2,
                )
            except SimulatedCrash as exc:
                print(exc)
                print("Checkpoint saved. Resuming S1...")

            # Do not reset the mock model or delete the checkpoint on resume.
            result = run_agent(
                scenario_id,
                user_message,
                max_iterations=6,
                token_limit=6000,
                time_limit_seconds=10,
                checkpoint_path=checkpoint,
                resume=True,
            )

            s1_tool_names = result["tool_names"]
        else:
            result = run_agent(
                scenario_id,
                user_message,
                max_iterations=6,
                token_limit=6000,
                time_limit_seconds=10,
                checkpoint_path=checkpoint,
            )

        expected_outcome = str(row["expected_outcome"]).strip().upper()
        expected_refund = round(float(row["expected_refund_inr"]), 2)
        actual_refund = round(float(result["refund_inr"]), 2)

        if expected_outcome == "FALLBACK":
            outcome_pass = result["status"].startswith("FALLBACK")
        else:
            outcome_pass = result["status"] == expected_outcome

        refund_pass = abs(actual_refund - expected_refund) < 0.01
        passed = outcome_pass and refund_pass

        result_rows.append(
            {
                "scenario_id": scenario_id,
                "customer_id": row["customer_id"],
                "expected_outcome": expected_outcome,
                "actual_outcome": result["status"],
                "expected_refund_inr": expected_refund,
                "actual_refund_inr": actual_refund,
                "model_calls": result["llm_calls"],
                "token_usage": result["tokens_used"],
                "tool_names": " -> ".join(result["tool_names"]),
                "passed": passed,
                "final_answer": result["final_answer"],
            }
        )

        print(
            f"{scenario_id}: {result['status']} | "
            f"Refund INR {actual_refund:.2f} | "
            f"Model calls: {result['llm_calls']} | "
            f"{'PASS' if passed else 'CHECK'}"
        )

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(RESULTS_FILE, index=False)

    print("\n" + "=" * 65)
    print("AGENT SCENARIO RESULTS")
    print("=" * 65)

    print(
        result_df[
            [
                "scenario_id",
                "expected_outcome",
                "actual_outcome",
                "expected_refund_inr",
                "actual_refund_inr",
                "model_calls",
                "passed",
            ]
        ].to_string(index=False)
    )

    expected_s1_tools = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
    ]

    if s1_tool_names != expected_s1_tools:
        raise AssertionError(
            "S1 did not resume with the expected tool sequence. "
            f"Observed: {s1_tool_names}"
        )

    s1_row = result_df[result_df["scenario_id"] == "S1"].iloc[0]

    if int(s1_row["model_calls"]) != 5:
        raise AssertionError(
            f"S1 should use 5 model calls; used {s1_row['model_calls']}."
        )

    if not result_df["passed"].all():
        raise AssertionError(
            "One or more scenarios failed. Check agent_scenario_results.csv."
        )

    # =====================================================
    # 6. SAFETY-LIMIT DEMONSTRATIONS
    # =====================================================

    s1 = scenarios[scenarios["scenario_id"].astype(str) == "S1"].iloc[0]
    safety_message = str(s1["message"])

    mock_llm.reset()

    token_demo = run_agent(
        "SAFETY_TOKEN",
        safety_message,
        max_iterations=6,
        token_limit=200,
        time_limit_seconds=10,
        checkpoint_path=STATE_DIR / "safety_token_checkpoint.json",
    )

    mock_llm.reset()

    timeout_demo = run_agent(
        "SAFETY_TIMEOUT",
        safety_message,
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=0,
        checkpoint_path=STATE_DIR / "safety_timeout_checkpoint.json",
    )

    safety_df = pd.DataFrame(
        [
            {
                "demo": "token_limit_200",
                "expected_status": "FALLBACK_TOKEN_LIMIT",
                "actual_status": token_demo["status"],
                "passed": (
                    token_demo["status"] == "FALLBACK_TOKEN_LIMIT"
                ),
            },
            {
                "demo": "time_limit_0",
                "expected_status": "FALLBACK_TIMEOUT",
                "actual_status": timeout_demo["status"],
                "passed": (
                    timeout_demo["status"] == "FALLBACK_TIMEOUT"
                ),
            },
        ]
    )

    safety_df.to_csv(SAFETY_FILE, index=False)

    print("\n" + "=" * 65)
    print("SAFETY-LIMIT DEMONSTRATIONS")
    print("=" * 65)
    print(safety_df.to_string(index=False))

    if not safety_df["passed"].all():
        raise AssertionError("A safety-limit demonstration failed.")

    # =====================================================
    # 7. VALIDATE RUN LOG AND REFUND LEDGER
    # =====================================================

    log_rows = read_jsonl(RUN_LOG_FILE)

    required_log_fields = {
        "scenario_id",
        "status",
        "final_answer",
        "llm_calls",
        "tokens_used",
        "steps",
    }

    if not log_rows:
        raise AssertionError("Agent run log is empty.")

    if any(
        not required_log_fields.issubset(record)
        for record in log_rows
    ):
        raise AssertionError("Agent log is missing required fields.")

    # Seven scenario runs plus two safety demonstrations.
    if len(log_rows) != 9:
        raise AssertionError(
            f"Expected 9 agent run records, found {len(log_rows)}."
        )

    ledger = read_jsonl(LEDGER_FILE)

    if not any(item.get("status") == "issued" for item in ledger):
        raise AssertionError("Refund ledger lacks an issued-refund record.")

    if not any(item.get("status") == "escalated" for item in ledger):
        raise AssertionError("Refund ledger lacks an escalation record.")

    print("\n" + "=" * 65)
    print("PART D AGENT COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"Scenario results: {RESULTS_FILE}")
    print(f"Safety demos:     {SAFETY_FILE}")
    print(f"Refund ledger:    {LEDGER_FILE}")
    print(f"Agent run log:    {RUN_LOG_FILE}")
    print(f"Checkpoints:      {STATE_DIR}")
    print(f"Agent runs logged: {len(log_rows)}")


if __name__ == "__main__":
    main()

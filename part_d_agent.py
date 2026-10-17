
"""Part D: ReAct refund agent with tools, safety limits and resumable state."""

from pathlib import Path
import contextlib
import csv
import io
import json
import re
import time

import pandas as pd
import mock_llm

# Suppress the policy module's setup display when importing it.
with contextlib.redirect_stdout(io.StringIO()):
    from part_d_policy import retrieve


# ---------------------------------------------------------
# 1. FILE LOCATIONS
# ---------------------------------------------------------

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

for folder in (OUTPUT_DIR, STATE_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. LOAD CLEANED ORDERS
# ---------------------------------------------------------

if not ORDERS_FILE.exists():
    raise FileNotFoundError(f"Orders file not found: {ORDERS_FILE}")

orders_df = pd.read_csv(ORDERS_FILE, keep_default_na=False)
orders_df["order_id"] = orders_df["order_id"].astype(str).str.strip()

if orders_df["order_id"].duplicated().any():
    raise ValueError("Cleaned orders contain duplicate order IDs.")


def get_order_record(order_id):
    """Return one cleaned order as a normal Python dictionary."""
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


# ---------------------------------------------------------
# 3. TOOL IMPLEMENTATIONS
# ---------------------------------------------------------

def tool_get_order(args, scenario_id):
    order_id = str(args.get("order_id", "")).strip()

    if not order_id:
        return {"error": "Missing order ID."}

    order = get_order_record(order_id)

    if order is None:
        return {"error": "Order not found.", "order_id": order_id}

    return order


def tool_search_policy(args, scenario_id):
    query = str(args.get("query", "")).strip()

    if not query:
        return {"error": "A policy search query is required."}

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
    """Calculate a full refund only for completed orders delivered after 45 minutes."""
    try:
        amount = float(args.get("order_amount"))
        delivery = float(args.get("delivery_minutes"))
    except (TypeError, ValueError):
        return {
            "error": "A valid order amount and delivery time are required.",
            "refund_inr": 0.0,
        }

    if amount < 0 or delivery < 0:
        return {
            "error": "Negative amounts and delivery times are invalid.",
            "refund_inr": 0.0,
        }

    refund = round(amount, 2) if delivery > 45 else 0.0

    return {
        "order_amount": round(amount, 2),
        "delivery_minutes": delivery,
        "refund_inr": refund,
        "reason": (
            "Completed-order late-delivery rule"
            if refund > 0
            else "Delivery was within 45 minutes"
        ),
    }


def read_ledger():
    """Read the existing refund ledger, if one exists."""
    entries = []

    if LEDGER_FILE.exists():
        with LEDGER_FILE.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        print("Warning: Skipped a malformed ledger line.")

    return entries


def append_jsonl(path, record):
    """Append one JSON record to a JSON Lines file."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def tool_issue_refund(args, scenario_id):
    """Issue an eligible refund, with basic validation and idempotency."""
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

    if order["delivery_minutes"] is None or order["delivery_minutes"] <= 45:
        return {"error": "The order does not qualify for a late-delivery refund."}

    if order["order_amount"] is None:
        return {"error": "The order amount is unavailable."}

    if refund <= 0 or abs(refund - order["order_amount"]) > 0.01:
        return {"error": "The refund amount does not match the eligible order amount."}

    if refund > 1000:
        return {
            "error": "Refunds above INR 1000 require human approval.",
            "requires_escalation": True,
        }

    # Avoid adding the same scenario/order combination twice on a rerun.
    existing = read_ledger()
    existing_entry = next(
        (
            entry for entry in existing
            if entry.get("scenario_id") == scenario_id
            and entry.get("order_id") == order_id
        ),
        None,
    )

    if existing_entry:
        return {
            "status": "already_issued",
            "order_id": order_id,
            "refund_inr": existing_entry["refund_inr"],
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
    """Record an escalation request for an eligible refund above INR 1000."""
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

    if order["delivery_minutes"] is None or order["delivery_minutes"] <= 45:
        return {"error": "The order does not qualify for a late-delivery refund."}

    if order["order_amount"] is None:
        return {"error": "The order amount is unavailable."}

    if refund <= 1000 or abs(refund - order["order_amount"]) > 0.01:
        return {"error": "Human escalation requires an eligible refund above INR 1000."}

    return {
        "status": "escalated",
        "order_id": order_id,
        "refund_inr": refund,
        "message": "Sent to a human support lead for approval.",
    }


TOOL_MAP = {
    "get_order": tool_get_order,
    "search_policy": tool_search_policy,
    "calculate_refund": tool_calculate_refund,
    "issue_refund": tool_issue_refund,
    "escalate_to_human": tool_escalate_to_human,
}


# Tool definitions supplied to the model.
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order using its order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Retrieve relevant policy chunks for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate a refund under the late-delivery rule.",
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
            "description": "Issue a validated refund of INR 1000 or less.",
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
            "description": "Escalate an eligible refund above INR 1000.",
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


# ---------------------------------------------------------
# 4. CHECKPOINTS AND AGENT LOOP
# ---------------------------------------------------------

SYSTEM_PROMPT = """
AGENT
You are an Instamart customer-support refund agent.
Use tools to verify orders and follow the supplied refund policy.
Only completed orders can receive a refund.
A completed order delivered in more than 45 minutes qualifies
for a full refund. Delivery in 45 minutes or less does not qualify.
Refunds up to INR 1000 can be issued by the system.
Refunds above INR 1000 must be escalated to a human support lead.
Never invent an order or skip the required checks.
"""


class SimulatedCrash(Exception):
    """Used to demonstrate checkpoint recovery after a tool execution."""


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def estimated_message_tokens(messages):
    total = 0

    for message in messages:
        content = message.get("content") or ""

        if isinstance(content, str):
            total += mock_llm.est_tokens(content)
        else:
            total += mock_llm.est_tokens(json.dumps(content, default=str))

    return total


def classify_outcome(final_answer):
    answer_text = str(final_answer).strip().upper()

    if answer_text.startswith("REFUND:"):
        return "REFUND"
    if answer_text.startswith("ESCALATE:"):
        return "ESCALATE"
    if answer_text.startswith("DENIED:"):
        return "DENIED"
    if answer_text.startswith("NOT_FOUND:"):
        return "NOT_FOUND"
    if answer_text.startswith("NEEDS_INFO:"):
        return "NEEDS_INFO"
    if answer_text.startswith("FALLBACK_TOKEN_LIMIT"):
        return "FALLBACK_TOKEN_LIMIT"
    if answer_text.startswith("FALLBACK_TIMEOUT"):
        return "FALLBACK_TIMEOUT"
    if answer_text.startswith("FALLBACK_MAX_ITERATIONS"):
        return "FALLBACK_MAX_ITERATIONS"

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
    return {
        "scenario_id": state["scenario_id"],
        "outcome": classify_outcome(final_answer),
        "final_answer": str(final_answer),
        "refund_inr": result_refund_amount(state["tool_executions"]),
        "model_calls": state["model_calls"],
        "token_usage": state["token_usage"],
        "tool_executions": state["tool_executions"],
        "tool_names": [event["tool"] for event in state["tool_executions"]],
    }


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
    """Run the ReAct loop, checkpoint after each tool, and support resuming."""
    if checkpoint_path is None:
        checkpoint_path = STATE_DIR / f"{scenario_id}_checkpoint.json"

    started_at = time.monotonic()

    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Cannot resume; checkpoint does not exist: {checkpoint_path}"
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

    def safe_fallback(reason, label):
        final_answer = f"{label}: {reason}"
        state["final_answer"] = final_answer
        state["outcome"] = label
        save_checkpoint(checkpoint_path, state)
        append_jsonl(
            RUN_LOG_FILE,
            {
                "event": "fallback",
                "scenario_id": scenario_id,
                "status": label,
                "reason": reason,
                "model_calls": state["model_calls"],
                "token_usage": state["token_usage"],
            },
        )
        return agent_result(state, final_answer)

    if time_limit_seconds <= 0:
        return safe_fallback(
            "Time limit reached before the first model call.",
            "FALLBACK_TIMEOUT",
        )

    while state["model_calls"] < max_iterations:
        elapsed = time.monotonic() - started_at

        if elapsed >= time_limit_seconds:
            return safe_fallback(
                "Agent exceeded its time limit.",
                "FALLBACK_TIMEOUT",
            )

        estimated_prompt = estimated_message_tokens(state["messages"])

        if state["token_usage"] + estimated_prompt > token_limit:
            return safe_fallback(
                "Agent reached its token budget.",
                "FALLBACK_TOKEN_LIMIT",
            )

        response = mock_llm.chat(state["messages"], tools=TOOL_SPECS)
        state["model_calls"] += 1

        usage = getattr(response, "usage", None)
        if usage is not None:
            state["token_usage"] += int(getattr(usage, "total_tokens", 0))

        elapsed = time.monotonic() - started_at

        if elapsed >= time_limit_seconds:
            return safe_fallback(
                "Agent exceeded its time limit.",
                "FALLBACK_TIMEOUT",
            )

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        finish_reason = getattr(response.choices[0], "finish_reason", "")

        if tool_calls:
            # Record the assistant's tool request in the conversation.
            assistant_tool_calls = []

            for call in tool_calls:
                arguments = call.arguments

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                assistant_tool_calls.append({
                    "id": call.id,
                    "name": call.name,
                    "arguments": arguments,
                })

            state["messages"].append({
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
            })

            for call_record in assistant_tool_calls:
                tool_name = call_record["name"]
                tool_args = call_record["arguments"]

                if tool_name not in TOOL_MAP:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    tool_result = TOOL_MAP[tool_name](
                        tool_args,
                        scenario_id,
                    )

                event = {
                    "scenario_id": scenario_id,
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result,
                }

                state["tool_executions"].append(event)
                state["steps"].append({
                    "step": len(state["steps"]) + 1,
                    "type": "tool",
                    "tool": tool_name,
                })

                state["messages"].append({
                    "role": "tool",
                    "tool_call_id": call_record["id"],
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

                # Persist state BEFORE simulating a crash.
                save_checkpoint(checkpoint_path, state)
                append_jsonl(RUN_LOG_FILE, {"event": "tool_execution", **event})

                if (
                    crash_after_tool is not None
                    and len(state["tool_executions"]) == crash_after_tool
                ):
                    append_jsonl(
                        RUN_LOG_FILE,
                        {
                            "event": "simulated_crash",
                            "scenario_id": scenario_id,
                            "after_tool_execution": crash_after_tool,
                        },
                    )
                    raise SimulatedCrash(
                        f"Simulated crash after tool execution {crash_after_tool}."
                    )

            continue

        final_answer = message.content or ""

        state["steps"].append({
            "step": len(state["steps"]) + 1,
            "type": "final_answer",
            "finish_reason": finish_reason,
        })

        state["final_answer"] = final_answer
        state["outcome"] = classify_outcome(final_answer)
        save_checkpoint(checkpoint_path, state)

        result = agent_result(state, final_answer)

        append_jsonl(
            RUN_LOG_FILE,
            {
                "event": "run_complete",
                "scenario_id": scenario_id,
                "outcome": result["outcome"],
                "final_answer": result["final_answer"],
                "model_calls": result["model_calls"],
                "token_usage": result["token_usage"],
                "tool_names": result["tool_names"],
            },
        )

        return result

    return safe_fallback(
        "Maximum model-call iterations reached.",
        "FALLBACK_MAX_ITERATIONS",
    )


# ---------------------------------------------------------
# 5. SCENARIO TESTS AND CRASH/RESUME DEMONSTRATION
# ---------------------------------------------------------

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
        raise ValueError(f"Scenario CSV is missing columns: {sorted(missing)}")

    # Begin a fresh run log; leave the refund ledger intact.
    RUN_LOG_FILE.write_text("", encoding="utf-8")

    result_rows = []
    s1_tool_names = []

    for _, row in scenarios.iterrows():
        scenario_id = str(row["scenario_id"]).strip()
        message = str(row["message"])
        checkpoint = STATE_DIR / f"{scenario_id}_checkpoint.json"

        mock_llm.reset()

        if scenario_id == "S1":
            # Demonstrate restart after the second completed tool execution.
            if checkpoint.exists():
                checkpoint.unlink()

            print("\nS1: Starting crash-and-resume demonstration.")

            try:
                run_agent(
                    scenario_id,
                    message,
                    max_iterations=6,
                    token_limit=6000,
                    time_limit_seconds=10,
                    checkpoint_path=checkpoint,
                    resume=False,
                    crash_after_tool=2,
                )
            except SimulatedCrash as exc:
                print(exc)
                print("Checkpoint was saved. Resuming S1...")

            # Do not reset the model or remove the checkpoint here.
            result = run_agent(
                scenario_id,
                message,
                max_iterations=6,
                token_limit=6000,
                time_limit_seconds=10,
                checkpoint_path=checkpoint,
                resume=True,
            )

            s1_tool_names = result["tool_names"]

        else:
            if checkpoint.exists():
                checkpoint.unlink()

            result = run_agent(
                scenario_id,
                message,
                max_iterations=6,
                token_limit=6000,
                time_limit_seconds=10,
                checkpoint_path=checkpoint,
            )

        expected_outcome = str(row["expected_outcome"]).strip().upper()
        expected_refund = round(float(row["expected_refund_inr"]), 2)
        actual_refund = round(float(result["refund_inr"]), 2)

        if expected_outcome == "FALLBACK":
            outcome_pass = result["outcome"].startswith("FALLBACK")
        else:
            outcome_pass = result["outcome"] == expected_outcome

        # Compare refund values for the expected outcomes in the scenario file.
        refund_pass = abs(actual_refund - expected_refund) < 0.01
        passed = outcome_pass and refund_pass

        result_rows.append({
            "scenario_id": scenario_id,
            "customer_id": row["customer_id"],
            "expected_outcome": expected_outcome,
            "actual_outcome": result["outcome"],
            "expected_refund_inr": expected_refund,
            "actual_refund_inr": actual_refund,
            "model_calls": result["model_calls"],
            "token_usage": result["token_usage"],
            "tool_names": " -> ".join(result["tool_names"]),
            "passed": passed,
            "final_answer": result["final_answer"],
        })

        print(
            f"{scenario_id}: {result['outcome']} | "
            f"Refund INR {actual_refund:.2f} | "
            f"Model calls: {result['model_calls']} | "
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

    # Verify the interrupted S1 execution did not repeat earlier tools.
    expected_s1_tools = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
    ]

    if s1_tool_names != expected_s1_tools:
        raise AssertionError(
            "S1 resume did not produce the expected unique tool sequence. "
            f"Observed: {s1_tool_names}"
        )

    s1_row = result_df[result_df["scenario_id"] == "S1"].iloc[0]

    if int(s1_row["model_calls"]) != 5:
        raise AssertionError(
            f"S1 should use 5 model calls; used {s1_row['model_calls']}."
        )

    if not result_df["passed"].all():
        raise AssertionError(
            "One or more agent scenarios failed. Review the scenario results CSV."
        )

    # -----------------------------------------------------
    # 6. SAFETY-LIMIT DEMONSTRATIONS
    # -----------------------------------------------------

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

    safety_rows = [
        {
            "demo": "token_limit_200",
            "expected_status": "FALLBACK_TOKEN_LIMIT",
            "actual_status": token_demo["outcome"],
            "passed": token_demo["outcome"] == "FALLBACK_TOKEN_LIMIT",
        },
        {
            "demo": "time_limit_0",
            "expected_status": "FALLBACK_TIMEOUT",
            "actual_status": timeout_demo["outcome"],
            "passed": timeout_demo["outcome"] == "FALLBACK_TIMEOUT",
        },
    ]

    safety_df = pd.DataFrame(safety_rows)
    safety_df.to_csv(SAFETY_FILE, index=False)

    print("\n" + "=" * 65)
    print("SAFETY-LIMIT DEMONSTRATIONS")
    print("=" * 65)
    print(safety_df.to_string(index=False))

    if not safety_df["passed"].all():
        raise AssertionError(
            "A safety-limit demonstration did not produce its expected status."
        )

    print("\n" + "=" * 65)
    print("PART D AGENT COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"Scenario results: {RESULTS_FILE}")
    print(f"Safety demos:     {SAFETY_FILE}")
    print(f"Refund ledger:    {LEDGER_FILE}")
    print(f"Agent run log:    {RUN_LOG_FILE}")
    print(f"Checkpoints:      {STATE_DIR}")


if __name__ == "__main__":
    main()

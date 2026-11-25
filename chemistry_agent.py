"""Command-line entry point for the Chemistry Experiment Results Assistant."""

import argparse
import json
from pathlib import Path
import sys

from agent_loop import run_agent, verify_trace
from chemistry_tools import ChemistryRecords
from model_client import ModelClient, ModelError

DATA = Path(__file__).resolve().parent / "examples" / "experiments.json"
DEMOS = {
    "comparison": "Using A as the baseline, how much lower is B's isolated yield? What in the record might help explain the difference, and how certain is that explanation?",
    "missing-unit": "Using A as the baseline, what is the numerical difference in isolated yield for experiment C?",
    "number-only": "What is B minus A in isolated yield? Just give the numerical difference and units.",
}


def describe(result):
    if not result["ok"]:
        missing = ", ".join(result.get("missing_unit_ids", []))
        return f"{result['code']}: {result['message']}" + (f" Missing unit for {missing}." if missing else "")
    tool = result.get("tool")
    if tool == "compare_results":
        return (f"{result['b']} minus {result['a']} = {result['value_b']} {result['unit_b']} - "
                f"{result['value_a']} {result['unit_a']} = {result['difference']} {result['difference_unit']}.")
    if tool == "read_record":
        r = result["record"]
        m = r["result"]
        return (f"{r['id']} | {m['metric'] or 'unnamed measurement'} = "
                f"{m['value'] or 'missing value'} {m['unit'] or '[unit missing]'}. "
                + " ".join(r["notes"]))
    if tool == "lookup_note":
        return f"Teaching note [{result['topic']}] {result['text']} Source: {result['source']}"
    if tool == "list_records":
        return ", ".join(f"{r['id']} ({r['title']})" for r in result['records'])
    return "Accepted."


def show_event(event):
    decision = event["decision"] if isinstance(event["decision"], dict) else {}
    print(f"\n[{event['step']}] Model chose {decision.get('action')} {json.dumps(decision.get('arguments'))}")
    print("    Purpose:", decision.get("reason", ""))
    print("    Observation:", describe(event["observation"]), flush=True)


def show_outcome(outcome):
    print(f"\nOutcome: {outcome['status']}")
    for item in outcome.get("evidence", []):
        print(f"  [step {item['step']}] {describe(item['result'])}")
    if outcome.get("kind") == "interpretation":
        print("  The records support a possible explanation only. These observations do not establish a cause.")
    if outcome["status"] == "needs_input":
        print("  Clarification:", outcome["question"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ask", metavar="QUESTION", help="Run a new question through the model.")
    group.add_argument("--demo", choices=DEMOS, help="Run a sample question through the model.")
    group.add_argument("--replay", type=Path, metavar="TRACE", help="Verify recorded tool calls offline.")
    parser.add_argument("--data", type=Path, default=DATA, help="Experiment log (JSON).")
    parser.add_argument("--save-trace", type=Path, help="Write a new live trace file; the destination must be new.")
    args = parser.parse_args(argv)
    if args.replay and args.save_trace:
        parser.error("--save-trace is only for a live run.")
    try:
        records = ChemistryRecords.from_path(args.data)
        if args.replay:
            with args.replay.open("rb") as src:
                raw = src.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError("The trace exceeds 2 MiB.")
            trace = json.loads(raw)
            result = verify_trace(records, trace)
            print("Offline replay | recomputed all observations | recorded model:", trace["model"])
            print("Request:", trace["request"])
            for event in trace["steps"]:
                show_event(event)
        else:
            if args.save_trace and args.save_trace.exists():
                raise ValueError("Trace path already exists. Choose a new filename.")
            question = args.ask if args.ask is not None else DEMOS[args.demo]
            client = ModelClient()
            print("Live model:", client.model, "\nRequest:", question, flush=True)
            trace = run_agent(records, question, client, on_step=show_event)
            result = trace["outcome"]
            if args.save_trace:
                args.save_trace.parent.mkdir(parents=True, exist_ok=True)
                with args.save_trace.open("x", encoding="utf-8") as out:
                    json.dump(trace, out, indent=2, ensure_ascii=True, allow_nan=False)
                    out.write("\n")
        show_outcome(result)
        return 1 if result["status"] == "step_limit" else 0
    except (OSError, ValueError, ModelError) as exc:
        print(f"Could not complete the request: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

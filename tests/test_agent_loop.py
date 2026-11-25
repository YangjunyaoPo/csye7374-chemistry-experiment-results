"""Check the protocol, grounded reports and independent answers for real model runs."""

import copy
import json
from pathlib import Path
import unittest

from agent_loop import MAX_STEPS, run_agent, verify_trace
from chemistry_agent import DEMOS, describe
from chemistry_tools import ChemistryRecords

ROOT = Path(__file__).resolve().parents[1]


def decision(action, **arguments):
    return {"action": action, "arguments": arguments, "reason": "Check the supplied evidence."}


class ScriptedClient:
    """Test double for exercising failure paths without a model server."""

    model = "unit-test-double"

    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.histories = []

    def decide(self, messages):
        self.histories.append(messages)
        return next(self.decisions)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.records = ChemistryRecords.from_path(ROOT / "examples" / "experiments.json")

    def run_steps(self, *decisions):
        client = ScriptedClient(decisions)
        trace = run_agent(self.records, "Compare recorded yields.", client)
        return trace, client

    def load_trace(self, name):
        return json.loads((ROOT / "evidence" / (name + ".json")).read_bytes())

    def test_model_receives_failed_observation_before_next_decision(self):
        trace, client = self.run_steps(decision("compare_results", a="A", b="C"),
                                      decision("ask", question="What is C's unit?", step=1))
        observed = json.loads(client.histories[1][-1]["content"])
        self.assertEqual(observed["observed_step"], 1)
        self.assertEqual(observed["tool_observation"]["missing_unit_ids"], ["C"])
        self.assertEqual(trace["outcome"]["status"], "needs_input")
        self.assertNotIn("difference", trace["outcome"]["evidence"][0]["result"])

    def test_invalid_reports_feed_back_and_can_recover(self):
        trace, _ = self.run_steps(decision("report", kind="comparison", steps=[5]),
                                  decision("read_record", id="A"),
                                  decision("report", kind="interpretation", steps=[2]),
                                  decision("report", kind="record", steps=[2]))
        self.assertEqual([trace["steps"][i]["observation"]["code"] for i in (0, 2)],
                         ["invalid_reference", "insufficient_evidence"])
        self.assertEqual(trace["outcome"]["kind"], "record")

    def test_interpretation_needs_target_record(self):
        trace, _ = self.run_steps(decision("compare_results", a="A", b="B"),
                                  decision("read_record", id="A"), decision("lookup_note", topic="transfer"),
                                  decision("report", kind="interpretation", steps=[1, 2, 3]),
                                  decision("read_record", id="B"),
                                  decision("report", kind="interpretation", steps=[1, 3, 5]))
        self.assertEqual(trace["steps"][3]["observation"]["code"], "insufficient_evidence")
        self.assertEqual(trace["outcome"]["kind"], "interpretation")

    def test_a_failed_comparison_cannot_be_reported_as_success(self):
        trace, _ = self.run_steps(decision("compare_results", a="A", b="C"),
                                  decision("report", kind="comparison", steps=[1]),
                                  decision("ask", question="What unit did C use?", step=1))
        self.assertEqual(trace["steps"][1]["observation"]["code"], "invalid_reference")

    def test_bounded_loop_on_unknown_actions(self):
        trace, _ = self.run_steps(*[decision("unlisted_tool")] * MAX_STEPS)
        self.assertEqual(trace["outcome"]["status"], "step_limit")
        self.assertEqual(verify_trace(self.records, trace), trace["outcome"])

    def test_replay_detects_changed_observation_and_identity(self):
        base = self.load_trace("comparison")
        for key in ("notes_sha256", "source_sha256", "model", "request"):
            trace = copy.deepcopy(base)
            trace[key] = ""
            with self.assertRaises(ValueError):
                verify_trace(self.records, trace)
        trace = copy.deepcopy(base)
        trace["steps"][0]["observation"]["ok"] = 1
        with self.assertRaisesRegex(ValueError, "differs at step"):
            verify_trace(self.records, trace)
        trace = copy.deepcopy(base)
        trace["outcome"]["status"] = "needs_input"
        with self.assertRaisesRegex(ValueError, "Final outcome"):
            verify_trace(self.records, trace)

    def test_recorded_outcomes_against_hand_worked_oracle(self):
        # Raw fixture gives 645 tenths minus 780 tenths = -135 tenths of a point.
        for name, kind in (("comparison", "interpretation"), ("number-only", "comparison")):
            out = verify_trace(self.records, self.load_trace(name))
            self.assertEqual((out["status"], out["kind"]), ("reported", kind))
            results = [item["result"] for item in out["evidence"]]
            result = next(r for r in results if r["tool"] == "compare_results")
            self.assertEqual((result["a"], result["b"], result["difference"], result["difference_unit"]),
                             ("A", "B", "-13.5", "percentage points"))
            if kind == "interpretation":
                self.assertTrue(any(r.get("topic") == "transfer" for r in results))
                self.assertTrue(any("Visible solid" in describe(r) for r in results))
        missing = self.load_trace("missing-unit")
        out = verify_trace(self.records, missing)
        self.assertEqual(out["status"], "needs_input")
        self.assertIn("unit", out["question"].lower())
        self.assertIn("C", out["question"])
        for event in missing["steps"]:
            self.assertNotIn("difference", event["observation"])

    def test_explicit_reverse_direction_rejects_wrong_operands_before_arithmetic(self):
        client = ScriptedClient([decision("compare_results", a="A", b="B"),
                                 decision("compare_results", a="B", b="A"),
                                 decision("report", kind="comparison", steps=[2])])
        trace = run_agent(self.records, "Using B as the baseline, what is A minus B?", client)
        rejected = trace["steps"][0]["observation"]
        self.assertEqual(rejected["code"], "direction_mismatch")
        self.assertNotIn("difference", rejected)
        result = verify_trace(self.records, trace)["evidence"][0]["result"]
        self.assertEqual((result["a"], result["b"], result["difference"]), ("B", "A", "13.5"))

    def test_wrong_pair_supplies_exact_baseline_and_recovers_before_clarifying(self):
        q = "Using B as the baseline, what is A minus B in percentage points?"
        client = ScriptedClient([
            decision("compare_results", a="A", b="B"),
            decision("ask", question="Which baseline did you mean?", step=1),
            decision("compare_results", a="B", b="A"),
            decision("report", kind="comparison", steps=[3]),
        ])
        trace = run_agent(self.records, q, client)
        for event in trace["steps"][:2]:
            self.assertEqual(event["observation"]["expected_arguments"], {"a": "B", "b": "A"})
            self.assertNotIn("difference", event["observation"])
        self.assertEqual(verify_trace(self.records, trace)["evidence"][0]["result"]["difference"], "13.5")

    def test_missing_unit_target_cannot_be_substituted_or_answered_by_a_catalog(self):
        client = ScriptedClient([
            decision("compare_results", a="A", b="B"),
            decision("list_records"),
            decision("report", kind="catalog", steps=[2]),
            decision("compare_results", a="A", b="C"),
            decision("ask", question="Which unit did trial C use?", step=4),
        ])
        trace = run_agent(self.records, DEMOS["missing-unit"], client)
        wrong, _, catalog, missing, _ = [e["observation"] for e in trace["steps"]]
        self.assertEqual(wrong["expected_arguments"], {"a": "A", "b": "C"})
        self.assertEqual(catalog["code"], "comparison_required")
        self.assertEqual(missing["code"], "missing_unit")
        self.assertTrue(all("difference" not in ev["observation"] for ev in trace["steps"]))
        self.assertEqual(verify_trace(self.records, trace)["status"], "needs_input")

    def test_explicit_pair_conflicting_with_baseline_can_prompt_for_clarification(self):
        client = ScriptedClient([decision("compare_results", a="B", b="A"),
                                 decision("ask", question="Should the baseline be A or B?", step=1)])
        trace = run_agent(self.records, "Using A as the baseline, calculate A minus B.", client)
        blocked = trace["steps"][0]["observation"]
        self.assertEqual(blocked["code"], "conflicting_request")
        self.assertNotIn("expected_arguments", blocked)
        self.assertEqual(verify_trace(self.records, trace)["status"], "needs_input")

    def test_named_participants_bound_an_unordered_comparison(self):
        client = ScriptedClient([decision("compare_results", a="A", b="B"),
                                 decision("compare_results", a="C", b="A"),
                                 decision("ask", question="What unit does C use?", step=2)])
        trace = run_agent(self.records, "Compare trial A and experiment C.", client)
        self.assertEqual(trace["steps"][0]["observation"]["expected_ids"], ["A", "C"])
        self.assertEqual(trace["steps"][1]["observation"]["code"], "missing_unit")
        self.assertEqual(verify_trace(self.records, trace)["status"], "needs_input")

    def test_asking_after_malformed_action_returns_controlled_observation(self):
        client = ScriptedClient([decision(["read_record"]), decision("ask", question="Which record?", step=1),
                                 decision("ask", question="Which record?")])
        trace = run_agent(self.records, "Read the experiment notes.", client)
        self.assertEqual(trace["steps"][1]["observation"]["code"], "invalid_reference")
        self.assertEqual(verify_trace(self.records, trace)["status"], "needs_input")

    def test_noncomparison_record_question_with_an_article(self):
        client = ScriptedClient([decision("read_record", id="B"),
                                 decision("report", kind="record", steps=[1])])
        trace = run_agent(self.records, "Did the log record a yield for experiment B?", client)
        self.assertEqual(verify_trace(self.records, trace)["status"], "reported")

    def test_named_record_evidence_is_required_for_a_notebook_question(self):
        client = ScriptedClient([decision("lookup_note", topic="transfer"),
                                 decision("report", kind="note", steps=[1]),
                                 decision("read_record", id="B"),
                                 decision("report", kind="record", steps=[3])])
        trace = run_agent(self.records, "What was written about the transfer in experiment B?", client)
        self.assertEqual(trace["steps"][1]["observation"]["code"], "record_evidence_required")
        self.assertIn("Visible solid", describe(verify_trace(self.records, trace)["evidence"][0]["result"]))

    def test_classroom_explanation_requires_a_chemical_note_and_target_record(self):
        for topic in ("yield", "transfer"):
            with self.subTest(topic=topic):
                client = ScriptedClient([
                    decision("compare_results", a="A", b="B"),
                    decision("report", kind="comparison", steps=[1]),
                    decision("read_record", id="B"),
                    decision("lookup_note", topic="comparison"),
                    decision("report", kind="interpretation", steps=[1, 3, 4]),
                    decision("lookup_note", topic=topic),
                    decision("report", kind="interpretation", steps=[1, 3, 6]),
                ])
                trace = run_agent(self.records, DEMOS["comparison"], client)
                self.assertEqual([trace["steps"][i]["observation"]["code"] for i in (1, 4)],
                                 ["yield_explanation_required", "yield_explanation_required"])
                self.assertEqual(verify_trace(self.records, trace)["kind"], "interpretation")

    def test_v1_replays_the_historical_report_rule(self):
        trace = self.load_trace("number-only")
        trace["request"] = DEMOS["comparison"]
        self.assertEqual(verify_trace(self.records, trace)["kind"], "comparison")
        trace["format"] = "chemistry-agent-trace-v1.1"
        with self.assertRaisesRegex(ValueError, "differs at step 2"):
            verify_trace(self.records, trace)

    def test_background_question_can_still_use_a_teaching_note_alone(self):
        q = "Why could an isolated yield be lower than the theoretical yield?"
        client = ScriptedClient([decision("lookup_note", topic="yield"),
                                 decision("report", kind="note", steps=[1])])
        self.assertEqual(run_agent(self.records, q, client)["outcome"]["kind"], "note")

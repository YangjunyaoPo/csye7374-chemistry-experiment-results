import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from chemistry_tools import ChemistryInputError, ChemistryRecords, MAX_BYTES


SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "experiments.json"


class ChemistryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "records.json"
        self.book = ChemistryRecords.from_path(SAMPLE)
        self.fixtures = json.loads(SAMPLE.read_bytes())

    def load(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return ChemistryRecords.from_path(self.path)

    def compare(self, a="A", b="B", book=None):
        return (book or self.book).run("compare_results", {"a": a, "b": b})

    def test_catalog_and_frozen_read_point_to_the_original_record(self):
        listed = self.book.run("list_records", {})
        self.assertEqual([r["id"] for r in listed["records"]], ["A", "B", "C"])
        self.assertEqual(listed["source_sha256"], hashlib.sha256(SAMPLE.read_bytes()).hexdigest())
        read = self.book.run("read_record", {"id": "B"})
        self.assertEqual(read["source_ref"], "/records/1")
        self.assertEqual(read["record"], self.fixtures["records"][1])
        read["record"]["notes"].clear()
        self.assertEqual(self.book.run("read_record", {"id": "B"})["record"], self.fixtures["records"][1])

    def test_difference_matches_integer_tenths_and_cites_both_raw_values(self):
        result = self.compare()
        self.assertTrue(result["ok"])
        self.assertEqual((result["value_a"], result["value_b"], result["unit"]), ("78.0", "64.5", "%"))
        # In tenths of a percentage point, trial B minus A is 645 - 780.
        self.assertEqual(result["difference"], str((645 - 780) / 10))
        self.assertEqual((result["difference_unit"], result["direction"]), ("percentage points", "lower"))
        self.assertEqual(result["source_refs"], {"A": "/records/0", "B": "/records/1"})
        reverse = self.compare("B", "A")
        self.assertEqual((reverse["difference"], reverse["direction"]), ("13.5", "higher"))

    def test_missing_unit_is_preserved_and_comparison_identifies_both_operands(self):
        read = self.book.run("read_record", {"id": "C"})
        self.assertIsNone(read["record"]["result"]["unit"])
        for a, b in [("A", "C"), ("C", "B")]:
            with self.subTest(a=a, b=b):
                result = self.compare(a, b)
                self.assertFalse(result["ok"])
                self.assertEqual((result["code"], result["missing_unit_ids"]), ("missing_unit", ["C"]))
                self.assertEqual(result["value_a" if a == "C" else "value_b"], "61.0")
                self.assertEqual(result["source_refs"]["C"], "/records/2")
                self.assertNotIn("difference", result)

    def test_missing_metric_groups_mixed_units_and_invalid_values(self):
        cases = [
            ({"unit": "g"}, "different_units"), ({"unit": "  "}, "missing_unit"),
            ({"metric": "purity"}, "different_metrics"), ({"metric": None}, "missing_metric"),
            ({"value": None}, "missing_value"), ({"value": "nd"}, "invalid_value"),
            ({"value": "NaN"}, "invalid_value"), ({"value": "1e61"}, "invalid_value"),
        ]
        for patch, code in cases:
            with self.subTest(patch=patch):
                data = copy.deepcopy(self.fixtures)
                data["records"][1]["result"].update(patch)
                result = self.compare(book=self.load(data))
                self.assertEqual(result["code"], code)
                self.assertNotIn("difference", result)
        self.fixtures["records"][1]["comparison_group"] = "other exercise"
        self.assertEqual(self.compare(book=self.load(self.fixtures))["code"], "different_groups")

    def test_nonpercentage_and_tied_measurements_keep_the_recorded_unit(self):
        for record in self.fixtures["records"][:2]:
            record["result"] = {"metric": "mass", "value": 0, "unit": "mg"}
        result = self.compare(book=self.load(self.fixtures))
        self.assertEqual((result["difference"], result["difference_unit"], result["direction"]), ("0", "mg", "equal"))
        self.fixtures["records"][1]["result"]["value"] = 0.2
        self.assertEqual(self.compare(book=self.load(self.fixtures))["difference"], "0.2")

    def test_large_scale_separation_does_not_round_the_difference(self):
        self.fixtures["records"][0]["result"]["value"] = "1e-60"
        self.fixtures["records"][1]["result"]["value"] = "1e60"
        expected = "9" * 60 + "." + "9" * 60
        self.assertEqual(self.compare(book=self.load(self.fixtures))["difference"], expected)

    def test_input_changed_on_disk_does_not_change_loaded_evidence(self):
        original = self.load(self.fixtures)
        old_digest = original.source_sha256
        self.fixtures["records"][1]["result"]["value"] = "100.0"
        changed = self.load(self.fixtures)
        self.assertNotEqual(changed.source_sha256, old_digest)
        self.assertEqual(self.compare(book=original)["difference"], "-13.5")
        self.assertEqual(self.compare(book=changed)["difference"], "22")

    def test_invalid_requests_return_structured_observations(self):
        for tool, args, code in [
            ("other", {}, "unknown_tool"),
            ("list_records", {"anything": 1}, "invalid_arguments"),
            ("read_record", {"id": []}, "invalid_arguments"),
            ("read_record", {"id": "absent"}, "unknown_record"),
            ("compare_results", {"a": "B"}, "invalid_arguments"),
            ("compare_results", {"a": "B", "b": "B"}, "same_record"),
            ("compare_results", {"a": "absent", "b": "B"}, "unknown_record"),
        ]:
            with self.subTest(tool=tool, args=args):
                res = self.book.run(tool, args)
                self.assertEqual((res["ok"], res["code"]), (False, code))
                self.assertTrue(res["source_sha256"])
                json.dumps(res, allow_nan=False)

    def test_bad_json_encoding_file_size_or_absent_input_rejected(self):
        for raw in [b"[]", b"{}", b'{"records":[]}', b"\xff", b"[NaN]", b" ",
                    b'{"records":[],"records":[]}', b"0" * (MAX_BYTES + 1)]:
            with self.subTest(n=len(raw)):
                self.path.write_bytes(raw)
                with self.assertRaises(ChemistryInputError):
                    ChemistryRecords.from_path(self.path)
        self.path.unlink()
        with self.assertRaises(ChemistryInputError):
            ChemistryRecords.from_path(self.path)

    def test_duplicate_ids_and_nonliteral_notes_rejected(self):
        for key, bad in [("id", "A"), ("notes", "a string"), ("notes", [{"nested": "object"}])]:
            with self.subTest(key=key, bad=bad):
                data = copy.deepcopy(self.fixtures)
                data["records"][1][key] = bad
                with self.assertRaises(ChemistryInputError):
                    self.load(data)


if __name__ == "__main__":
    unittest.main()

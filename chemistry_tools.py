"""Read one experiment-log snapshot and compare its recorded measurements."""

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path

MAX_BYTES = 256 * 1024
MAX_RECORDS = 100
DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


class ChemistryInputError(ValueError):
    """The experiment log could not be loaded."""


def _unique_keys(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ChemistryInputError(f"Duplicate JSON key {key!r}.")
        output[key] = value
    return output


def _text(value, name, limit, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, str) or len(value) > limit:
        raise ChemistryInputError(f"{name} must be a string of at most {limit} characters.")
    value = value.strip()
    if not value and optional:
        return None
    if not value or not value.isprintable():
        raise ChemistryInputError(f"{name} must contain printable text.")
    return value


def _finite_decimal(text):
    if not DECIMAL.fullmatch(text):
        raise ValueError("Expected a finite decimal measurement.")
    # Check the exponent text before constructing a Decimal.
    exponent = re.search(r"[eE]([+-]?[0-9]+)$", text)
    if exponent and abs(int(exponent.group(1))) > 60:
        raise ValueError("Measurement exponent exceeds the supported range.")
    value = Decimal(text)
    parts = value.as_tuple()
    if len(parts.digits) > 40 or abs(parts.exponent) > 60:
        raise ValueError("Measurement precision exceeds the supported range.")
    return value


def _display_difference(value):
    if not value:
        return "0"
    fixed = format(value, "f")
    return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed


@dataclass(frozen=True)
class Experiment:
    id: str
    title: str
    comparison_group: str
    metric: str | None
    value: str | None
    unit: str | None
    notes: tuple
    source_ref: str

    @classmethod
    def from_dict(cls, obj, index):
        if not isinstance(obj, dict) or not isinstance(obj.get("result"), dict):
            raise ChemistryInputError(f"Record {index} must be an object with a result object.")
        result = obj["result"]
        notes = obj.get("notes", [])
        if not isinstance(notes, list) or len(notes) > 8:
            raise ChemistryInputError(f"Record {index} notes must be a list with at most eight entries.")
        value = result.get("value")
        if isinstance(value, Decimal):
            value = str(value)
        return cls(
            _text(obj.get("id"), "id", 64),
            _text(obj.get("title"), "title", 160),
            _text(obj.get("comparison_group"), "comparison_group", 160),
            _text(result.get("metric"), "result.metric", 100, optional=True),
            _text(value, "result.value", 80, optional=True),
            _text(result.get("unit"), "result.unit", 40, optional=True),
            tuple(_text(note, "note", 500) for note in notes),
            f"/records/{index}",
        )

    def as_dict(self):
        return {
            "id": self.id, "title": self.title, "comparison_group": self.comparison_group,
            "result": {"metric": self.metric, "value": self.value, "unit": self.unit},
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ChemistryRecords:
    records: tuple
    source_sha256: str

    @classmethod
    def from_path(cls, path):
        try:
            with Path(path).open("rb") as source:
                raw = source.read(MAX_BYTES + 1)
        except OSError as exc:
            raise ChemistryInputError(f"Could not read the experiment log ({exc.strerror}).") from exc
        if len(raw) > MAX_BYTES:
            raise ChemistryInputError("The experiment log exceeds 256 KiB.")
        def reject_constant(token):
            raise ChemistryInputError(f"Invalid JSON numeric constant {token}.")

        try:
            data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_keys,
                              parse_float=Decimal, parse_int=Decimal, parse_constant=reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ArithmeticError) as exc:
            raise ChemistryInputError("Expected a UTF-8 JSON experiment log.") from exc
        items = data.get("records") if isinstance(data, dict) else None
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_RECORDS:
            raise ChemistryInputError(f"The log must contain 1 to {MAX_RECORDS} records.")
        records = tuple(Experiment.from_dict(obj, index) for index, obj in enumerate(items))
        if len({record.id for record in records}) != len(records):
            raise ChemistryInputError("Each record must have a unique ID.")
        return cls(records, hashlib.sha256(raw).hexdigest())

    def run(self, tool, arguments):
        base = {"tool": tool, "source_sha256": self.source_sha256}

        def failure(code, message, **details):
            return {**base, "ok": False, "code": code, "message": message, **details}

        schemas = {"list_records": set(), "read_record": {"id"}, "compare_results": {"a", "b"}}
        if not isinstance(tool, str) or tool not in schemas:
            return failure("unknown_tool", "Choose a supported record tool.")
        if (not isinstance(arguments, dict) or set(arguments) != schemas[tool]
                or any(not isinstance(v, str) or not v for v in arguments.values())):
            return failure("invalid_arguments", f"Expected nonempty string arguments {sorted(schemas[tool])}.")
        if tool == "list_records":
            return {**base, "ok": True, "records": [{"id": r.id, "title": r.title} for r in self.records]}
        by_id = {record.id: record for record in self.records}
        unknown = [value for value in arguments.values() if value not in by_id]
        if unknown:
            return failure("unknown_record", "Use list_records to find available IDs.", unknown_ids=unknown)
        if tool == "read_record":
            record = by_id[arguments["id"]]
            return {**base, "ok": True, "source_ref": record.source_ref, "record": record.as_dict()}

        a, b = by_id[arguments["a"]], by_id[arguments["b"]]
        base.update(a=a.id, b=b.id, value_a=a.value, value_b=b.value, unit_a=a.unit, unit_b=b.unit,
                    source_refs={a.id: a.source_ref, b.id: b.source_ref})
        if a.id == b.id:
            return failure("same_record", "Select two distinct records.")
        if a.comparison_group != b.comparison_group:
            return failure("different_groups", "The records have different comparison groups.",
                           group_a=a.comparison_group, group_b=b.comparison_group)
        if not a.metric or not b.metric:
            return failure("missing_metric", "A recorded measurement name is required for both outcomes.",
                           missing_metric_ids=[r.id for r in (a, b) if not r.metric])
        if a.metric != b.metric:
            return failure("different_metrics", "The recorded measurement names differ.",
                           metric_a=a.metric, metric_b=b.metric)
        base["metric"] = a.metric
        if not a.unit or not b.unit:
            return failure("missing_unit", "Supply the recorded unit before comparing the outcomes.",
                           missing_unit_ids=[r.id for r in (a, b) if not r.unit])
        if a.unit != b.unit:
            return failure("different_units", "The recorded units differ. No conversion is configured.")
        base["unit"] = a.unit
        if a.value is None or b.value is None:
            return failure("missing_value", "Both outcomes need a recorded numeric value.",
                           missing_value_ids=[r.id for r in (a, b) if r.value is None])
        operands = []
        for record in (a, b):
            try:
                operands.append(_finite_decimal(record.value))
            except ValueError as exc:
                return failure("invalid_value", str(exc), invalid_value_id=record.id)
        # At most 40 coefficient digits and exponent magnitude 60 per operand;
        # the difference needs at most 161 digits even at opposite scales.
        with localcontext() as context:
            context.prec = 170
            difference = operands[1] - operands[0]
        direction = "higher" if difference > 0 else "lower" if difference < 0 else "equal"
        return {
            **base, "ok": True, "difference": _display_difference(difference),
            "difference_unit": "percentage points" if a.unit in ("%", "percent") else a.unit,
            "direction": direction,
        }

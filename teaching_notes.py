"""Project-authored teaching notes. Background material carries its own provenance."""

import hashlib
import json


SOURCE = "https://openstax.org/books/chemistry-2e/pages/4-4-reaction-yields"
NOTES = {
    "transfer": {
        "title": "Recovery during transfer",
        "text": "Product left in a transfer vessel can reduce the amount collected. Visible residue makes transfer loss a possible explanation. Its composition and amount would need to be measured to test that explanation.",
        "source": SOURCE,
    },
    "yield": {
        "title": "Meaning of percent yield",
        "text": "Percent yield compares the amount actually obtained with the theoretical amount on the same basis. A lower reported yield alone does not distinguish incomplete reaction, collection losses, or other explanations.",
        "source": SOURCE,
    },
    "comparison": {
        "title": "Comparing two percentages",
        "text": "Subtract the baseline percentage from the comparison percentage to obtain a difference in percentage points. A relative percent change uses a different denominator and answers a different question.",
        "source": "Project arithmetic convention",
    },
}
NOTES_SHA256 = hashlib.sha256(json.dumps(NOTES, sort_keys=True).encode()).hexdigest()


def lookup_note(arguments):
    base = {"tool": "lookup_note", "notes_sha256": NOTES_SHA256}
    if (set(arguments) != {"topic"} or not isinstance(arguments["topic"], str)
            or arguments["topic"] not in NOTES):
        return {**base, "ok": False, "code": "unknown_topic",
                "message": "Choose one of the indexed topics.", "topics": list(NOTES)}
    topic = arguments["topic"]
    return {**base, "ok": True, "topic": topic, **NOTES[topic]}

"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_lab_spot_check.py"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_spot_check as P  # noqa: E402


def item(id, reason, cls):
    return {"id": id, "kind": "case", "audio": {"path": "x.wav", "startS": 0, "endS": 1}, "text": "", "title": reason,
            "triage": {"reasons": [reason], "class": cls}}


def test_spot_check_takes_new_sessions_only_every_disagreement_and_inserted_seams_first():
    items = [item("old-p01-x1", "disagree", "added_word")]
    items += [item(f"new-p01-d{n}", "disagree", "omitted_word") for n in range(3)]
    items += [item(f"new-p02-u{n}", "unsure", cls) for n, cls in enumerate(["omitted_word"] * 20 + ["wrong_word"] * 2 + ["clean"] * 30)]
    items += [item(f"new-p03-s{n}", "seam", cls) for n, cls in enumerate(["omitted_word"] * 15 + ["added_word"] * 4)]
    items += [item(f"new-p04-k{n}", "spot", "clean") for n in range(9)]
    doc = P.spot_check({"testSet": "t", "items": items}, "new-")
    got = Counter(P.pool(i) for i in doc["items"])
    assert got == {"disagree": 3, "unsure-edit": 12, "unsure-clean": 6, "seam": 10, "spot": 6}
    assert not any(i["id"].startswith("old-") for i in doc["items"])
    seams = [i["triage"]["class"] for i in doc["items"] if P.pool(i) == "seam"]
    assert seams[:4] == ["added_word"] * 4
    assert sum(i["triage"]["class"] == "wrong_word" for i in doc["items"]) == 2  # the rare class is not crowded out
    assert doc["maxBad"] == 1 and doc == P.spot_check({"testSet": "t", "items": items}, "new-")

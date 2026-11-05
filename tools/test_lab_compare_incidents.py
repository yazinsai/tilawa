import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_compare_incidents as C  # noqa: E402


def report(run, outcome):
    lab = {"id": "s1", "speaker": "a", "durationS": 60, "recitedS": 60, "split": "practice",
           "variants": {v: {"missed": int(outcome == "miss"), "falseWarnings": 0} for v in ("phone", "studio")}}
    real = {"id": "r1", "speaker": "b", "durationS": 9, "recitedS": 9, "split": "practice",
            "variants": {"phone": {"missed": 0, "falseWarnings": 0}}}  # realuser sessions are phone only
    cases = [{"key": f"s1/{v}/m", "kind": "mistake", "variant": v, "split": "practice", "session": "s1", "errorClass": "added_word",
              "word": "1:1:1", "startS": 1, "endS": 2, "outcome": outcome} for v in ("phone", "studio")]
    return {"testSet": "t", "run": run, "sha": run, "results": [lab, real], "cases": cases}


def test_sessions_without_a_studio_copy_are_compared_on_phone_only():
    out = C.compare(report("b", "miss"), report("n", "hit"))
    assert [len(out["variants"][v]["recoveries"]) for v in ("phone", "studio")] == [1, 1]

# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest -q tools/test_autolab_dashboard.py"""

import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolab_dashboard_core as V  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"
NOW = "2026-11-01T03:00:00+00:00"


def state():
    return json.loads((FIX / "autolab-state.json").read_text())


def events():
    return [json.loads(x) for x in (FIX / "autolab-events.jsonl").read_text().splitlines() if x.strip()]


def mainline():
    return json.loads((FIX / "autolab-ledger.json").read_text())["mainLine"]


def page(st=None, now=NOW, ml=None):
    return V.view(state() if st is None else st, events(), now, mainline() if ml is None else ml)


def test_heartbeat_fresh_and_stale():
    hb = page()["heartbeat"]
    assert hb == {"at": "2026-11-01T02:58:30+00:00", "ageSeconds": 90, "tickSeconds": 120, "stale": False, "paused": None, "ticks": 412}
    # stale only after three missed ticks: 360 s is still fine, 361 s is not
    assert not page(now="2026-11-01T03:04:30+00:00")["heartbeat"]["stale"]
    late = page(now="2026-11-01T03:04:31+00:00")["heartbeat"]
    assert late["stale"] and late["ageSeconds"] == 361
    st = state()
    st["heartbeat"], st["paused"] = None, "test data changed"
    hb = page(st)["heartbeat"]
    assert hb["stale"] and hb["ageSeconds"] is None and hb["paused"] == "test data changed"


def test_running_and_queue():
    v = page()
    assert [r["id"] for r in v["running"]] == ["checker-madd-length", "listener-vowel-prior"]
    r = v["running"][0]
    assert r["component"] == "checker" and r["tier"] == "exploit" and r["tierText"] == "Quick fix"
    assert r["state"] == "implementing" and r["stateText"] == "An agent is writing the change"
    assert r["ageSeconds"] == 50 * 60 and r["limitMinutes"] == 240 and not r["late"]
    assert r["executor"] == "gpt-5.2-codex" and r["hypothesis"].startswith("Treat a long madd")
    assert v["running"][1]["executor"] == "claude-opus-4.6" and v["running"][1]["limitMinutes"] == 90
    assert [(q["id"], q["state"]) for q in v["queue"]] == [("tracker-jump-hysteresis", "accepted"), ("listener-noise-augment", "proposed")]
    assert page(now="2026-11-01T06:11:00+00:00")["running"][0]["late"]


def test_needs_human():
    (h,) = page()["needsHuman"]
    assert h["id"] == "tracker-restart-window" and h["component"] == "tracker" and h["ageSeconds"] == 29 * 60
    assert "failed twice" in h["what"] and h["hypothesis"].startswith("Widen the restart")


def test_verdicts_newest_first_with_gate_in_plain_words():
    vs = page()["verdicts"]
    assert [(x["id"], x["state"]) for x in vs] == [("listener-blank-penalty", "discarded"), ("checker-added-word-confirm", "folded"),
                                                   ("checker-repeat-grace", "abandoned")]
    bad = vs[0]
    assert bad["journal"] == ".scratch/autolab/journal.md#listener-blank-penalty"
    assert bad["practice"] == {"where": "on its branch", "overall": -4, "component": -6, "expected": {"component": -6, "overall": -5}, "why": None}
    g = bad["gate"]
    assert g["verdict"] == "fail" and g["tuning"] and g["failed"] == ["Fewer wrong calls", "Not one reader's luck", "Breaks nothing else"]
    by = {c["name"]: c for c in g["checks"]}
    assert by["gain"]["text"] == "4 fewer wrong calls overall (needs 5 fewer or better); listener 6 fewer (may not get worse beyond noise)"
    assert by["spread"]["text"] == "Leaving out the reader who gained most (mp3quran-112) leaves no change: the gain rests on that reader"
    assert by["collateral"]["text"] == "9 cases put right, 5 newly wrong; 3 new false warnings on recitations with no mistakes; studio audio 4 more"
    assert by["hygiene"]["result"] == "passed" and by["significance"]["result"] == "note" and by["surah spread"]["result"] == "note"
    assert by["significance"]["text"] == "11 sessions changed; a cautious estimate of the change is 2 more"
    assert all(c["detail"] for c in g["checks"])
    ok = vs[1]
    assert next(c for c in ok["gate"]["checks"] if c["name"] == "collateral")["text"] == (
        "6 cases put right, 1 newly wrong; studio audio 1 fewer; false warnings on real users' clean recitation: no change")
    assert ok["gate"]["verdict"] == "pass" and not ok["gate"]["failed"] and ok["examTripwire"] == "pass"
    assert ok["practice"]["where"] == "after rebase" and ok["practice"]["overall"] == -5 and ok["practice"]["component"] == -5
    assert vs[2]["gate"] is None and vs[2]["practice"] is None and "240 minutes" in vs[2]["note"]


def test_gate_text_for_hygiene_failure_and_odd_checks():
    g = V.gate_view({"verdict": "fail", "tuning": False, "checks": [
        {"name": "hygiene", "ok": False, "blocking": True, "why": "protected: tools/lab_gate.py", "protected": ["tools/lab_gate.py"],
         "literals": [{"file": "packages/core/src/x.ts", "kind": "session id", "text": "lab-v1.5-p07"}]},
        {"name": "tests", "ok": False, "blocking": True, "why": "build or tests fail"},
        {"name": "something new", "ok": True, "blocking": False, "why": "as it says"}]})
    t = [c["text"] for c in g["checks"]]
    assert t == ["Changes the lab's own files (tools/lab_gate.py); Writes test-set details into the app (lab-v1.5-p07)",
                 "The build or its tests fail", "as it says"]
    assert g["checks"][2]["title"] == "something new"


def test_trend_is_practice_on_the_current_test_set():
    t = page()["trend"]
    assert t["testSet"] == "lab-v1.5"
    o = t["series"]["overall"]
    assert [p["n"] for p in o["points"]] == [107, 101, 98, 94, 89] and o["change"] == -18 and o["best"] == 89
    assert [p["n"] for p in t["series"]["checker"]["points"]] == [45, 43, 42, 42, 37]
    assert t["series"]["listener"]["last"] == 47 and t["series"]["tracker"]["first"] == 7
    assert o["points"][-1]["experiment"] == "checker-added-word-confirm"
    # briefs (an int per part) read the same as full scores
    brief = [{"at": "a", "testSet": "x", "sha": "s", "practice": {"wrongCalls": 10, "listener": 4, "tracker": 1, "checker": 5}}]
    assert V.trend(brief, "x")["series"]["checker"]["points"][0]["n"] == 5


def test_exam_numbers_never_reach_the_page():
    st = state()
    st["main"]["exam"] = {"wrongCalls": 777041}
    st["experiments"]["checker-added-word-confirm"]["results"]["fold"]["exam"] = {"overall": {"change": 777042}}
    st["experiments"]["checker-added-word-confirm"]["results"]["examScore"] = 777043
    out = json.dumps(V.view(st, events() + [{"at": NOW, "id": "x", "from": None, "to": "proposed", "note": "", "exam": 777044}], NOW, mainline()))
    assert not re.search(r"7770\d\d", out)
    assert '"exam"' not in out and "examScore" not in out
    assert V.scrub({"a": [{"exam": 1, "Exams": 2, "examTripwire": "pass", "b": 3}]}) == {"a": [{"examTripwire": "pass", "b": 3}]}


def test_portfolio_coverage_spend():
    v = page()
    p = v["portfolio"]
    # slots: running and queued, not finished ones and not the one waiting on you
    assert p["held"] == 4 and p["slots"] == 6 and p["byTier"] == {"exploit": 1, "explore": 2, "bet": 1}
    by = {c["key"]: c for c in p["components"]}
    assert by["listener"]["tiers"] == {"exploit": 0, "explore": 1, "bet": 1} and by["listener"]["sinceFold"] == 2
    assert by["checker"]["held"] == 1 and by["tracker"]["held"] == 1 and p["lastFoldAt"] == "2026-11-01T01:22:00+00:00"
    c = v["coverage"]
    assert c["counts"] == {"untouched": 2, "tried": 4, "fixed": 1}
    assert [r["state"] for r in c["clusters"]][:4] == ["tried"] * 4 and c["clusters"][0]["incidents"] == 14
    far = next(r for r in c["clusters"] if r["cluster"] == "tracker/restart/far-back")
    assert far["triedTwice"] and far["component"] == "tracker" and far["label"] == "restart · far-back"
    s = v["spend"]
    assert s["day"] == "2026-11-01" and s["atCap"] == ["Modal"]
    runs = s["rows"][0]
    assert runs == {"key": "agentRuns", "label": "Agent runs", "unit": "runs", "used": 11, "cap": 40, "share": 0.275, "over": False}
    st = state()
    del st["caps"]
    assert all(r["cap"] is None and r["share"] is None for r in page(st)["spend"]["rows"])


def test_activity_newest_first():
    a = page()["activity"]
    assert a[0]["to"] == "spend-cap" and a[0]["toText"] == "spend-cap" and "Modal dollars" in a[0]["note"]
    assert a[1] == {"at": "2026-11-01T02:52:00+00:00", "id": "listener-noise-augment", "from": None, "to": "proposed",
                    "toText": "Proposed, waiting to be checked", "note": "Fine-tune the listener on phone-noise augmented recitations so it hears quiet particles"}
    assert len(a) == V.ACTIVITY


def test_not_started():
    v = V.view(None, [], NOW, None)
    assert not v["started"] and v["running"] == v["queue"] == v["verdicts"] == [] and v["portfolio"] is None
    assert v["trend"] == {"testSet": None, "series": {}}
    # the ledger may exist before the supervisor's first tick
    assert V.view(None, [], NOW, mainline())["trend"]["series"]["overall"]["last"] == 89


def test_main_line_brief():
    m = page()["main"]
    assert m == {"sha": "a41c9e2", "run": "main-a41c9e2-20261101T012200Z", "wrongCalls": 89, "missed": 51, "falseWarnings": 38,
                 "parts": {"listener": 47, "tracker": 5, "checker": 37}}
    st = copy.deepcopy(state())
    st["main"] = None
    assert page(st)["main"] is None

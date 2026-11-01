# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest -q tools/test_autolab_graph.py"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autolab_graph_core as G  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"
NOW = "2026-11-01T03:00:00+00:00"


def load(name):
    return json.loads((FIX / name).read_text())


def state():
    return load("autolab-state.json")


def events():
    return [json.loads(x) for x in (FIX / "autolab-events.jsonl").read_text().splitlines() if x.strip()]


def lab():
    return load("autolab-lab-ledger.json")


def auto():
    # the shadow line's two scores after the hand-run lab's last fold (98 at 31 Oct 15:40)
    led = load("autolab-ledger.json")
    return {**led, "mainLine": led["mainLine"][-2:]}


def page(st=None, ev=None, now=NOW, report=None):
    return G.progress(state() if st is None else st, events() if ev is None else ev, lab(), auto(), now, report)


def by_id(view):
    return {b["id"]: b for b in view["branches"]}


def test_spine_joins_both_ledgers_and_marks_the_test_set_change():
    v = page()["views"]["all"]
    assert [(p["n"], p["source"], p["kind"]) for p in v["spine"]] == [
        (19, "lab", "score"), (14, "lab", "fold"), (107, "lab", "score"), (101, "lab", "fold"), (98, "lab", "fold"),
        (94, "autolab", "fold"), (89, "autolab", "fold")]
    assert v["boundaries"] == [{"at": "2026-10-31T06:36:32+00:00", "from": "lab-v1.4.v3", "to": "lab-v1.5", "before": 14, "after": 107}]
    assert v["spine"][3]["parts"] == {"listener": 55, "tracker": 7, "checker": 39}
    assert v["autolabStart"] == "2026-11-01T00:45:00+00:00"
    # the current test set's view starts at 107 and has no boundary
    t = page()["views"]["testSet"]
    assert t["spine"][0]["n"] == 107 and t["boundaries"] == [] and {p["testSet"] for p in t["spine"]} == {"lab-v1.5"}


def test_hand_run_lab_history():
    b = by_id(page()["views"]["testSet"])
    assert "checker-listener-noise" not in b  # an earlier test set: only in the full view
    j = b["checker-joint-additions"]
    assert j["source"] == "lab" and j["outcome"] == "folded" and j["tier"] is None
    assert (j["base"], j["result"], j["change"], j["foldAt"], j["foldN"]) == (107, 101, -6, "2026-10-31T11:21:49+00:00", 101)
    s = b["checker-skip-boundaries"]
    assert s["outcome"] == "ended" and s["state"] == "discarded" and s["change"] == 3 and s["foldAt"] is None
    p = b["listener-phone-channel"]
    assert p["base"] == 101 and p["change"] == -6 and p["reason"] == "User approved discard; introduces new false alarms."
    o = b["listener-targeted-consonants"]
    assert o["outcome"] == "open" and o["end"] == NOW and o["base"] == 98 and o["change"] == -6
    assert o["stateText"] == "Open in the hand-run lab, waiting for a decision"
    full = by_id(page()["views"]["all"])
    assert full["checker-listener-noise"]["foldN"] == 14 and full["checker-noise-gaps"]["reason"].startswith("superseded")


def test_autolab_branches_live_and_finished():
    v = page()["views"]["autolab"]
    b = by_id(v)
    assert {x["source"] for x in v["branches"]} == {"autolab"} and [p["n"] for p in v["spine"]] == [94, 89]
    assert v["counts"] == {"folded": 1, "ended": 2, "live": 5, "open": 0}
    f = b["checker-added-word-confirm"]
    assert f["outcome"] == "folded" and (f["foldAt"], f["foldN"]) == ("2026-11-01T01:22:00+00:00", 89) and f["change"] == -5
    assert f["start"] == "2026-10-31T21:05:00+00:00" and f["end"] == "2026-11-01T01:22:00+00:00" and f["tierText"] == "Quick fix"
    assert [s["state"] for s in f["stages"]] == ["proposed", "accepted", "implementing", "validating", "scoring", "folding", "folded"]
    m = b["checker-madd-length"]
    assert m["outcome"] == "live" and m["end"] == NOW and m["state"] == "implementing" and m["ageSeconds"] == 50 * 60 and m["reason"] == ""
    assert b["tracker-restart-window"]["waiting"] and b["tracker-restart-window"]["stateText"] == "Waiting for you"
    d = b["listener-blank-penalty"]
    assert d["outcome"] == "ended" and d["change"] == -4
    assert d["reason"] == "failed the gate: fewer wrong calls, not one reader's luck, breaks nothing else"
    assert b["checker-repeat-grace"]["reason"].startswith("implementing for more than 240 minutes")
    # base: the main-line score current when the branch left it (none in the autolab-only view before its first score)
    assert f["base"] is None and by_id(page()["views"]["testSet"])["checker-added-word-confirm"]["base"] == 98


def test_events_fill_in_experiments_state_no_longer_holds_and_drop_removed_ones():
    ev = events() + [
        {"at": "2026-11-01T02:56:00+00:00", "id": "tracker-ghost", "from": None, "to": "proposed", "note": "Hold the last ayah longer"},
        {"at": "2026-11-01T02:58:00+00:00", "id": "tracker-ghost", "from": "proposed", "to": "accepted", "note": ""},
        {"at": "2026-11-01T02:56:00+00:00", "id": "listener-dupe", "from": None, "to": "proposed", "note": "same card"},
        {"at": "2026-11-01T02:57:00+00:00", "id": "listener-dupe", "from": "proposed", "to": "discarded", "note": "restates itself"},
        {"at": "2026-11-01T02:58:00+00:00", "id": "listener-dupe", "from": "discarded", "to": "removed", "note": "judge compared the card with itself"}]
    b = by_id(page(ev=ev)["views"]["autolab"])
    g = b["tracker-ghost"]
    assert g["component"] == "tracker" and g["state"] == "accepted" and g["outcome"] == "live" and g["hypothesis"] == "Hold the last ayah longer"
    assert g["start"] == "2026-11-01T02:56:00+00:00" and g["ageSeconds"] == 120
    assert "listener-dupe" not in b


def test_lanes_never_overlap():
    v = page()["views"]["autolab"]
    lane = {b["id"]: b["lane"] for b in v["branches"]}
    assert lane == {"checker-repeat-grace": 0, "checker-added-word-confirm": 1, "listener-vowel-prior": 2, "listener-blank-penalty": 3,
                    "checker-madd-length": 0, "tracker-restart-window": 1, "tracker-jump-hysteresis": 3, "listener-noise-augment": 4}
    assert v["lanes"] == 5
    for view in page()["views"].values():
        bs = view["branches"]
        for a in bs:
            for b in bs:
                if a is not b and a["lane"] == b["lane"]:
                    assert G.ts(a["end"]) + G.timedelta(seconds=G.LANE_GAP_S) <= G.ts(b["start"]) or G.ts(b["end"]) + G.timedelta(seconds=G.LANE_GAP_S) <= G.ts(a["start"])
    # a lane frees only LANE_GAP_S after its branch ends; a live branch (ending now) never frees it
    t = lambda h, m=0: f"2026-11-01T{h:02d}:{m:02d}:00+00:00"
    bs = [{"id": "a", "start": t(1), "end": t(2)}, {"id": "b", "start": t(2, 10), "end": t(3)}, {"id": "c", "start": t(2, 20), "end": t(4)},
          {"id": "d", "start": t(3, 30), "end": t(9)}, {"id": "e", "start": t(5), "end": t(6)}]
    assert G.lanes(bs) == 2 and [b["lane"] for b in bs] == [0, 1, 0, 1, 0]
    bs[2]["start"] = t(2, 19)  # a's lane is still cooling down
    assert G.lanes(bs) == 3 and [b["lane"] for b in bs] == [0, 1, 2, 0, 1]


def test_quiet_stretches_become_breaks():
    ws = page()["views"]["all"]["windows"]
    assert [(w["from"], w["to"]) for w in ws] == [
        ("2026-10-30T14:03:25+00:00", "2026-10-30T17:40:26+00:00"),
        ("2026-10-31T06:26:32+00:00", "2026-10-31T15:58:22+00:00"),
        ("2026-10-31T18:50:00+00:00", NOW)]
    assert ws[0]["seconds"] == 3 * 3600 + 37 * 60 + 1
    assert len(page()["views"]["autolab"]["windows"]) == 1


def test_score_how_close_and_pace():
    s = page(report=load("autolab-score.json"))["score"]
    assert s["testSet"] == "lab-v1.5" and [p["n"] for p in s["series"]] == [107, 101, 98, 94, 89]
    assert s["current"]["wrongCalls"] == 89 and s["current"]["parts"] == {"listener": 47, "tracker": 5, "checker": 37}
    assert s["start"] == {"n": 107, "at": "2026-10-31T06:36:32+00:00"} and s["change"] == -18
    assert s["goal"] == {"n": 0, "source": "zero"} and s["toGo"] == 89 and s["teacher"] == 102
    assert s["rate"] == {"perDay": 21.2, "since": "2026-10-31T06:36:32+00:00", "days": 0.85}
    assert s["projection"]["days"] == 4.2 and s["lastDay"] is None
    assert s["autolab"] == {"start": 94, "at": "2026-11-01T00:45:00+00:00", "change": -5, "folds": 1}
    # a day later with no progress: the pace slows, the last day shows the change
    later = page(now="2026-11-02T12:00:00+00:00")["score"]
    assert later["lastDay"] == 0 and later["rate"]["perDay"] < 21.2 and later["projection"]["days"] > 4.2
    st = state()
    st["goal"] = 60
    g = page(st)["score"]
    assert g["goal"] == {"n": 60, "source": "state"} and g["toGo"] == 29
    st["goal"] = {"wrongCalls": 95}
    assert page(st)["score"]["toGo"] == 0 and page(st)["score"]["projection"] is None


def test_teacher_only_from_a_practice_split_on_the_same_test_set():
    r = load("autolab-score.json")
    assert G.teacher(r, "lab-v1.5") == 102
    assert G.teacher({**r, "testSet": "lab-v1.4.v3"}, "lab-v1.5") is None
    assert G.teacher({**r, "kind": "other"}, "lab-v1.5") is None
    assert G.teacher({"kind": "lab-score", "splits": {"exam": {"teacher": {"wrongCalls": 5}}}}, "lab-v1.5") is None


def test_exam_numbers_never_reach_the_progress_page():
    st = state()
    st["main"]["exam"] = {"wrongCalls": 777301}
    st["experiments"]["checker-added-word-confirm"]["results"]["fold"]["exam"] = {"overall": {"change": 777302}}
    st["experiments"]["checker-added-word-confirm"]["card"]["examTarget"] = 777303
    ev = events() + [{"at": NOW, "id": "x", "from": None, "to": "proposed", "note": "", "exam": 777304}]
    led = auto()
    led["mainLine"][-1]["exam"] = {"wrongCalls": 777305}
    out = json.dumps(G.progress(st, ev, lab(), led, NOW, load("autolab-score.json")))
    assert not re.search(r"777\d\d\d", out)
    assert '"exam' not in out.lower()


def test_every_branch_says_what_it_is_about_what_it_expects_and_what_came_of_it():
    explained = {"checker-joint-additions": "Added words now get lined up with both neighbours.", "checker-madd-length": "From the file."}
    st = state()
    st["experiments"]["checker-madd-length"]["card"]["eli5"] = "A long held vowel is not an extra word."
    v = page(st)
    b = by_id(G.progress(st, events(), lab(), auto(), NOW, None, explained)["views"]["testSet"])
    assert b["checker-joint-additions"]["eli5"] == {"text": "Added words now get lined up with both neighbours.", "plain": True}
    assert b["checker-madd-length"]["eli5"] == {"text": "A long held vowel is not an extra word.", "plain": True}  # the card beats the file
    assert b["listener-phone-channel"]["eli5"] == {"text": "Fine-tune the listener on phone-channel audio", "plain": False}
    j = b["checker-joint-additions"]
    assert j["expectText"] == "Expected: about 6 fewer wrong calls overall (6 fewer charged to the checker)"
    assert j["resultText"] == "Result: 6 fewer (expected 6 fewer), kept"
    assert b["listener-phone-channel"]["resultText"] == "Result: 6 fewer (expected 8 fewer), dropped: user approved discard; introduces new false alarms."
    assert b["checker-skip-boundaries"]["resultText"] == "Result: 3 more (expected 3 fewer), dropped"
    assert b["listener-targeted-consonants"]["resultText"] == "So far: 6 fewer (expected 6 fewer) on its branch, waiting for a decision"
    assert b["listener-blank-penalty"]["resultText"] == (
        "Result: 4 fewer (expected 5 fewer), dropped: failed the gate: fewer wrong calls, not one reader's luck, breaks nothing else")
    assert b["checker-repeat-grace"]["resultText"].startswith("Result: not scored, given up: implementing for more than 240 minutes")
    assert b["checker-madd-length"]["resultText"] is None and v["views"]["autolab"]["branches"][0]["eli5"]["text"]


def test_supervisor_fields_win_over_inference():
    st = state()
    st["experiments"]["checker-madd-length"]["base"] = {"sha": "a41c9e2", "wrongCalls": 91}
    st["main"]["teacherWrongCalls"] = 100
    led = auto()
    led["mainLine"] = led["mainLine"][:1]  # the fold's main-line score isn't in the ledger yet
    ev = events()
    ev[18] = {**ev[18], "wrongCalls": 88}
    assert ev[18]["to"] == "folded"
    v = G.progress(st, ev, lab(), led, NOW, load("autolab-score.json"))
    b = by_id(v["views"]["autolab"])
    assert b["checker-madd-length"]["base"] == 91
    f = b["checker-added-word-confirm"]
    assert f["foldN"] == 88 and f["foldAt"] == "2026-11-01T01:22:00+00:00"
    assert v["score"]["teacher"] == 100


def test_ticker_newest_first_with_components():
    t = page()["ticker"]
    assert len(t) == G.TICKER and t[0]["to"] == "spend-cap" and t[0]["id"] is None
    assert t[1]["id"] == "listener-noise-augment" and t[1]["component"] == "listener" and t[1]["toText"] == "Proposed, waiting to be checked"
    assert len({x["key"] for x in t}) == len(t)


def test_nothing_yet():
    v = G.progress(None, [], None, None, NOW)
    assert not v["started"] and v["heartbeat"] is None and v["score"]["current"] is None and v["score"]["series"] == []
    assert all(g["branches"] == [] and g["spine"] == [] and len(g["windows"]) == 1 for g in v["views"].values())
    # the hand-run lab's ledger alone already draws a line
    v = G.progress(None, [], lab(), None, NOW)
    assert v["testSet"] == "lab-v1.5" and v["score"]["current"]["wrongCalls"] == 98 and v["score"]["autolab"] is None


def test_split_gives_autolab_its_experiments_and_the_main_line_since_it_started():
    led = {"kind": "lab-ledger", "mainLine": [{"at": "2026-10-31T00:00:00+00:00", "experiment": "checker-old"},
                                              {"at": "2026-11-01T05:00:00+00:00", "experiment": None},
                                              {"at": "2026-11-01T06:00:00+00:00", "experiment": "checker-auto"},
                                              {"at": "2026-11-01T07:00:00+00:00", "experiment": "checker-hand"}],
           "experiments": {"checker-old": {}, "checker-auto": {}, "checker-hand": {}}}
    st = {"started": "2026-11-01T04:00:00+00:00", "experiments": {"checker-auto": {}}}
    lab, auto = G.split(led, st)
    assert [m["experiment"] for m in lab["mainLine"]] == ["checker-old", "checker-hand"] and set(lab["experiments"]) == {"checker-old", "checker-hand"}
    assert [m["experiment"] for m in auto["mainLine"]] == [None, "checker-auto"] and set(auto["experiments"]) == {"checker-auto"}
    assert G.split(led, None) == (led, None)

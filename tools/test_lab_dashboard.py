# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_lab_dashboard.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_dashboard_core as D  # noqa: E402
import lab_ledger as L  # noqa: E402
import lab_review_core as R  # noqa: E402
from test_lab_ledger import card, report, totals  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def full_totals(missed=10, fw=4, att=None):
    t = totals(missed, fw, att)
    t["components"]["tracker"] = {"onPlaceShare": 0.9, "onPlaceS": 540.0, "scoredS": 600.0, "falseJumps": 2,
                                  "oracleEvidence": {"onPlaceShare": 0.99, "onPlaceS": 594.0, "scoredS": 600.0, "falseJumps": 1}}
    t["teacher"] = {"wrongCalls": 12, "missed": 8, "falseWarnings": 4, "per5Min": 4.8, "per": 0.07}
    return t


def score(practice=None, exam=None, cases=()):
    return {**report(practice or full_totals(), exam or full_totals(8, 2), sha="m1"), "headline": {"split": "exam", "variant": "phone"},
            "recordedAt": "2026-10-28T13:00:00Z", "cases": list(cases)}


def ledger():
    led = L.empty()
    L.add_test_set(led, base="lab-v1.1", version=1, manifest="lab-v1.1/manifest-v1.json",
                   sessions={"practice": ["lab-v1.1-p01", "lab-v1.1-p02"], "exam": ["lab-v1.1-e01"]}, at="t0")
    L.record_main(led, L.from_report(report(totals(), totals(8, 2), sha="m1", ts="lab-v1.1"), ("practice", "exam")), at="2026-10-28T00:00:01+00:00")
    return led


def mistake(key="lab-v1.1-p01/phone/mistake:wrong_word/26:71:4/23.80", **kw):
    s = key.split("/")
    return {"key": key, "kind": "mistake", "session": s[0], "variant": s[1], "split": "practice", "errorClass": "wrong_word",
            "word": s[3], "startS": float(s[4]), "endS": float(s[4]) + 0.68, "outcome": "miss", "component": "checker",
            "oracle": "miss", "truePosition": "miss", "teacher": "miss", "trackerOnPlace": 1, "allFail": False,
            "review": {"path": f"lab-v1.1/sessions/{s[0]}.{s[1]}.wav", "startS": float(s[4]) - 3, "endS": float(s[4]) + 3.68}, **kw}


def warning(key="lab-v1.1-p02/phone/warning:added_word/2:5:1/50.00", **kw):
    s = key.split("/")
    return {"key": key, "kind": "false_warning", "session": s[0], "variant": s[1], "split": "practice", "errorClass": "added_word",
            "word": s[3], "timeS": float(s[4]), "confirmedS": float(s[4]) + 2, "nonError": None, "component": "listener",
            "oracle": False, "truePosition": True, "teacher": True, "trackerOnPlace": 0.5, "allFail": False,
            "review": {"path": f"lab-v1.1/sessions/{s[0]}.{s[1]}.wav", "startS": float(s[4]) - 4, "endS": float(s[4]) + 2}, **kw}


SDOC = {
    "id": "lab-v1.1-p01", "meta": {"reciter": "Khalid Abdulkafi"},
    "said": [{"startS": 20.0, "endS": 21.0, "text": "فَقَالُوا"}, {"startS": 23.795, "endS": 24.475, "text": "كَذَّبَتْ"},
             {"startS": 25.0, "endS": 26.0, "text": "أَصْنَامًا"}, {"startS": 40.0, "endS": 41.0, "text": "بَعِيدٌ"}],
    "events": [{"errorClass": "wrong_word", "location": {"surah": 26, "ayah": 71, "word": 4}, "startS": 23.795, "endS": 24.475,
                "expected": "فَنَظَلُّ", "observed": "كَذَّبَتْ", "splice": "x1"}],
    "splices": [{"id": "x1", "type": "wrong_word", "seams": [{"out": {"audio": "sources/mp3quran/22/026.wav", "srcS": 465.735}}]}],
}


def test_header_shows_the_exam_score_counts_and_pending_flags():
    led = ledger()
    h = D.header(score(), led=led, pending=2, check={"text": "listening check passed 40/40"})
    s = h["score"]
    assert (s["split"], s["audio"], s["wrongCalls"], s["missed"], s["falseWarnings"]) == ("exam", "phone", 10, 8, 2)
    assert h["testSet"] == "lab-v1.1" and h["pending"] == 2 and h["inLedger"] and h["check"]["text"].endswith("40/40")


def test_header_without_a_ledger_says_so_and_still_shows_a_run():
    h = D.header(None)
    assert h["score"] is None and not h["inLedger"] and h["baseline"] == "python3 tools/lab.py baseline"
    h = D.header(score())
    assert not h["inLedger"] and h["score"]["wrongCalls"] == 10 and h["testSet"] == "lab-v1"


def test_practice_only_runs_fall_back_to_all():
    sc = score()
    sc["splits"]["exam"]["sessions"] = 0
    assert D.headline_totals(sc)[0] == "all"


def test_cards_add_up_to_the_overall_and_carry_their_own_scores():
    t = full_totals(10, 4)
    cs = D.cards(t)
    assert [c["key"] for c in cs] == list(L.COMPONENTS)
    assert sum(c["wrongCalls"] for c in cs) == t["overall"]["wrongCalls"] == 14
    assert all(c["missed"] + c["falseWarnings"] == c["wrongCalls"] for c in cs)
    by = {c["key"]: c for c in cs}
    assert by["listener"]["own"]["text"] == "Mishears 80 of every 1,000 sounds"
    assert by["tracker"]["own"]["text"] == "Lost its place for 6 s of 10 min, 1 wrong jump"  # perfect hearing (oracle evidence)
    assert "60 s" in by["tracker"]["own"]["detail"] and by["checker"]["own"]["text"].startswith("4 wrong calls")
    b = D.bar(t)
    assert b["adds"] and b["total"] == 14 and abs(sum(p["share"] for p in b["parts"]) - 1) < 1e-9
    assert D.teacher(t)["mishears"] == 70 and D.teacher({"teacher": None}) is None


def test_experiments_waiting_first_with_expected_actual_and_exam_after_fold():
    led = ledger()
    base = L.main_score(led)
    for id in ("idle", "wait", "done"):
        L.add_experiment(led, card(id), branch=f"exp/{id}", worktree=f"/w/{id}")
    L.record_result(led, "wait", L.from_report(report(totals(missed=5), ts="lab-v1.1")), vs=base, at="t2")
    L.record_result(led, "done", L.from_report(report(totals(missed=4), ts="lab-v1.1")), vs=base, at="t2")
    L.record_result(led, "done", L.from_report(report(totals(missed=6), totals(5, 2), ts="lab-v1.1"), ("practice", "exam")), kind="rebased", vs=base, at="t3")
    L.decide(led, "done", "folded", at="t4", testSet="lab-v1.1")
    rows = D.experiment_rows(led)
    assert [(r["id"], r["state"]) for r in rows] == [("wait", L.WAITING), ("idle", "open"), ("done", "folded")]
    w = rows[0]
    assert w["practice"]["overall"] == -5 and w["practice"]["verdict"] == "better" and w["expected"] == {"component": -5, "overall": -3}
    assert w["next"] == ["python3 tools/lab.py fold wait", "python3 tools/lab.py discard wait"]
    assert rows[1]["practice"] is None and rows[1]["next"] == ["python3 tools/lab.py run /w/idle"]
    d = rows[2]
    assert d["exam"]["overall"] == -3 and d["rebased"]["overall"] == -4 and d["next"] == []


def test_noise_is_two_or_fewer():
    led = ledger()
    L.add_experiment(led, card("tiny"), branch="exp/tiny", worktree="/w/tiny")
    L.record_result(led, "tiny", L.from_report(report(totals(missed=8), ts="lab-v1.1")), vs=L.main_score(led))
    assert D.experiment_rows(led)[0]["practice"]["verdict"] == "noise"


def test_history_is_newest_first_with_each_fold_change():
    led = ledger()
    L.record_main(led, {**L.from_report(report(totals(), totals(5, 2), sha="m2", ts="lab-v1.1"), ("practice", "exam")), "experiment": "a", "note": "fewer"}, at="t5")
    L.add_test_set(led, base="lab-v1.1", version=2, manifest="m", sessions={"practice": [], "exam": []}, exclude="f", flags=3)
    L.record_main(led, L.from_report(report(totals(), totals(4, 2), sha="m2", ts="lab-v1.1.v2"), ("practice", "exam")), at="t6")
    h = D.history(led)
    assert [(x["kind"], x["change"]) for x in h] == [("new test set", None), ("fold", -3), ("first score", None)]
    assert h[1]["verdict"] == "better" and h[1]["experiment"] == "a" and h[1]["exam"]["wrongCalls"] == 7
    assert D.main_run(led) == led["mainLine"][-1]["run"] and D.main_run(None) is None
    assert D.header(None, led=led)["flagsApplied"] == 3


def test_base_of_strips_only_the_version_suffix():
    assert D.base_of("lab-v1.1") == "lab-v1.1" and D.base_of("lab-v1.1.v3") == "lab-v1.1" and D.base_of("lab-v1.v2") == "lab-v1"


def test_drilldown_is_practice_only_and_likely_bad_data_first():
    cases = [mistake(), warning(), mistake("lab-v1.1-p02/phone/mistake:wrong_word/2:9:1/80.00", allFail=True),
             mistake("lab-v1.1-p01/phone/mistake:added_word/26:80:1/90.00", errorClass="added_word", dataCheck={"pass": False, "check": "x", "reasons": ["pause-form donor"]}),
             mistake("lab-v1.1-e01/phone/mistake:wrong_word/28:35:7/54.75", split="exam"),
             mistake("lab-v1.1-p01/studio/mistake:wrong_word/26:71:4/23.80"),
             mistake("lab-v1.1-p01/phone/mistake:wrong_word/26:72:1/30.00", outcome="hit", component=None)]
    sc = score(cases=cases)
    assert {c["key"] for c in D.failing(sc)} == {cases[i]["key"] for i in (0, 1, 2, 3)}
    assert [c["key"] for c in D.failing(sc, "checker")] == [cases[i]["key"] for i in (0, 2, 3)]
    assert D.failing(sc, "checker", "studio") == [cases[5]]
    every = D.fails_everywhere(D.failing_keys(sc), [{cases[0]["key"], cases[3]["key"]}])
    assert every == {cases[0]["key"], cases[3]["key"]} and D.fails_everywhere({"a"}, []) == set()
    rows = D.order_cases(D.failing(sc, "checker"), every, {cases[2]["key"]: {"verdict": "bad_label"}})
    assert [r["key"] for r in rows] == [cases[3]["key"], cases[2]["key"], cases[0]["key"]]  # bad data, then wrong-even-with-perfect-hearing, then fails-everywhere
    assert [s["key"] for s in rows[0]["suspects"]] == ["dataCheck"] and "pause-form donor" in rows[0]["suspects"][0]["text"]
    assert [s["key"] for s in rows[0]["insights"]] == ["everywhere"]
    assert rows[1]["verdict"] == "bad_label" and rows[1]["verdictText"] == "Bad label" and rows[2]["verdict"] is None
    assert rows[0]["title"] == "The app didn't warn about an extra word" and rows[2]["where"] == "26:71, word 4"


def test_case_item_is_a_valid_review_item_with_text_label_and_originals():
    it = D.case_item(mistake(), SDOC)
    assert it["id"] == mistake()["key"] and it["kind"] == "splice"
    assert it["label"] == {"type": "wrong_word", "expected": "فَنَظَلُّ", "said": "كَذَّبَتْ"}
    assert it["text"] == "فَقَالُوا كَذَّبَتْ أَصْنَامًا" and it["audio"] == {"path": "lab-v1.1/sessions/lab-v1.1-p01.phone.wav", "startS": 20.8, "endS": 27.48}
    assert it["before"] == {"path": "sources/mp3quran/22/026.wav", "startS": 462.735, "endS": 465.735} and it["after"]["endS"] == 468.935
    assert "Khalid Abdulkafi" in it["context"] and "phone copy" in it["context"]
    w = D.case_item(warning(), None)
    assert w["kind"] == "case" and w["label"] == {"type": "clean"} and w["text"] == "" and "before" not in w
    assert D.case_item(warning(nonError="repeat"))["label"] == {"type": "repeat"}
    assert R.validate_queue(D.upsert_queue_item(None, "lab-v1.1", it)) == []
    assert D.session_doc_path(mistake()) == "lab-v1.1/sessions/lab-v1.1-p01.json"


def test_engine_story_says_what_happened_and_why_it_is_charged():
    m = D.engine_story(mistake())
    assert m[0].startswith("The labels say a wrong word was planted") and "missed it" in m[1] and m[-1].startswith("Charged to the checker")
    w = D.engine_story(warning(nonError="restart"))
    assert "on a restart, which is not a mistake" in w[0] and "stayed quiet" in w[1] and "50%" in w[2] and "listener" in w[-1]


def test_engine_facts_split_the_story_into_arms_and_the_charge():
    f = D.engine_facts(mistake())
    assert f["what"].startswith("The labels say a wrong word was planted")
    assert [(a["who"], a["did"], a["right"]) for a in f["arms"]] == [
        ("Perfect transcript", "missed it", False), ("App's listener", "missed it", False), ("Big reference listener", "missed it", False)]
    assert f["onPlace"] == 100 and f["charged"] == "checker" and f["why"] == D.WHY["checker"]
    w = D.engine_facts(warning())
    assert [a["right"] for a in w["arms"]] == [True, False, False] and w["arms"][0]["did"] == "stayed quiet" and w["onPlace"] == 50
    assert D.engine_facts(warning(teacher=None, trackerOnPlace=None))["arms"][2] == {"who": "Big reference listener", "did": "not run", "right": None}


def test_biggest_lever_is_the_largest_part_or_every_part_within_noise_of_it():
    assert D.levers([{"key": "listener", "wrongCalls": 74}, {"key": "tracker", "wrongCalls": 3}, {"key": "checker", "wrongCalls": 73}]) == ["listener", "checker"]
    assert D.levers([{"key": "listener", "wrongCalls": 10}, {"key": "tracker", "wrongCalls": 3}, {"key": "checker", "wrongCalls": 40}]) == ["checker"]
    assert D.levers([{"key": "listener", "wrongCalls": 0}, {"key": "tracker", "wrongCalls": 0}, {"key": "checker", "wrongCalls": 0}]) == []
    assert D.overview(score())["levers"] == D.levers(D.overview(score())["cards"])


def test_drilldown_rows_carry_their_strongest_reason_as_a_group():
    every = {mistake()["key"]}
    rows = D.order_cases([mistake(), warning(), mistake("lab-v1.1-p02/phone/mistake:wrong_word/2:9:1/80.00", allFail=True),
                          mistake("lab-v1.1-p01/phone/mistake:added_word/26:80:1/90.00", dataCheck={"reasons": ["x"]})], every)
    assert [r["group"] for r in rows] == ["dataCheck", "checkerAlways", "everywhere", "other"]
    assert rows[1]["suspects"] == [] and rows[1]["insights"][0]["key"] == "checkerAlways"


def test_data_check_reasons_drop_raw_scores_but_keep_them_for_the_tooltip():
    reason = "teacher doesn't hear the planted version cleanly (local PER 0.28)"
    s = D.suspects(mistake(dataCheck={"reasons": [reason]}))[0]
    assert s["text"] == "Failed an automatic data check: the big reference listener doesn't hear the planted version cleanly"
    assert s["raw"] == reason


def test_dashboard_verdicts_become_pending_flags_and_apply_once():
    q = D.upsert_queue_item(None, "lab-v1.1", D.case_item(mistake(), SDOC))
    q = D.upsert_queue_item(q, "lab-v1.1", D.case_item(warning(), None))
    q = D.upsert_queue_item(q, "lab-v1.1", D.case_item(warning(), None))
    assert len(q["items"]) == 2
    path = "lab-v1.1/review/dashboard.json"
    doc = R.empty_reviews("lab-v1.1", "me")
    R.upsert(doc, mistake()["key"], "bad_splice", "click at the join", "2026-10-28T20:00:00+00:00", path)
    R.upsert(doc, warning()["key"], "listener", "", "2026-10-28T20:01:00+00:00", path)
    sessions = {"lab-v1.1-p01", "lab-v1.1-p02", "lab-v1.1-e01"}
    flags, warn = D.pending_flags([doc], {path: q}, sessions, [])
    assert warn == [] and [(f["session"], f["verdict"], f["startS"], f["endS"]) for f in flags] == [("lab-v1.1-p01", "bad_splice", 20.8, 27.48)]
    assert D.pending_flags([doc], {path: q}, sessions, flags)[0] == []


def test_listening_check_takes_the_latest_verdict_across_reviewers():
    q = {"version": 1, "testSet": "lab-v1.1", "maxBad": 1, "items": [{"id": f"i{n}"} for n in range(3)]}
    me, you = R.empty_reviews("lab-v1.1", "me"), R.empty_reviews("lab-v1.1", "you")
    for n in range(3):
        R.upsert(me, f"i{n}", "ok", "", "t1")
    assert D.check_status(q, [me])["text"] == "listening check passed 3/3"
    R.upsert(you, "i0", "bad_audio", "", "t2")
    R.upsert(you, "i1", "bad_label", "", "t2")
    assert D.check_status(q, [me, you])["text"] == "listening check failed: 2 bad of 3 (at most 1)"
    assert D.check_status(q, [R.empty_reviews("lab-v1.1", "me")])["text"] == "listening check: 0 of 3 heard, 0 bad"
    assert D.check_status(None, []) is None


def test_release_reads_the_scoreboard_gates():
    r = D.release((REPO / "docs/benchmarks/scoreboard.md").read_text())
    assert r["title"].startswith("Release readiness") and r["columns"][:3] == ["Gate", "Now", "Bar"]
    assert len(r["gates"]) == 13 and all("Gate" in g for g in r["gates"]) and r["summary"].startswith("**")
    assert D.release("# nothing here") is None


def test_failing_with_perfect_hearing_is_not_called_bad_data():
    c = mistake("lab-v1.4-p01/phone/mistake:omitted_word/26:128:1/90.31", errorClass="omitted_word", allFail=True)
    assert D.suspects(c) == [] and [x["key"] for x in D.insights(c)] == ["checkerAlways"]


def test_case_context_says_how_far_into_the_session():
    c = mistake("lab-v1.4-p19/phone/mistake:added_word/51:30:1/81.20", errorClass="added_word", trackerOnPlace=1.0)
    assert D.case_context(c, {"durationS": 300.0}).endswith("into a 5.0 min session; the tracker was on the right place 100% of the time around here")
    assert D.case_context(c, None).endswith("into the session; the tracker was on the right place 100% of the time around here")

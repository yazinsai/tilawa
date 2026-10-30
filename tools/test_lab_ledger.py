# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_lab_ledger.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_ledger as L  # noqa: E402


def totals(missed=10, fw=4, att=None, checker=(3, 1), sessions=3):
    """A scorer `Totals` with the given overall counts; `att` splits wrong calls per component as (missed, false warnings)."""
    att = att or {"listener": (6, 1), "tracker": (2, 2), "checker": (2, 1)}
    return {"sessions": sessions, "recitedMinutes": 12.5,
            "overall": {"wrongCalls": missed + fw, "missed": missed, "falseWarnings": fw, "per5Min": 5.6, "mistakes": 30, "caught": 30 - missed, "falseWarningsOnNonErrors": 1},
            "components": {"listener": {"per": 0.08, "edits": 400, "phonemes": 5000},
                           "tracker": {"onPlaceShare": 0.9, "falseJumps": 2},
                           "checker": {"wrongCalls": sum(checker), "missed": checker[0], "falseWarnings": checker[1], "per5Min": 1.6, "falseWarningsOnNonErrors": 0}},
            "attribution": {**{c: sum(v) for c, v in att.items()}, "total": missed + fw,
                            "missed": {c: v[0] for c, v in att.items()}, "falseWarnings": {c: v[1] for c, v in att.items()}}}


def report(practice, exam=None, sha="abc1234", ts="lab-v1"):
    empty = {**totals(), "sessions": 0}
    return {"kind": "lab-score", "sha": sha, "run": f"x-{sha}-20261028T000000Z", "testSet": ts, "listener": L.DEFAULT_LISTENER,
            "sessions": {"excluded": [{"id": "p9", "reason": "no evidence"}]},
            "splits": {"practice": practice, "exam": exam or empty, "all": practice}}


def card(id="noop", component="checker", ec=-5, eo=-3):
    return L.new_card(id, component, "nothing changes", ec, eo, test_set="lab-v1", author="me", created="2026-10-28T00:00:00+00:00")


def ledger():
    led = L.empty()
    L.add_test_set(led, base="lab-v1", version=1, manifest="lab-v1/manifest-v1.json", sessions={"practice": ["p1", "p2"], "exam": ["e1"]}, at="t0")
    L.record_main(led, L.from_report(report(totals(), totals(8, 2), sha="m1"), ("practice", "exam")), at="2026-10-28T00:00:01+00:00")
    return led


def test_card_validation():
    c = card()
    assert c["expected"] == {"component": -5, "overall": -3} and c["listener"] == "spliced-clean-int8"
    for bad in [dict(id="No Spaces"), dict(component="decoder"), dict(hypothesis="two\nlines"), dict(expected={"component": 1.5, "overall": 0})]:
        with pytest.raises(ValueError):
            L.validate_card({**c, **bad})


def test_report_keeps_counts_and_refuses_a_missing_split():
    r = L.from_report(report(totals()))
    p = r["practice"]
    assert "exam" not in r and r["excluded"] == 1
    assert (p["wrongCalls"], p["missed"], p["falseWarnings"]) == (14, 10, 4)
    assert p["listener"]["wrongCalls"] == 7 and p["tracker"]["wrongCalls"] == 4 and p["checker"]["wrongCalls"] == 3
    assert sum(p[c]["wrongCalls"] for c in L.COMPONENTS) == p["wrongCalls"] and p["checker"]["alone"]["wrongCalls"] == 4
    assert p["listener"]["phonemeErrors"] == 400 and p["tracker"]["falseJumps"] == 2
    with pytest.raises(ValueError, match="no exam"):
        L.from_report(report(totals()), ("practice", "exam"))


def test_verdict_two_or_fewer_is_noise():
    assert [L.verdict(x) for x in (-3, -2, 0, 2, 3)] == ["better", "noise", "noise", "noise", "worse"]


def test_compare_rows_and_headline():
    base = L.score(totals())
    same = L.compare(base, base, card())
    assert all(r["change"] == 0 and r["verdict"] == "noise" for r in same.values())
    assert same["checker"]["expected"] == -5 and same["overall"]["expected"] == -3 and "expected" not in same["listener"]
    assert L.headline(same, card()) == "noise"
    better = L.compare(L.score(totals(missed=6, att={"listener": (6, 1), "tracker": (2, 2), "checker": (0, 0)}, checker=(0, 0))), base, card())
    assert better["overall"]["change"] == -4 and better["checker"]["change"] == -3 and better["checker"]["verdict"] == "better"
    assert L.headline(better, card()) == "better" and L.gains(better, card()) == ["overall", "checker"]
    only = L.compare(L.score(totals(missed=9, att={"listener": (6, 1), "tracker": (2, 2), "checker": (0, 0)})), base, card())
    assert L.headline(only, card()) == "noise overall, checker better"
    lines = L.format_compare(better, card())
    assert "lost by checker <" in lines[4] and "(expected -5)" in lines[4] and "14 (10 missed + 4 false)" in lines[1]
    d = L.format_details(L.score(totals(checker=(0, 0))), base)
    assert "phoneme errors 400 (same)" in d and "perfect inputs 4 (3 missed + 1 false) -> 0 (0 missed + 0 false)" in d


def test_status_waiting_first_and_decided_hidden():
    led = ledger()
    for id in ("a", "b", "c"):
        L.add_experiment(led, {**card(id), "created": f"2026-10-28T0{id == 'a'}:00:00+00:00"}, branch=f"exp/{id}", worktree=f"/w/{id}")
    L.record_result(led, "b", L.from_report(report(totals(missed=4))))
    L.decide(led, "c", "discarded", testSet="lab-v1")
    rows = L.status_rows(led)
    assert [(r["id"], r["state"]) for r in rows] == [("b", L.WAITING), ("a", "open")]
    assert rows[0]["actual"] == {"component": 0, "overall": -6} and rows[0]["verdict"] == "better"
    assert [r["id"] for r in L.status_rows(led, all=True)] == ["b", "a", "c"]
    out = L.format_status(led, rows)
    assert out[0].startswith("test set lab-v1; main line m1: 14 wrong calls") and "waiting for your decision" in out[3]
    with pytest.raises(ValueError, match="already exists"):
        L.add_experiment(led, card("a"), branch="exp/a", worktree="/w/a")


def test_new_test_set_version_needs_new_results():
    led = ledger()
    L.add_experiment(led, card(), branch="exp/noop", worktree="/w")
    L.record_result(led, "noop", L.from_report(report(totals())))
    assert L.shown_state(led["experiments"]["noop"], "lab-v1") == L.WAITING
    ts = L.add_test_set(led, base="lab-v1", version=2, manifest="lab-v1/manifest-v1.json", exclude="lab-v1/flags-v2.json",
                        sessions=led["testSets"]["lab-v1"]["sessions"], flags=1)
    assert ts == "lab-v1.v2" and led["testSet"] == ts and led["testSets"][ts]["previous"] == "lab-v1"
    assert L.shown_state(led["experiments"]["noop"], ts) == "open" and L.main_score(led) is None
    assert L.main_score(led, "lab-v1")["sha"] == "m1"


def test_fold_check():
    c, base = card(), L.score(totals())
    gain = L.compare(L.score(totals(missed=5, checker=(0, 0), att={"listener": (5, 1), "tracker": (0, 2), "checker": (0, 0)})), base, c)
    noise = L.compare(base, base, c)
    assert L.fold_check(c, gain, gain)[0] == "ok"
    state, why = L.fold_check(c, gain, noise)
    assert state == "superseded" and "gone after rebase" in why and "checker +0" in why
    assert L.fold_check(c, noise, noise)[0] == "no-gain"
    partial = L.compare(L.score(totals(missed=9, att={"listener": (6, 1), "tracker": (2, 2), "checker": (0, 0)})), base, c)
    assert L.fold_check(c, gain, partial) == ("ok", "keeps its checker gain (lost overall)")


def test_pending_flags_take_the_latest_bad_verdict_with_its_window():
    queue = {"items": [
        {"id": "lab-v1-p01-x3", "audio": {"path": "lab-v1/sessions/lab-v1-p01.studio.wav", "startS": 10.0, "endS": 16.5}},
        {"id": "lab-v1-p02-phone-53", "audio": {"path": "lab-v1/sessions/lab-v1-p02.phone.wav", "startS": 53.0, "endS": 61.0}},
        {"id": "odd", "audio": {"path": "lab-v1/sessions/lab-v1-e01.phone.wav", "startS": 1.0, "endS": 2.0}},
        {"id": "gone", "audio": {"path": "elsewhere/x.wav", "startS": 1.0, "endS": 2.0}},
    ]}
    rec = lambda v, at: {"verdict": v, "note": "click", "at": at, "queue": "lab-v1/review/v1-check.json", "history": []}
    reviews = [
        {"reviewer": "me", "items": {"lab-v1-p01-x3": rec("bad_splice", "2026-10-28T07:00:00"), "lab-v1-p02-phone-53": rec("bad_audio", "2026-10-28T07:00:00"),
                                     "odd": rec("bad_label", "2026-10-28T07:00:00"), "gone": rec("bad_label", "2026-10-28T07:00:00")}},
        {"reviewer": "asr", "items": {"lab-v1-p02-phone-53": rec("ok", "2026-10-28T08:00:00"), "lab-v1-p01-x3": rec("listener", "2026-10-28T06:00:00")}},
    ]
    sessions = {"lab-v1-p01", "lab-v1-p02", "lab-v1-e01"}
    flags, warn = L.pending_flags(reviews, {"lab-v1/review/v1-check.json": queue}, sessions, [])
    assert [(f["item"], f["session"], f["verdict"], f["reviewer"]) for f in flags] == [
        ("lab-v1-p01-x3", "lab-v1-p01", "bad_splice", "me"), ("odd", "lab-v1-e01", "bad_label", "me")]
    assert flags[0]["startS"] == 10.0 and flags[0]["endS"] == 16.5 and len(warn) == 1 and "gone" in warn[0]
    again, _ = L.pending_flags(reviews, {"lab-v1/review/v1-check.json": queue}, sessions, flags)
    assert again == []
    doc = L.flags_doc("lab-v1", 2, flags[::-1], at="t")
    assert doc["testSet"] == "lab-v1.v2" and doc["base"] == "lab-v1" and doc["kind"] == "lab-flags"
    assert [e["item"] for e in doc["excluded"]] == ["odd", "lab-v1-p01-x3"]


def test_session_of_case_keys_and_dotted_session_ids():
    sessions = {"lab-v1.1-p01", "lab-v1.1-p010", "lab-v1.1-e02"}
    case = lambda id, path: {"id": id, "audio": {"path": path, "startS": 0, "endS": 1}}
    assert L.session_of(case("lab-v1.1-p01/phone/mistake:wrong_word/1:2:3/4.5", "x/other.wav"), sessions) == "lab-v1.1-p01"
    assert L.session_of(case("lab-v1.1-p01-x3", "x/other.wav"), sessions) == "lab-v1.1-p01"
    assert L.session_of(case("anything", "lab-v1.1/sessions/lab-v1.1-p010.phone.wav"), sessions) == "lab-v1.1-p010"
    assert L.session_of(case("anything", "lab-v1.1/sessions/lab-v1.1-e02.studio.wav"), sessions) == "lab-v1.1-e02"
    assert L.session_of(case("anything", "lab-v1.1/sessions/lab-v1.1-e09.wav"), sessions) is None


def test_summary_has_one_row_per_main_line_change_and_decided_experiment():
    led = ledger()
    L.add_experiment(led, card(), branch="exp/noop", worktree="/w")
    L.add_experiment(led, card("open-one"), branch="exp/open-one", worktree="/w2")
    L.record_result(led, "noop", L.from_report(report(totals(missed=5))), at="t2")
    L.record_result(led, "noop", L.from_report(report(totals(missed=6), totals(7, 2)), ("practice", "exam")), kind="rebased", at="t3")
    L.decide(led, "noop", "folded", at="t4", testSet="lab-v1")
    L.record_main(led, {**L.from_report(report(totals(missed=6), totals(7, 2), sha="m2"), ("practice", "exam")), "experiment": "noop"}, at="t5")
    s = L.summary(led)
    assert [(r["kind"], r.get("id") or r["sha"]) for r in s["rows"]] == [("main", "m1"), ("experiment", "noop"), ("main", "m2")]
    row = s["rows"][1]
    assert row["branch"]["wrongCalls"] == 9 and row["rebased"]["practice"]["wrongCalls"] == 10 and row["rebased"]["exam"]["wrongCalls"] == 9
    assert row["branch"]["checker"] == 3 and s["rows"][2]["experiment"] == "noop"


def test_results_compare_with_the_main_line_they_were_measured_on():
    led = ledger()
    first = L.main_score(led)
    L.add_experiment(led, card("a"), branch="exp/a", worktree="/w/a")
    L.add_experiment(led, card("b"), branch="exp/b", worktree="/w/b")
    L.record_result(led, "a", L.from_report(report(totals(missed=5))), vs=first)
    L.record_result(led, "b", L.from_report(report(totals(missed=5))))
    L.record_main(led, L.from_report(report(totals(missed=5), totals(8, 2), sha="m2"), ("practice", "exam")))
    assert L.base_for(led, led["experiments"]["a"]["results"][0]) is first
    assert L.base_for(led, led["experiments"]["b"]["results"][0]) is first
    rows = L.status_rows(led)
    assert [r["actual"]["overall"] for r in rows] == [-5, -5]
    assert all(r["state"] == L.WAITING + " (main line moved since; fold re-checks)" for r in rows)

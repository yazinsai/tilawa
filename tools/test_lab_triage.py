"""uv run --python 3.12 --with pytest --with numpy --with rapidfuzz python -m pytest -q tools/test_lab_triage.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_dashboard_core as D  # noqa: E402
import lab_ledger as L  # noqa: E402
import lab_sessions as LS  # noqa: E402
import lab_triage_core as T  # noqa: E402
from lab_review_core import tally, validate_queue  # noqa: E402

WORDS = ["قُلْ", "هُوَ", "ٱللَّهُ", "أَحَدٌ", "ٱلصَّمَدُ", "لَمْ", "يَلِدْ", "وَلَمْ", "يُولَدْ", "يَكُن", "لَّهُۥ", "كُفُوًا", "رَبِّ", "ٱلنَّاسِ",
         "مَلِكِ", "إِلَـٰهِ", "شَرِّ", "ٱلْوَسْوَاسِ", "ٱلْخَنَّاسِ", "ٱلَّذِى", "يُوَسْوِسُ", "فِى", "صُدُورِ", "ٱلْجِنَّةِ", "وَٱلنَّاسِ",
         "ٱلْفَلَقِ", "خَلَقَ", "غَاسِقٍ", "وَقَبَ", "ٱلنَّفَّـٰثَـٰتِ", "ٱلْعُقَدِ", "حَاسِدٍ", "حَسَدَ", "أَعُوذُ", "بِرَبِّ", "مِن", "مَا", "إِذَا"]


def session(n=60, seam_at=24, wrong_at=None):
    """n ref words of 0.5 s back to back; segments split at word seam_at; optionally a planted wrong word at wrong_at."""
    said = []
    for k in range(n):
        w = {"startS": k * 0.5, "endS": k * 0.5 + 0.45, "at": {"surah": 112, "ayah": 1 + k // 10, "word": 1 + k % 10},
             "from": {"audio": "sources/a.wav", "surah": 112, "ayah": 1 + k // 10, "word": 1 + k % 10}, "text": WORDS[k % len(WORDS)], "role": "ref"}
        said.append(w)
    splices = []
    if wrong_at is not None:
        said[wrong_at] = {**said[wrong_at], "at": None, "text": "مَلِكِ", "role": "wrong", "op": "x1"}
        t = seam_at * 0.5
        splices.append({"id": "x1", "type": "wrong_word", "mutashabihat": False, "formsOtherAyah": False,
                        "location": {"surah": 112, "ayah": 1 + wrong_at // 10, "word": 1 + wrong_at % 10},
                        "expected": WORDS[wrong_at % len(WORDS)], "said": "مَلِكِ", "startS": said[wrong_at]["startS"], "endS": said[wrong_at]["endS"],
                        "host": {"audio": "sources/a.wav"},
                        "seams": [{"outS": t, "out": {"audio": "sources/a.wav", "srcS": t, "boundary": "pause", "after": "112:3:4", "piece": "host"},
                                   "in": {"audio": "sources/a.wav", "boundary": "pause", "before": "112:3:5", "piece": "donor"}},
                                  {"outS": t + 0.5, "out": {"audio": "sources/a.wav", "boundary": "pause", "after": "112:3:5", "piece": "donor"},
                                   "in": {"audio": "sources/a.wav", "boundary": "pause", "before": "112:3:6", "piece": "host"}}]})
    return {"id": "lab-x-p01", "durationS": n * 0.5, "said": said, "splices": splices,
            "segments": [{"startS": 0.0}, {"startS": seam_at * 0.5}] + ([{"startS": seam_at * 0.5 + 0.5}] if wrong_at is not None else []),
            "meta": {"audio": {"studio": "sessions/lab-x-p01.studio.wav"}, "reciter": "R"}}


def scores(cases, nem, fc):
    """Per-model scores giving each case the requested decision ('match' | 'mismatch' | 'unsure' | None for no score)."""
    val = {"match": (0.0, -10.0), "mismatch": (-10.0, 0.0), "unsure": (0.0, -0.5)}
    out = {"nemotron": {}, "fastconformer": {}}
    for c in cases:
        for m, want in (("nemotron", nem(c)), ("fastconformer", fc(c))):
            if want:
                a, b = val[want]
                out[m][c["id"]] = {"claimed": a, "alternative": b}
    return out


def test_outcome_and_pairwise():
    assert T.outcome("match", "match") == "auto-accepted"
    assert T.outcome("mismatch", "mismatch") == "both-mismatch"
    assert T.outcome("match", "mismatch") == T.outcome("mismatch", "match") == "disagree"
    assert T.outcome("match", "unsure") == T.outcome("unsure", "mismatch") == T.outcome("unsure", "unsure") == "unsure"
    assert T.pairwise(-10.0, -12.0, 2.0)["decision"] == "match"
    assert T.pairwise(-10.0, -11.0, 2.0) == {"decision": "unsure", "claimed": -10.0, "alternative": -11.0, "diff": 1.0}
    assert T.pairwise(-14.0, -10.0, 2.0)["decision"] == "mismatch"
    # no CTC path through the window: that side is infinitely worse
    assert T.pairwise(-10.0, None, 2.0)["decision"] == "match"
    assert T.pairwise(None, -10.0, 2.0)["decision"] == "mismatch"
    assert T.pairwise(None, None, 2.0)["decision"] == "unsure"


def test_spot_sample_is_seeded_order_free_and_fixed_size():
    ids = [f"c{k}" for k in range(31)]
    a = T.spot_sample(ids, 15, 17, "lab-x:spot:clean")
    assert a == T.spot_sample(list(reversed(ids)), 15, 17, "lab-x:spot:clean") and len(a) == 15
    assert a != T.spot_sample(ids, 15, 18, "lab-x:spot:clean")
    assert T.spot_sample([], 15, 17, "t") == [] and T.spot_sample(["a", "b"], 15, 17, "t") == ["a", "b"]
    assert [T.seam_count(n) for n in (0, 1, 31, 200, 360)] == [0, 1, 4, 20, 20]


def test_max_bad_scales_one_per_forty():
    assert [T.max_bad(n) for n in (1, 39, 40, 79, 80, 250)] == [1, 1, 1, 1, 2, 6]


def test_clean_runs_stop_at_seams_and_planted_words():
    s = session(60, seam_at=24, wrong_at=24)
    runs = T.clean_runs(s)
    assert [(r[0]["startS"], r[-1]["startS"]) for r in runs] == [(0.0, 11.5), (12.5, 29.5)]
    assert all(w["role"] == "ref" for r in runs for w in r)


def test_split_run_cuts_at_words_and_merges_short_windows():
    s = session(60, seam_at=59)
    run = T.clean_runs(s)[0]
    parts = T.split_run(run)
    assert [w for p in parts for w in p] == run  # every word once, in order
    assert len(parts) == round((run[-1]["endS"] - run[0]["startS"]) / T.CLEAN_TARGET_S)
    assert T.split_run(run[:4]) == [run[:4]]
    assert all(len(p) >= T.CLEAN_MIN_WORDS for p in T.split_run(run[:7], target=0.9))


def test_session_cases_cover_every_edit_and_clean_word():
    s = session(60, seam_at=24, wrong_at=24)
    cases = T.session_cases(s, "lab-x")
    edit = [c for c in cases if c["group"] == "edit"]
    assert [c["id"] for c in edit] == ["lab-x-p01-x1"]
    assert edit[0]["item"] == LS.review_splice(s, s["splices"][0], "lab-x")  # the v1 listening check's window and id
    assert edit[0]["alternative"].split() == [WORDS[k % len(WORDS)] for k in range(19, 30)]  # planted word back to the expected
    clean = [c for c in cases if c["group"] == "clean"]
    covered = [w for c in clean for w in c["item"]["text"].split()]
    assert len(covered) == sum(len(r) for r in T.clean_runs(s))
    for c in clean:
        assert c["claimed"] == c["item"]["text"] and c["alternative"] != c["claimed"]
        assert not c["alternativeOp"].startswith("text drops")  # CLEAN_OPS: no deletions
        assert abs(len(c["alternative"].split()) - len(c["claimed"].split())) <= 1
    assert T.session_cases(s, "lab-x") == cases  # seeded by case id
    assert len({c["id"] for c in cases}) == len(cases)


def triaged(nem, fc, reviews=None, earlier=None, spot=(15, 15), seam_spot=(0.1, 20)):
    s = session(60, seam_at=24, wrong_at=24)
    cases = T.session_cases(s, "lab-x")
    return cases, T.triage(cases, scores(cases, nem, fc), "lab-x", spot, seam_spot, 17, reviews, earlier)


def test_triage_routes_each_outcome_to_the_queue_with_a_plain_title():
    first_clean = "lab-x-p01-clean-0-0"
    nem = lambda c: "mismatch" if c["group"] == "edit" else "match"
    fc = lambda c: "mismatch" if c["group"] == "edit" else ("mismatch" if c["id"] == first_clean else None if c["id"].endswith("-1-1") else "match")
    cases, r = triaged(nem, fc, spot=(0, 0), seam_spot=(0.0, 0))
    out = {c["id"]: c["outcome"] for c in r["cases"]}
    assert out["lab-x-p01-x1"] == "both-mismatch" and out[first_clean] == "disagree" and out["lab-x-p01-clean-1-1"] == "unsure"
    q = r["queue"]
    assert validate_queue(q) == []
    assert [(it["id"], it["title"]) for it in q["items"]] == [
        ("lab-x-p01-x1", "Both models think the label is wrong — please confirm"),
        (first_clean, "The two models disagree"),
        ("lab-x-p01-clean-1-1", "A model couldn't tell — please check")]
    assert q["items"][0]["kind"] == "splice" and "before" in q["items"][0]  # the splice item keeps its host audio
    assert q["items"][0]["context"].startswith("Nemotron: the text without the planted mistake fits better than the label by 10.0")
    assert "FastConformer: a one-word change of the text fits better than the label" in q["items"][1]["context"]
    assert q["maxBad"] == 1 and r["counts"]["queue"]["byFirstReason"]["both-mismatch"] == 1
    assert r["counts"]["outcomes"]["auto-accepted"] == len(cases) - 3


def test_spot_and_seam_checks_merge_into_one_item_per_case():
    cases, r = triaged(lambda c: "match", lambda c: "match", spot=(1, 3), seam_spot=(1.0, 20))
    q = r["queue"]
    assert validate_queue(q) == []
    seam = next(it for it in q["items"] if it["id"] == "lab-x-p01-x1")
    assert r["seamIds"] == ["lab-x-p01-x1"]
    assert seam["title"].startswith("Seam spot check") and seam["triage"]["reasons"][0] == "seam"
    spot = [it for it in q["items"] if "spot" in it["triage"]["reasons"]]
    assert len(spot) == len(r["spotIds"]) == 4 and "lab-x-p01-x1" in r["spotIds"]  # the one edit and 3 of the clean windows
    assert "15 accepted" not in spot[0]["context"] and any("3 accepted clean windows are picked at random" in it["context"] for it in spot)
    assert len({it["id"] for it in q["items"]}) == len(q["items"])
    assert all(it["title"] == "Random spot check" for it in spot if it["triage"]["reasons"] == ["spot"])
    assert any(it["title"] == "Seam spot check · also a random spot check" for it in q["items"]) == ("lab-x-p01-x1" in r["spotIds"])
    _, again = triaged(lambda c: "match", lambda c: "match", spot=(1, 3), seam_spot=(1.0, 20))
    assert again["queue"] == q


def test_seam_pools_sample_inserted_audio_apart():
    planted = [{"id": f"i{k}", "class": c} for k, c in enumerate(["wrong_word", "added_word", "mutashabihat"] * 50)]
    planted += [{"id": f"o{k}", "class": c} for k, c in enumerate(["omitted_word", "skipped_passage", "repeat", "restart"] * 30)]
    got = T.seam_sample(planted, T.SEAM_POOLS, 17, "lab-x")
    assert sum(i.startswith("i") for i in got) == 30 and sum(i.startswith("o") for i in got) == 12  # 20% of 150 capped at 30; 10% of 120
    assert got == T.seam_sample(list(reversed(planted)), T.SEAM_POOLS, 17, "lab-x")
    few = [p for p in planted if p["id"] in ("i0", "i1", "i2", "i3", "i4")] + planted[150:160]
    got = T.seam_sample(few, T.SEAM_POOLS, 17, "lab-x")
    assert sum(i.startswith("i") for i in got) == 1 and sum(i.startswith("o") for i in got) == 1
    assert T.seam_sample(planted, (0.1, 20), 17, "lab-x") == set(T.spot_sample([p["id"] for p in planted], 20, 17, "lab-x:seam"))
    _, r = triaged(lambda c: "match", lambda c: "match", spot=(0, 0), seam_spot=T.SEAM_POOLS)
    assert r["seamIds"] == ["lab-x-p01-x1"] and r["counts"]["queue"]["seamByPool"] == {"inserted": 1}


def test_confirm_queue_holds_only_the_both_mismatch_items():
    edit_bad = lambda c: "mismatch" if c["group"] == "edit" else "match"
    _, r = triaged(edit_bad, edit_bad, spot=(1, 1), seam_spot=(1.0, 20))
    q = T.confirm_queue(r["queue"])
    assert q["maxBad"] == 0 and q["testSet"] == "lab-x" and validate_queue(q) == []
    assert [it["id"] for it in q["items"]] == ["lab-x-p01-x1"] and r["counts"]["queue"]["byFirstReason"]["both-mismatch"] == 1
    assert len(r["queue"]["items"]) > 1


def test_earlier_verdicts_carry_over_and_count_false_alarms():
    s = session(60, seam_at=24, wrong_at=24)
    earlier_item = LS.review_splice(s, s["splices"][0], "lab-x")
    reviews = {"lab-x-p01-x1": {"verdict": "ok", "at": "2026-10-28T07:00:00", "queue": "lab-x/review/v1-check.json", "reviewer": "me"}}
    _, r = triaged(lambda c: "mismatch" if c["group"] == "edit" else "match", lambda c: "mismatch" if c["group"] == "edit" else "match",
                   reviews, {earlier_item["id"]: earlier_item}, spot=(0, 0), seam_spot=(0.0, 0))
    it = r["queue"]["items"][0]
    assert it["triage"]["carriedOver"] == {"verdict": "ok", "note": "", "at": "2026-10-28T07:00:00", "queue": "lab-x/review/v1-check.json",
                                           "sameWindow": True, "reviewer": "me"}
    assert r["counts"]["queue"]["carriedOver"] == 1
    cmp = T.compare_reviewed(r["cases"], {"items": [earlier_item]}, reviews)
    assert cmp["falseAlarms"] == ["lab-x-p01-x1"] and cmp["items"]["lab-x-p01-x1"]["sameWindow"]
    # the same id in the review page shows the earlier verdict, so the tally counts it already
    assert tally(r["queue"], reviews)["reviewed"] == 1


def test_reviewed_clean_windows_are_scored_but_never_queued():
    s = session(60, seam_at=24, wrong_at=24)
    phone = {"id": "lab-x-p01-phone-3", "kind": "case", "audio": {"path": "lab-x/sessions/lab-x-p01.phone.wav", "startS": 1.0, "endS": 5.0},
             "text": " ".join(w["text"] for w in s["said"][2:10]), "label": {"type": "clean"}}
    extra = T.reviewed_clean_cases({"items": [phone]}, {s["id"]: s}, set())
    assert [c["id"] for c in extra] == [phone["id"]] and extra[0]["reviewedOnly"]
    cases = T.session_cases(s, "lab-x") + extra
    r = T.triage(cases, scores(cases, lambda c: "match", lambda c: "mismatch"), "lab-x", (0, 0), (0.0, 0), 17)
    assert phone["id"] not in {it["id"] for it in r["queue"]["items"]}
    assert r["counts"]["cases"] == len(cases) - 1
    assert next(c for c in r["cases"] if c["id"] == phone["id"])["outcome"] == "disagree"


# ---- seam gate -----------------------------------------------------------------------------------------------------

def seam(outS, **kw):
    return {"outS": outS, "beforeDb": -20.0, "afterDb": -22.0, "srcBeforeDb": -20.0, "srcAfterDb": -22.0, "levelErrDb": 0.0, "seamDb": -18.0,
            "clickRatio": 0.5, **kw}


def gate_session():
    s = session(60, seam_at=24, wrong_at=24)
    checks = {"seams": [seam(12.0), seam(12.5)], "words": {"words": 60, "found": 60, "clipped": 0, "worst": []}}
    return s, checks


def test_seam_gate_passes_clean_renders():
    s, checks = gate_session()
    g = LS.seam_gate(s, checks)
    assert g["failed"] == [] and g["seamsPass"] == 2 and g["wordsFound"] == 60 and g["unmapped"] == []


def test_seam_gate_fails_the_splice_of_a_bad_seam():
    s, checks = gate_session()
    checks["seams"][1] = seam(12.5, clickRatio=3.1, seamDb=-2.0)
    g = LS.seam_gate(s, checks)
    assert g["failed"] == [{"session": "lab-x-p01", "splice": "x1", "id": "lab-x-p01-x1",
                            "reasons": ["click at the seam (ratio 3.10)", "seam -2.0 dB against speech"]}]
    doc, new = LS.refusals_from_checks({"testSet": "lab-x", "failed": g["failed"]}, {s["id"]: {**s, "meta": {**s["meta"], "readId": 9, "surah": 112}}})
    assert [x["kind"] for x in new] == ["sites"] and doc["sites"][0]["class"] == "wrong_word"


def test_seam_gate_level_ignores_digital_silence():
    s, checks = gate_session()
    checks["seams"][0] = seam(12.0, beforeDb=-84.6, srcBeforeDb=-86.0, levelErrDb=1.4)
    assert LS.seam_gate(s, checks)["failed"] == []
    checks["seams"][0] = seam(12.0, afterDb=-20.5, srcAfterDb=-22.0, levelErrDb=1.5)
    assert LS.seam_gate(s, checks)["failed"][0]["reasons"] == ["seam level off the source by 1.50 dB"]


def test_seam_gate_words_unfound_and_cut_short():
    s, checks = gate_session()
    w = s["said"][23]
    checks["words"] = {"words": 60, "found": 59, "clipped": 0, "worst": [{"at": w["at"], "role": "ref", "lagSamples": 0, "corr": 0.9963}]}
    assert LS.seam_gate(s, checks)["failed"] == []  # found, a little off (PCM16 saturation) but over wordCorr
    checks["words"]["worst"] = [{"at": w["at"], "role": "ref", "lagSamples": 40, "corr": 0.7}]
    g = LS.seam_gate(s, checks)
    assert g["failed"][0]["splice"] == "x1" and "not found" in g["failed"][0]["reasons"][0] and g["wordsFound"] == 59
    s, checks = gate_session()
    aligns = {"sources/a.wav": {(t["from"]["surah"], t["from"]["ayah"], t["from"]["word"]): {"startS": 0.0, "endS": 0.45} for t in s["said"]}}
    aligns["sources/a.wav"][(112, 3, 6)] = {"startS": 0.0, "endS": 0.48}  # said word 25, ending at the 12.5 s seam, lost 30 ms
    g = LS.seam_gate(s, checks, aligns)
    assert g["failed"][0]["reasons"] == ["word 112:3:6 cut short by 30 ms"] and g["clipsOver"] == 1


def test_seam_gate_summary_counts():
    s, checks = gate_session()
    checks["seams"][1] = seam(12.5, clickRatio=3.1)
    d = LS.seam_gate_summary([LS.seam_gate(s, checks)], "lab-x")
    assert d["summary"]["seams"] == 2 and d["summary"]["seamsPass"] == 1
    assert d["summary"]["byCheck"] == {"level": 2, "click": 1, "depth": 2, "linkCut": 2, "onset": 2}
    assert d["summary"]["splicesFailed"] == 1 and d["limits"] == LS.SEAM_LIMITS


# ---- dashboard -----------------------------------------------------------------------------------------------------

def test_check_queue_prefers_triage_and_verdicts_skip_model_records():
    assert D.check_queue("lab-x", lambda n: True) == "lab-x/review/triage.json"
    assert D.check_queue("lab-x", lambda n: n == "v1-check.json") == "lab-x/review/v1-check.json"
    me = {"reviewer": "me", "items": {"a": {"verdict": "ok", "at": "2026-10-28T07:00:00"}}}
    asr = {"queue": "lab-x/review/v1-check.json", "items": {"a": {"transcript": "..."}}}  # asr-*.json in the same folder
    calib = {"reviewer": "me", "items": {"a": {"human": "ok"}}}
    for order in ([asr, me, calib], [me, asr, calib], [calib, asr, me]):
        assert L.latest_verdicts(order) == {"a": {"verdict": "ok", "at": "2026-10-28T07:00:00", "reviewer": "me"}}
    q = {"version": 1, "testSet": "lab-x", "maxBad": 1, "items": [{"id": "a", "kind": "case", "audio": {"path": "x.wav", "startS": 0, "endS": 1},
                                                                  "text": "", "label": {"type": "clean"}}]}
    assert D.check_status(q, [asr, me], "lab-x/review/triage.json")["queue"] == "lab-x/review/triage.json"

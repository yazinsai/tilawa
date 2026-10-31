"""uv run --python cpython-3.12-macos-aarch64-none --with pytest python -m pytest -q tools/test_lab_phonemes.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_phonemes_core as P  # noqa: E402

REF = P.Ref({"ayahs": [
    {"s": 26, "a": 127, "w": ["رَبِّ", "ٱلْعَـٰلَمِينَ"], "p": "رَببِلعَاالَمِۦۦن", "o": [0, 5]},
    {"s": 26, "a": 128, "w": ["أَتَبْنُونَ", "بِكُلِّ"], "p": "ءَتَبڇنُۥۥنَبِكُللِ", "o": [0, 12]},
]})


def said(text, ph, a, b, at, role="ref", **kw):
    return {"startS": a, "endS": b, "at": dict(zip(("surah", "ayah", "word"), at)) if at else None, "text": text, "phonemes": ph, "role": role, **kw}


def omission(a=2.0, b=2.3):
    return {"errorClass": "omitted_word", "location": {"surah": 26, "ayah": 128, "word": 1}, "startS": a, "endS": b, "expected": "أَتَبْنُونَ"}


SDOC = {"said": [said("رَبِّ", "رَببِ", 0.5, 1.0, (26, 127, 1)), said("ٱلْعَـٰلَمِينَ", "لعَاالَمِۦۦن", 1.0, 2.0, (26, 127, 2)),
                 said("بِكُلِّ", "بِكُللِ", 2.3, 3.0, (26, 128, 2))],
        "events": [omission()]}


def evidence(spec, fm=40):
    """spec: [(phonemes, from s, to s, conf)]: each run's phonemes spread evenly from `from` to `to`."""
    out = []
    for ph, a, b, conf in spec:
        for i, p in enumerate(ph):
            f = round((a + (b - a) * i / max(len(ph), 1)) * 1000 / fm)
            out.append({"p": p, "start": f, "end": f, "conf": conf})
    return {"frameMs": fm, "phonemes": sorted(out, key=lambda p: p["start"])}


CLEAN = evidence([("رَببِ", 0.52, 0.98, 0.99), ("لعَاالَمِۦۦن", 1.02, 1.98, 0.99), ("بِكُللِ", 2.32, 2.98, 0.99)])


def cells(v, key):
    return [(c["kind"], c.get("role"), c["heard"][key]["heard"], c["heard"][key]["status"]) for c in v["columns"]]


# ---- files and the window ------------------------------------------------------------------------------------------

def test_session_of_reads_the_audio_path():
    assert P.session_of("lab-v1.4/sessions/lab-v1.4-p01.phone.wav") == ("lab-v1.4/sessions", "lab-v1.4-p01", "phone")
    assert P.session_of("lab-v1.4/sessions/x.studio.wav")[2] == "studio"
    assert P.session_of("sources/mp3quran/22/026.wav") is None
    assert P.session_of("../sessions/x.phone.wav") is None
    assert P.evidence_rel("s/sessions", "p01", "teacher", "phone") == "s/sessions/p01.teacher.phone.evidence.json"


def test_ref_gives_words_in_context_and_runs_across_ayahs():
    assert REF.words["26:128:1"] == ("أَتَبْنُونَ", "ءَتَبڇنُۥۥنَ")
    assert REF.words["26:128:2"][1] == "بِكُللِ"
    assert [t for t, _ in REF.run("26:127:2", 2)] == ["ٱلْعَـٰلَمِينَ", "أَتَبْنُونَ"]
    assert REF.run("1:1:9", 2) == []


def test_window_takes_overlapping_words_and_the_omitted_slot_in_order():
    cols = P.window_columns(SDOC, REF, 0.9, 2.5)
    assert [(c["text"], c["role"]) for c in cols] == [("رَبِّ", "ref"), ("ٱلْعَـٰلَمِينَ", "ref"), ("أَتَبْنُونَ", "omitted"), ("بِكُلِّ", "ref")]
    om = cols[2]
    assert om["expected"] == "" and om["planted"] and om["reference"]["phonemes"] == "ءَتَبڇنُۥۥنَ"
    assert [c["text"] for c in P.window_columns(SDOC, REF, 0.0, 0.6)] == ["رَبِّ"]
    assert P.window_columns(SDOC, REF, 3.5, 4.0) == []
    assert P.span_of(cols, 0.9, 2.5) == (0.5, 3.0)


def test_wrong_word_carries_the_original_and_expects_what_was_planted():
    sdoc = {"said": [said("لُوطٌ", "لُۥۥطُن", 0.0, 0.6, (26, 128, 2), role="wrong")], "events": []}
    c = P.window_columns(sdoc, REF, 0, 1)[0]
    assert c["expected"] == "لُۥۥطُن" and c["planted"] and c["reference"] == {"text": "بِكُلِّ", "phonemes": "بِكُللِ"}


def test_heard_in_uses_start_frames():
    ev = {"frameMs": 40, "phonemes": [{"p": "a", "start": 10, "end": 12, "conf": 1}, {"p": "b", "start": 25, "end": 25, "conf": 1}]}
    assert [p["p"] for p in P.heard_in(ev, 0.4, 1.0)] == ["a"]
    assert P.heard_in(ev, 0.4, 1.01)[1] == {"p": "b", "t": 1.0, "conf": 1}


# ---- alignment -----------------------------------------------------------------------------------------------------

def test_clean_evidence_matches_every_word_and_the_omission_stays_empty():
    v = P.view(SDOC, REF, {"student": CLEAN, "teacher": CLEAN}, 0.5, 3.0, "omitted_word")
    assert cells(v, "student") == [("word", "ref", "رَببِ", "match"), ("word", "ref", "لعَاالَمِۦۦن", "match"),
                                   ("word", "omitted", "", "absent"), ("word", "ref", "بِكُللِ", "match")]
    assert [c["focus"] for c in v["columns"]] == [False, False, True, False]
    assert v["summary"] == "Where «أَتَبْنُونَ» should be missing (1.5 s into the clip), neither ear heard anything there (as expected)"
    assert v["elsewhere"] == ""


def test_a_phoneme_just_outside_its_word_still_pairs_with_it():
    ev = evidence([("ر", 0.42, 0.43, 0.99), ("َببِ", 0.6, 0.98, 0.99), ("لعَاالَمِۦۦن", 1.02, 1.98, 0.99), ("بِكُللِ", 2.32, 2.98, 0.99)])
    v = P.view(SDOC, REF, {"student": ev}, 0.5, 3.0)
    assert cells(v, "student")[0] == ("word", "ref", "رَببِ", "match")
    assert not [c for c in v["columns"] if c["kind"] == "gap"]


def test_phonemes_heard_in_the_omitted_slot_are_the_word_heard_anyway():
    ev = evidence([("رَببِ", 0.52, 0.98, 0.99), ("لعَاالَمِۦۦن", 1.02, 1.98, 0.99), ("ءَتَبڇنُۥۥنَ", 2.02, 2.28, 0.9), ("بِكُللِ", 2.32, 2.98, 0.99)])
    v = P.view(SDOC, REF, {"student": ev, "teacher": CLEAN}, 0.5, 3.0, "omitted_word")
    om = v["columns"][2]["heard"]["student"]
    assert om["heard"] == "ءَتَبڇنُۥۥنَ" and om["status"] == "heard" and om["likeReference"] == 1.0
    assert all(o["op"] == "ins" for o in om["ops"])
    assert cells(v, "student")[3] == ("word", "ref", "بِكُللِ", "match")
    assert v["summary"] == "Where «أَتَبْنُونَ» should be missing (1.5 s into the clip), the app's ear heard «أَتَبْنُونَ» after all (unexpected); the reference ear heard nothing there (as expected)"


def test_a_few_sounds_in_the_slot_are_not_the_word():
    ev = evidence([("رَببِ", 0.52, 0.98, 0.99), ("لعَاالَمِۦۦن", 1.02, 1.98, 0.99), ("ءَ", 2.1, 2.15, 0.5), ("بِكُللِ", 2.32, 2.98, 0.99)])
    v = P.view(SDOC, REF, {"student": ev, "teacher": ev}, 0.5, 3.0, "omitted_word")
    assert v["columns"][2]["heard"]["student"]["status"] == "heard"
    assert v["summary"] == "Where «أَتَبْنُونَ» should be missing (1.5 s into the clip), both ears picked up a few sounds there"


def test_substitutions_missing_and_extra_phonemes_are_per_word_ops():
    ev = evidence([("رَببِ", 0.52, 0.98, 0.99), ("لعَالَمِۦۦنن", 1.02, 1.98, 0.99), ("لِكُللِ", 2.32, 2.98, 0.7)])
    v = P.view(SDOC, REF, {"student": ev}, 0.5, 3.0)
    w2, w4 = v["columns"][1]["heard"]["student"], v["columns"][3]["heard"]["student"]
    assert w2["status"] == "differs" and [o["op"] for o in w2["ops"]].count("del") == 1 and [o["op"] for o in w2["ops"]].count("ins") == 1
    assert w4["ops"][0] == {"op": "sub", "e": "ب", "h": "ل", "conf": 0.7, "t": 2.32}
    assert w4["likeExpected"] == round(1 - 1 / 7, 2)


def test_nothing_heard_for_a_word():
    ev = evidence([("رَببِ", 0.52, 0.98, 0.99), ("بِكُللِ", 2.32, 2.98, 0.99)])
    v = P.view(SDOC, REF, {"student": ev}, 0.5, 3.0)
    assert cells(v, "student")[1] == ("word", "ref", "", "nothing")
    assert [o["op"] for o in v["columns"][1]["heard"]["student"]["ops"]] == ["del"] * len("لعَاالَمِۦۦن")


def test_a_substitution_never_pairs_far_from_its_word():
    ops = P.align([("x", 0)], [{"p": "y", "t": 5.0, "conf": 1}], [(0.0, 1.0)])
    assert sorted(o["op"] for o in ops) == ["del", "ins"]
    ops = P.align([("x", 0)], [{"p": "y", "t": 1.05, "conf": 1}], [(0.0, 1.0)])
    assert [o["op"] for o in ops] == ["sub"]


# ---- gaps ----------------------------------------------------------------------------------------------------------

def test_phonemes_between_words_are_a_gap_column():
    sdoc = {"said": [said("رَبِّ", "رَببِ", 0.5, 1.0, (26, 127, 1)), said("بِكُلِّ", "بِكُللِ", 2.0, 2.5, (26, 128, 2))], "events": []}
    ev = evidence([("رَببِ", 0.52, 0.98, 0.99), ("تت", 1.4, 1.6, 0.4), ("بِكُللِ", 2.02, 2.48, 0.99)])
    v = P.view(sdoc, REF, {"student": ev, "teacher": evidence([("رَببِ", 0.52, 0.98, 0.99), ("بِكُللِ", 2.02, 2.48, 0.99)])}, 0.5, 2.5)
    kinds = [c["kind"] for c in v["columns"]]
    assert kinds == ["word", "gap", "word"]
    gap = v["columns"][1]
    assert (gap["startS"], gap["endS"]) == (1.0, 2.0)
    assert gap["heard"]["student"]["heard"] == "تت" and gap["heard"]["student"]["status"] == "extra"
    assert gap["heard"]["teacher"] == {"heard": "", "ops": [], "status": "nothing"}
    assert v["summary"] == "Both ears heard every word as expected"


def test_neighbours_phonemes_past_the_window_edges_are_dropped():
    sdoc = {"said": [said("رَبِّ", "رَببِ", 0.5, 1.0, (26, 127, 1))], "events": []}
    ev = evidence([("ن", 0.4, 0.41, 0.99), ("رَببِ", 0.52, 0.98, 0.99), ("ل", 1.08, 1.09, 0.99)])
    v = P.view(sdoc, REF, {"student": ev}, 0.5, 1.0)
    assert [c["kind"] for c in v["columns"]] == ["word"] and cells(v, "student")[0][3] == "match"


# ---- plain words ---------------------------------------------------------------------------------------------------

def test_focus_is_the_labelled_word_nearest_the_centre():
    sdoc = {"said": [said("a", "a", 0, 1, None, role="added"), said("b", "b", 1, 2, (26, 127, 1)), said("c", "c", 8, 9, None, role="added")], "events": []}
    cols = P.window_columns(sdoc, None, 0, 9)
    assert P.focus_of(cols, "added_word", 6, 9) == [2]
    assert P.focus_of(cols, "added_word", 0, 3) == [0]
    assert P.focus_of(cols, "clean", 0, 9) == []
    assert P.focus_of(cols, "wrong_word", 0, 9) == []
    rep = P.window_columns({"said": [said("a", "a", 0, 1, (26, 127, 1)), said("a", "a", 1, 2, (26, 127, 1), role="repeat"),
                                     said("b", "b", 2, 3, (26, 127, 2), role="repeat")], "events": []}, None, 0, 3)
    assert P.focus_of(rep, "repeat", 0, 3) == [1, 2]


def test_wrong_word_summary_tells_planted_from_original():
    sdoc = {"said": [said("لُوطٌ", "لُۥۥطُن", 1.0, 1.6, (26, 128, 2), role="wrong")], "events": []}
    planted = evidence([("لُۥۥطُن", 1.02, 1.58, 0.99)])
    original = evidence([("بِكُللِ", 1.02, 1.58, 0.99)])
    v = P.view(sdoc, REF, {"student": original, "teacher": planted}, 0.5, 2.0, "wrong_word")
    assert v["summary"] == "At «لُوطٌ» (0.5 s into the clip), the app's ear heard the mushaf's word instead (unexpected); the reference ear heard «لُوطٌ» (as described)"
    v = P.view(sdoc, REF, {"student": planted, "teacher": planted}, 0.5, 2.0, "wrong_word")
    assert v["summary"] == "At «لُوطٌ» (0.5 s into the clip), both ears heard «لُوطٌ» (as described)"


def test_summary_counts_other_words_heard_differently():
    ev = evidence([("رَببِ", 0.52, 0.98, 0.99), ("لعَاالَمِۦۦن", 1.02, 1.98, 0.99), ("لِكُللِ", 2.32, 2.98, 0.7)])
    v = P.view(SDOC, REF, {"student": ev, "teacher": CLEAN}, 0.5, 3.0, "omitted_word")
    assert v["elsewhere"] == "In the rest of the clip, the app's ear heard 1 of 3 words slightly differently (usually a vowel or an ending)"
    v = P.view(SDOC, REF, {"student": ev, "teacher": CLEAN}, 0.5, 3.0, "clean")
    assert v["summary"] == "The app's ear heard 1 of 3 words slightly differently, the reference ear 0 (usually a vowel or an ending)"


def test_missing_evidence_is_said_plainly():
    v = P.view(SDOC, REF, {"student": CLEAN, "teacher": None}, 0.5, 3.0, "omitted_word", files={"teacher": "s/p.teacher.phone.evidence.json"})
    t = next(x for x in v["listeners"] if x["key"] == "teacher")
    assert not t["found"] and t["message"] == "No evidence for the reference ear (s/p.teacher.phone.evidence.json)"
    assert set(v["columns"][0]["heard"]) == {"student"}
    assert v["summary"] == "Where «أَتَبْنُونَ» should be missing (1.5 s into the clip), the app's ear heard nothing there (as expected); no evidence for the reference ear"
    v = P.view(SDOC, REF, {}, 0.5, 3.0)
    assert v["summary"] == "No listener evidence for this session"

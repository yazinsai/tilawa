# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with pytest pytest tools/test_review_asr.py"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_asr_core import (  # noqa: E402
    combine, compare, controls, norm_words, perturb_text, prescreen, revert_text, score, session_id, wilson,
)


def test_uthmani_matches_plain_spelling():
    assert compare("ذَٰلِكَ ٱلْكِتَـٰبُ لَا رَيْبَ", "ذلك الكتاب لا ريب.")["decision"] == "match"
    assert compare("ءَالَآءِ رَبِّكُمَا تُكَذِّبَانِ", "آلاء ربكما تكذبان")["decision"] == "match"
    assert compare("وَٱلصَّلَوٰةِ ٱلْوُسْطَىٰ", "والصلاة الوسطى")["decision"] == "match"
    assert compare("يَسْتَـْٔذِنُكَ ٱلَّذِينَ لَا يُؤْمِنُونَ", "يستأذنك الذين لا يؤمنون")["decision"] == "match"
    assert norm_words("كَفَرُوا۟ فِىٓ") == norm_words("كفروا في")


def test_split_and_joined_words_match():
    assert compare("يَـٰٓأَيُّهَا ٱلَّذِينَ ءَامَنُوا۟", "يا أيها الذين آمنوا")["decision"] == "match"


def test_wrong_missing_and_extra_words_mismatch():
    text = "جَنَّـٰتٍۢ وَحَبَّ ٱلْحَصِيدِ وَٱلنَّخْلَ بَاسِقَـٰتٍۢ"
    assert compare(text, "جنات وحب الوريد والنخل باسقات")["decision"] != "match"
    assert compare(text, "جنات وحب والنخل باسقات")["decision"] == "mismatch"
    r = compare(text, "جنات وحب الحصيد قوم والنخل باسقات")
    assert r["decision"] == "mismatch" and any(o["status"] == "extra" for o in r["ops"])
    assert compare("تَجْرِيَانِ فَبِأَىِّ ءَالَآءِ", "نضاختان فبأي آلاء")["decision"] == "mismatch"


def test_short_words_and_edge_repeats_are_not_absorbed():
    assert compare("وَلَـٰكِنَّ أَكْثَرَهُمْ لَا يَعْلَمُونَ", "ولكن أكثرهم يعلمون")["decision"] == "mismatch"
    assert compare("كَالُوهُمْ وَّزَنُوهُمْ يُخْسِرُونَ", "كالوهم أو وزنوهم يخسرون")["decision"] == "mismatch"
    assert compare("ذَٰلِكَ لَـَٔايَةًۭ وَمَا كَانَ أَكْثَرُهُم", "ذلك لآية وما ذلك لآية وما كان أكثرهم")["decision"] == "mismatch"


def test_window_edges_are_lenient():
    text = "وَلَـٰكِنَّ أَكْثَرَهُمْ لَا يَعْلَمُونَ"
    assert compare(text, "مين ولكن أكثرهم لا يعلمون إن")["decision"] == "match"
    assert compare(text, "ولكن أكثرهم لا")["decision"] == "unsure"
    assert compare(text, "")["decision"] == "unsure"


def test_combine():
    assert combine(["match", "match"]) == "match"
    assert combine(["mismatch", "unsure"]) == "mismatch"
    assert combine(["match", "mismatch"]) == "unsure"
    assert combine(["match", "unsure"]) == "unsure"


def session():
    w = lambda t, a, b, role="ref": {"text": t, "startS": a, "endS": b, "role": role}
    said = [w("قَالُوا۟", 0, 1), w("رَبَّنَا", 1, 2), w("كَلَّآ", 2, 3, "wrong"), w("لَنَا", 3, 4), w("هَـٰذَا", 4, 5, "added"),
            w("فَزِدْهُ", 5, 6), w("عَذَابًۭا", 6.5, 7), w("فِى", 7, 8)]
    splices = [{"id": "x1", "type": "wrong_word", "expected": "مَن", "said": "كَلَّآ", "startS": 2, "endS": 3},
               {"id": "x2", "type": "added_word", "expected": "", "said": "هَـٰذَا", "startS": 4, "endS": 5},
               {"id": "x3", "type": "omitted_word", "expected": "قَدَّمَ", "said": "", "startS": 6.1, "endS": 6.4}]
    return {"id": "t-e01", "said": said, "splices": splices}


def item(sid, a, b, s, kind="splice", label="wrong_word"):
    text = " ".join(w["text"] for w in s["said"] if w["endS"] > a and w["startS"] < b)
    return {"id": f"t-e01-{sid}", "kind": kind, "audio": {"path": "t/sessions/t-e01.studio.wav", "startS": a, "endS": b},
            "text": text, "label": {"type": label}}


def test_revert_undoes_the_planted_edit():
    s = session()
    assert revert_text(item("x1", 0, 5, s), s)["text"] == "قَالُوا۟ رَبَّنَا مَن لَنَا هَـٰذَا"
    assert revert_text(item("x2", 2, 7, s, label="added_word"), s)["text"] == "كَلَّآ لَنَا فَزِدْهُ عَذَابًۭا"
    assert revert_text(item("x3", 4, 8, s, label="omitted_word"), s)["text"] == "هَـٰذَا فَزِدْهُ قَدَّمَ عَذَابًۭا فِى"
    bad = item("x1", 0, 5, s)
    bad["text"] += " زيادة"
    assert revert_text(bad, s) is None


def test_perturb_and_controls_are_seeded_and_differ():
    text = "فَـٰكِهَةٌۭ وَٱلنَّخْلُ ذَاتُ ٱلْأَكْمَامِ وَٱلْحَبُّ"
    pool = ["ٱلْمُجْرِمُونَ", "تُكَذِّبُونَ", "قَـٰصِرَٰتُ", "يُخْسِرُونَ"]
    for op in ("sub", "ins", "del"):
        a, b = perturb_text(text, pool, op, random.Random(1)), perturb_text(text, pool, op, random.Random(1))
        assert a == b and a["text"] != text and compare(text, a["text"])["decision"] != "match"
    s = session()
    q = {"testSet": "t", "items": [item("x1", 0, 5, s), item("x2", 2, 7, s, label="added_word"),
                                   {**item("clean-3", 0, 8, s, kind="case", label="clean")}]}
    c = controls(q, {"t-e01": s})
    assert c["t-e01-x1"]["oneEdit"]["control"] == "revert" and c["t-e01-clean-3"]["oneEdit"]["control"] == "perturb"
    assert all(v["shifted"]["text"] != it["text"] for it, v in zip(q["items"], c.values()))
    assert c == controls(q, {"t-e01": s})
    assert session_id({"audio": {"path": "lab-v1.1/sessions/lab-v1.1-e20.phone.wav"}}) == "lab-v1.1-e20"


def test_score_and_wilson():
    d = {"p1": "match", "p2": "unsure", "n1": "mismatch", "n2": "unsure", "n3": "match"}
    t = {"p1": True, "p2": True, "n1": False, "n2": False, "n3": False}
    s = score(d, t)
    assert s["specificity"]["k"] == 1 and s["sensitivity"]["k"] == 1 and s["flagged"]["k"] == 2
    lo, hi = wilson(40, 40)
    assert 0.91 < lo < 0.92 and hi == 1.0


def test_prescreen_keeps_disagreements_seams_and_spot_check():
    items = [{"id": f"i{k}", "kind": "splice" if k < 3 else "case", "audio": {}, "text": "", "label": {"type": "clean"}} for k in range(30)]
    q = {"version": 1, "testSet": "t", "items": items}
    dec = {f"i{k}": "match" for k in range(30)} | {"i5": "mismatch", "i6": "unsure"}
    p = prescreen(q, dec)
    ids = {it["id"] for it in p["items"]}
    assert {"i0", "i1", "i2", "i5", "i6"} <= ids and len(ids) == 5 + 3  # 25 cleared -> 3 spot checks
    assert p == prescreen(q, dec)
    assert len(prescreen(q, dec, seams="skip")["items"]) == 2 + 3
    assert all("pre-screen:" in it["context"] for it in p["items"])

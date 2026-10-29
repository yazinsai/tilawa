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
    arm_stats, check_summary, check_verdict, cleanup_generation_loops, combine, compare, controls, diacritised_candidates,
    diacritised_spellings, edit_texts, near_best, norm_words, passes_bar, perturb_text, plain_candidates, plain_spellings,
    plain_transcript, prefer, prescreen, revert_text, score, session_id, wilson,
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
    n = prescreen(q, dec, notes={"i6": "no alternative text"})
    assert n["prescreen"]["reasons"] == p["prescreen"]["reasons"]
    assert next(it for it in n["items"] if it["id"] == "i6")["context"].endswith("ASR unsure (no alternative text)")


def test_plain_spellings_of_uthmani_words():
    assert plain_spellings("ٱلصَّلَوٰةِ") == ["الصلاة"]
    assert plain_spellings("ءَامَنُوا۟") == ["آمنوا"]
    assert plain_spellings("ءَالَآءِ") == ["آلاء"]
    assert plain_spellings("يَسْتَـْٔذِنُكَ") == ["يستأذنك"]
    assert plain_spellings("فِى") == ["في"] and plain_spellings("عَلَىٰ") == ["على"] and plain_spellings("شَىْءٍ") == ["شيء"]
    assert plain_spellings("إِنَّ") == ["إن"]
    assert set(plain_spellings("ذَٰلِكَ")) == {"ذالك", "ذلك"}


def test_plain_candidates_take_the_transcript_spelling_of_shared_words():
    text = "يُكَذِّبُ بِهَا ٱلْمُجْرِمُونَ كَٱلْأَعْلَـٰمِ"
    assert plain_candidates(text, "بها المجرمون، كالاعلام") == ["يكذب بها المجرمون كالاعلام"]
    many = plain_candidates("ذَٰلِكَ هَـٰذَا ٱلرَّحْمَـٰنِ إِلَـٰهَ")
    assert len(many) == 8 and len(set(many)) == 8 and "ذلك هذا الرحمن إلاه" in many
    assert plain_transcript("بها، المجرمون. abc كالأعلام") == "بها المجرمون كالأعلام"


def test_likelihood_rules_and_loop_guardrail():
    assert prefer(-10.0, -15.0) == "match" and prefer(-15.0, -10.0) == "mismatch" and prefer(-10.0, -11.0) == "unsure"
    assert near_best(-11.0, -10.0) == "match" and near_best(-13.0, -10.0) == "mismatch"
    assert cleanup_generation_loops("قال " + "ا" * 30) == "قال ا"
    assert cleanup_generation_loops("x " + "ab " * 25) == "x ab "


def test_diacritised_spellings_keep_harakat():
    assert diacritised_spellings("ٱلصَّلَوٰةِ") == ["الصَّلَاةِ"]
    assert diacritised_spellings("ءَامَنُوا۟") == ["آمَنُوا"]
    assert diacritised_spellings("ءَالَآءِ") == ["آلَاءِ"]
    assert diacritised_spellings("فِىٓ") == ["فِي"] and diacritised_spellings("عَلَىٰ") == ["عَلَى"]
    assert diacritised_spellings("شَىْءٍ") == ["شَيْءٍ"]
    assert diacritised_spellings("يَسْتَـْٔذِنُكَ") == ["يَسْتَأْذِنُكَ"]
    assert diacritised_spellings("لَـَٔايَةًۭ") == ["لَآيَةً"]
    assert diacritised_spellings("مِهَـٰدًۭا") == ["مِهَادًا", "مِهَدًا"]
    assert diacritised_spellings("يَوْمَئِذٍۢ") == ["يَوْمَئِذٍ"] and diacritised_spellings("لَهُۥ") == ["لَهُ"]
    assert diacritised_spellings("إِنَّ") == ["إِنَّ"] and diacritised_spellings("أَحْيَآءًۭ") == ["أَحْيَاءً"]
    assert diacritised_spellings("يُؤْمِنُونَ") == ["يُؤْمِنُونَ"]
    assert diacritised_spellings("ٱلصَّلَوٰةِ", charset=set("الصلاةِ")) == ["الصلاةِ"]
    for w in ("ٱلصَّلَوٰةِ", "مِهَـٰدًۭا", "يَسْتَـْٔذِنُكَ", "ءَالَآءِ"):
        assert {norm_words(s)[0] for s in diacritised_spellings(w)} <= {norm_words(s)[0] for s in plain_spellings(w)}


def test_diacritised_candidates_take_the_transcript_spelling():
    text = "يُكَذِّبُ بِهَا ٱلْمُجْرِمُونَ"
    assert diacritised_candidates(text, "بِهَا الْمُجْرِمُونَ.") == ["يُكَذِّبُ بِهَا الْمُجْرِمُونَ"]
    assert diacritised_candidates("ذَٰلِكَ ٱلْكِتَـٰبُ") == ["ذَالِكَ الْكِتَابُ", "ذَالِكَ الْكِتَبُ", "ذَلِكَ الْكِتَابُ", "ذَلِكَ الْكِتَبُ"]


def test_edit_texts_mirror_the_teacher_check_window():
    s = {**session(), "durationS": 10.0}
    s["said"][2]["op"], s["said"][4]["op"] = "x1", "x2"
    s["splices"][0]["expected"] = "مَن"
    w = edit_texts(s, s["splices"][0])
    assert w["said"] == "رَبَّنَا كَلَّآ لَنَا" and w["original"] == "رَبَّنَا مَن لَنَا" and (w["startS"], w["endS"]) == (0.9, 4.1)
    w = edit_texts(s, s["splices"][1])
    assert w["said"] == "لَنَا هَـٰذَا فَزِدْهُ" and w["original"] == "لَنَا فَزِدْهُ" and w["class"] == "added_word"
    w = edit_texts(s, s["splices"][2])
    assert w["said"] == "فَزِدْهُ عَذَابًۭا" and w["original"] == "فَزِدْهُ قَدَّمَ عَذَابًۭا"
    assert edit_texts(s, {"id": "x9", "type": "omitted_word", "expected": "مَن", "startS": -1.0, "endS": -0.5}) is None
    assert edit_texts(s, {**s["splices"][0], "mutashabihat": True})["class"] == "mutashabihat"
    w = edit_texts(s, s["splices"][0], context_s=1.0)
    assert w["said"] == "قَالُوا۟ رَبَّنَا كَلَّآ لَنَا هَـٰذَا" and w["original"] == "قَالُوا۟ رَبَّنَا مَن لَنَا هَـٰذَا" and w["startS"] == 0.0
    w = edit_texts(s, s["splices"][2], context_s=1.0)
    assert w["said"] == "هَـٰذَا فَزِدْهُ عَذَابًۭا فِى" and w["original"] == "هَـٰذَا فَزِدْهُ قَدَّمَ عَذَابًۭا فِى"


def test_check_verdict_and_summary():
    assert check_verdict(-10.0, -14.0) == {"pass": True, "llr": 4.0, "reason": None}
    assert check_verdict(-10.0, -11.0)["reason"] == "can't separate said from original"
    assert check_verdict(-14.0, -10.0)["reason"] == "prefers the original"
    assert check_verdict(-10.0, None)["pass"] and not check_verdict(None, -10.0)["pass"]
    s = check_summary([{"class": "wrong_word", "pass": True}, {"class": "wrong_word", "pass": False}, {"class": "repeat", "pass": True}])
    assert s["classes"]["wrong_word"] == {"n": 2, "pass": 1, "fail": 1} and s["mistakes"]["n"] == 2 and s["nonErrors"] == {"n": 1, "pass": 1, "fail": 0}


def test_arm_stats_and_bar():
    table = {f"i{k}": {"human": "ok", "label": "clean" if k < 10 else "wrong_word", "oneEditEdit": "repeat" if k == 39 else "sub",
                       "oneEditWords": 1} for k in range(40)}
    good = lambda i, k: "match" if k == "labelled" else "mismatch"
    s = arm_stats(table, good)
    assert s["agreement"]["k"] == 40 and s["oneEdit"]["flagged"]["k"] == 40 and s["oneWord"]["flagged"]["n"] == 39 and passes_bar(s)
    leaky = lambda i, k: "match" if k == "labelled" or (k == "oneEdit" and i == "i39") else "mismatch"
    assert arm_stats(table, leaky)["repeatMisses"] == ["i39"] and not passes_bar(arm_stats(table, leaky))
    assert arm_stats(table, good)["agreementEdited"]["n"] == 30


def test_context_words_never_count():
    text = "أَكْثَرَهُمْ لَا يَعْلَمُونَ"
    assert compare(text, "ولكن أكثرهم لا يعلمون قل", before="وَلَـٰكِنَّ", after="قُلْ")["decision"] == "match"
    assert compare(text, "أكثرهم لا يعلمون", before="وَلَـٰكِنَّ", after="قُلْ")["decision"] == "match"
    assert compare(text, "ولكن أكثرهم يعلمون قل", before="وَلَـٰكِنَّ", after="قُلْ")["decision"] == "mismatch"
    assert compare(text, "ولكن لا يعلمون قل", before="وَلَـٰكِنَّ", after="قُلْ")["decision"] == "mismatch"
    assert compare(text, "ولكن أكثرهم لا يعلمون", before="وَلَـٰكِنَّ")["decision"] == compare(text, "أكثرهم لا يعلمون")["decision"]

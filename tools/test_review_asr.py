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
    calib_stats, cleanup_generation_loops, combine, compare, compare_rasm, controls, edge_context, judge_rasm, near_best, norm_words,
    passes_bar, perturb_text, pick_prompt, plain_candidates, plain_spellings, plain_transcript, prefer, prescreen, rasm, revert_text,
    rover, score, session_id, wilson,
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


def test_context_words_never_count():
    text = "أَكْثَرَهُمْ لَا يَعْلَمُونَ"
    assert compare(text, "ولكن أكثرهم لا يعلمون قل", before="وَلَـٰكِنَّ", after="قُلْ")["decision"] == "match"
    assert compare(text, "أكثرهم لا يعلمون", before="وَلَـٰكِنَّ", after="قُلْ")["decision"] == "match"
    assert compare(text, "ولكن أكثرهم يعلمون قل", before="وَلَـٰكِنَّ", after="قُلْ")["decision"] == "mismatch"
    assert compare(text, "ولكن لا يعلمون قل", before="وَلَـٰكِنَّ", after="قُلْ")["decision"] == "mismatch"
    assert compare(text, "ولكن أكثرهم لا يعلمون", before="وَلَـٰكِنَّ")["decision"] == compare(text, "أكثرهم لا يعلمون")["decision"]


MADD = "\u0644\u064e\u0627\u0653"  # لَآ as the labels store it: alif + madd sign, not a precomposed آ


def test_rasm_skeleton_unifies_spellings_and_keeps_hamza():
    for uthmani, plain in [("ٱلصَّلَوٰةِ", "الصلاة"), ("يَسْتَـْٔذِنُكَ", "يستأذنك"), ("أُو۟لَـٰٓئِكَ", "أولئك"), ("ٱلْمَلَـٰٓئِكَةُ", "الملائكة"),
                           ("مَسْـُٔولُونَ", "مسؤولون"), ("عَلَىٰ", "على"), ("ٱلْكِتَـٰبُ", "الكتاب"), ("لَـَٔايَةًۭ", "لآية")]:
        assert rasm(uthmani) == rasm(plain), uthmani
    assert rasm(MADD) == rasm("لا") == "ل" and rasm("آلاء") == "ءلء"
    assert rasm("إِلَّا") != rasm("لا") and rasm("إِلَّا", hamza=False) == rasm("لا")
    assert rasm("فَثَبِّتُوا۟", phonetic=True) == rasm("فذبطوا", phonetic=True) != rasm("فذبطوا")


def test_compare_rasm_ignores_spacing_but_not_dropped_words():
    assert compare("كَٱلْأَعْلَـٰمِ بَيْنَهَا", "كلا علام بينها")["decision"] != "match"
    assert compare_rasm("يُكَذِّبُ كَٱلْأَعْلَـٰمِ بَيْنَهَا", "يكذب كلا علام بينها", hamza=False)["decision"] == "match"
    assert compare_rasm("يَـٰٓأَيُّهَا ٱلَّذِينَ ءَامَنُوا۟", "يا أيها الذين آمنوا")["decision"] == "match"
    assert compare_rasm("يَبْنَؤُمَّ لَا تَأْخُذْ", "يا ابن أم لا تأخذ")["decision"] == "match"
    for hamza in (True, False):
        assert compare_rasm("وَلَـٰكِنَّ أَكْثَرَهُمْ لَا يَعْلَمُونَ", "ولكن أكثرهم يعلمون", hamza=hamza)["decision"] == "mismatch"
        assert compare_rasm("كَالُوهُمْ أَو وَّزَنُوهُمْ يُخْسِرُونَ", "كالوهم أوزنوهم يخسرون", hamza=hamza)["decision"] == "mismatch"
        assert compare_rasm("وَٱسْتَكْبَرَ هُوَ وَجُنُودُهُۥ فِى", "واستكبروا وجنوده في", hamza=hamza)["decision"] == "mismatch"
        assert compare_rasm("جَنَّـٰتٍۢ وَحَبَّ ٱلْحَصِيدِ وَٱلنَّخْلَ", "جنات وحب الحصيد قوم والنخل", hamza=hamza)["decision"] == "mismatch"


def test_rasm_context_never_counts_but_spill_beyond_it_does():
    text = "إِنَّمَا تُنذِرُ مَنِ"
    assert compare_rasm(text, "إنما تنذر من اتبع", after="ٱتَّبَعَ")["decision"] == "match"
    assert compare_rasm(text, "إنما تنذر من اتبع الذكر", after="ٱتَّبَعَ")["decision"] == "match"  # one stray word past the context
    assert compare_rasm(text, "إنما تنذر من اتبع الحج والهدى", after="ٱتَّبَعَ")["decision"] == "mismatch"
    assert compare(text, "إنما تنذر من اتبع الحج والهدى", after="ٱتَّبَعَ")["decision"] == "match"  # the padded rule ignores it
    assert compare_rasm(text, "تنذر من اتبع", after="ٱتَّبَعَ")["decision"] == "unsure"


def test_edge_context_uses_session_word_times():
    w = lambda t, a, b: {"text": t, "startS": a, "endS": b, "role": "ref"}
    s = {"said": [w("قُلْ", 0.0, 0.9), w("هُوَ", 1.0, 1.4), w("ٱللَّهُ", 1.5, 2.2), w("أَحَدٌ", 2.3, 3.0), w("ٱللَّهُ", 3.5, 4.0)]}
    it = {"audio": {"startS": 0.95, "endS": 3.05}}
    assert edge_context(it, s) == {"before": "قُلْ", "after": "", "lead": [], "trail": []}
    cut = edge_context({"audio": {"startS": 1.2, "endS": 3.45}}, s)
    assert cut == {"before": "", "after": "ٱللَّهُ", "lead": ["هُوَ"], "trail": []}
    assert judge_rasm("هُوَ ٱللَّهُ أَحَدٌ", "الله أحد الله", cut)["decision"] == "match"
    assert judge_rasm("هُوَ ٱللَّهُ أَحَدٌ", "الله الله", cut)["decision"] == "mismatch"
    assert edge_context(it, None)["before"] == ""


def test_rover_votes_per_word():
    r = rover(["ذلك الكتاب لا ريب فيه", "ذالك الكتاب لا ريب فيه", "ذلك الكتاب ريب فيه هدى", "اللهم اغفر لي ولوالدي"])
    assert r["transcript"] == "ذلك الكتاب لا ريب فيه" and r["pivot"] == 0
    assert rover(["كلا علام بينها", "كالأعلام بينها", "كالاعلام بينها"])["transcript"] in ("كالاعلام بينها", "كالأعلام بينها", "كلا علام بينها")
    assert rover(["إنه لحق مثل قوم ما", "إنه لحق مثل قوم ما", "إنه لحق مثل ما"])["transcript"] == "إنه لحق مثل قوم ما"
    assert rover(["إنه لحق مثل ما", "إنه لحق مثل ما", "إنه لحق مثل قوم ما"])["transcript"] == "إنه لحق مثل ما"
    assert rover(["", ""])["transcript"] == ""


def test_calib_stats():
    row = lambda lab, one, sh, edit="sub", words=1: {"labelled": lab, "oneEdit": one, "shifted": sh, "oneEditEdit": edit, "oneEditWords": words}
    t = {"a": row("match", "mismatch", "mismatch"), "b": row("unsure", "match", "mismatch", "repeat", 2), "c": row("match", "unsure", "match")}
    s = calib_stats(t, ["a", "b", "c"])
    assert s["agreement"]["k"] == 2 and s["oneEdit"]["k"] == 2 and s["oneWord"]["n"] == 2 and s["shifted"]["k"] == 2
    assert s["repeatMisses"] == ["b"] and s["disagree"] == ["b"]
    assert not passes_bar(s)


def test_bar_and_prompt_pick():
    st = lambda agree, one, word, rep=(): {"agreement": {"k": agree, "n": 40}, "oneEdit": {"k": one, "n": 40, "rate": one / 40},
                                          "oneWord": {"k": word, "n": 31, "rate": word / 31}, "repeatMisses": list(rep)}
    assert passes_bar(st(38, 36, 28)) and not passes_bar(st(37, 40, 31)) and not passes_bar(st(40, 40, 31, ["x"]))
    assert not passes_bar(st(39, 35, 31))
    assert pick_prompt({"verbatim": st(23, 38, 29), "recitation": st(26, 40, 31), "mistakes": st(30, 30, 25)}) == "recitation"
    assert pick_prompt({"verbatim": st(26, 40, 31), "recitation": st(26, 40, 31)}) == "verbatim"
    assert pick_prompt({"a": st(20, 10, 10), "b": st(25, 10, 10)}) == "b"

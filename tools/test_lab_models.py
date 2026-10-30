"""uv run --python cpython-3.12-macos-aarch64-none --with pytest python -m pytest -q tools/test_lab_models.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_models_core as M  # noqa: E402
import lab_triage_core as T  # noqa: E402
import review_api_core as A  # noqa: E402
from lab_review_core import validate_queue  # noqa: E402
from test_lab_triage import session  # noqa: E402

LABEL = "قُلْ هُوَ ٱللَّهُ أَحَدٌ"
ALT = "قُلْ هُوَ ٱلرَّحْمَٰنُ أَحَدٌ"
AUDIO = {"path": "lab-x/sessions/p01.studio.wav", "startS": 1.0, "endS": 4.0}


def ops(d):
    return [(s["op"], s.get("ref"), s.get("heard")) for s in d["segments"]]


# ---- word diff -----------------------------------------------------------------------------------------------------

def test_diff_ignores_vowels_and_alif_spellings():
    d = M.word_diff(LABEL, "قل هو الله أحد.")
    assert [o for o, *_ in ops(d)] == ["same"] * 4
    assert d["counts"] == {"same": 4, "sub": 0, "del": 0, "ins": 0, "edge": 0}
    assert d["summary"] == "Every labelled word heard, nothing extra"
    assert d["segments"][2] == {"op": "same", "ref": "ٱللَّهُ", "heard": "الله"}  # the labelled word as written, the heard one as returned


def test_diff_marks_changed_missing_and_extra_words():
    d = M.word_diff(LABEL, "قل هو الرحمن أحد")
    assert ops(d)[2] == ("sub", "ٱللَّهُ", "الرحمن")
    d = M.word_diff(LABEL, "قل الله أحد")
    assert ("del", "هُوَ", None) in ops(d) and d["counts"]["del"] == 1
    d = M.word_diff(LABEL, "قل هو الله كتاب أحد")
    assert ops(d)[3] == ("ins", None, "كتاب") and d["counts"]["ins"] == 1
    assert d["summary"] == "1 word differs: 1 extra"


def test_diff_merges_words_heard_split_or_joined():
    d = M.word_diff("يَـٰٓأَيُّهَا ٱلنَّاسُ", "يا أيها الناس")
    assert ops(d) == [("same", "يَـٰٓأَيُّهَا", "يا أيها"), ("same", "ٱلنَّاسُ", "الناس")]
    d = M.word_diff("هَـٰذَا مِن كَلَّآ قَبْلُ", "هذا منك لا قبل")  # the same letters split elsewhere, as the Soniox judge reads them
    assert ops(d)[1] == ("same", "مِن كَلَّآ", "منك لا") and d["summary"] == "Every labelled word heard, nothing extra"


def test_diff_edges_are_marked_and_not_counted():
    d = M.word_diff(LABEL, "بسم قل هو الله أحد")  # a neighbouring sound the window caught
    assert d["segments"][0] == {"op": "ins", "heard": "بسم", "edge": True}
    assert d["counts"]["ins"] == 0 and d["counts"]["edge"] == 1
    assert d["summary"] == "Matches the label apart from 1 word at the window's edges"
    d = M.word_diff(LABEL, "قل هو الله")
    assert d["segments"][-1] == {"op": "del", "ref": "أَحَدٌ", "edge": True}
    d = M.word_diff(LABEL, "كتاب")  # nothing heard as labelled: no edges to speak of
    assert d["counts"]["edge"] == 0 and d["summary"].startswith("4 words differ")


def test_diff_without_a_transcript():
    assert M.word_diff(LABEL, None) == {"segments": [], "counts": {"same": 0, "sub": 0, "del": 0, "ins": 0, "edge": 0}, "summary": "No transcript to compare"}
    assert M.word_diff(LABEL, "")["summary"] == "No transcript to compare"


def test_diff_summary_counts_every_kind():
    c = {"same": 3, "sub": 1, "del": 2, "ins": 1, "edge": 2}
    assert M.diff_summary(c, True) == "4 words differ: 1 changed, 2 not heard, 1 extra; also 2 words at the window's edges"


# ---- plain words ---------------------------------------------------------------------------------------------------

def test_likelihood_score_lines():
    assert M.score_line({"decision": "match", "diff": 12.34})[0] == "label fits better by 12.3"
    assert M.score_line({"decision": "mismatch", "diff": -4.06})[0] == "alternative fits better by 4.1"
    assert M.score_line({"decision": "unsure", "diff": -1.0})[0] == "too close to call: alternative ahead by 1.0, needs 2"
    assert M.score_line({"decision": "match", "diff": None})[0] == "Only the label can fit this audio"
    assert M.score_line({"decision": "unsure", "diff": None})[0] == "Couldn't score either text"
    assert "nats" in M.score_line({"decision": "match", "diff": 3.0})[1]


def test_letter_score_lines_from_the_judge():
    j = A.judge("قل هو الله أحد", LABEL, ALT, "reach")
    s, how = M.score_line(j)
    assert j["decision"] == "match" and s == f"0 letters off the label, {j['alternative']} off the alternative · label closer by {j['diff']}, needs {j['need']:g}"
    assert "rasm" in how
    j = A.judge("قل هو الرحمن أحد", LABEL, ALT, "reach")
    assert M.score_line(j)[0].endswith(f"alternative closer by {-j['diff']}, needs {j['needAgainst']:g}")
    r = {"decision": "unsure", "reason": "too close to call", "claimed": 3, "alternative": 4, "diff": 1, "need": 2, "needAgainst": 2, "unit": "letters"}
    assert M.score_line(r)[0] == "3 letters off the label, 4 off the alternative · too close to call: label closer by 1, needs 2"
    assert M.score_line({**r, "reason": "transcript far from both texts"})[0].startswith("What it heard fits neither text")
    assert M.score_line(A.judge("", LABEL, ALT))[0] == "It heard nothing"
    assert M.score_line({"decision": "unsure", "reason": "no transcript", "unit": "letters", "claimed": None, "diff": None})[0] == "No usable transcript"


def test_why_line_and_names():
    assert M.why_line("disagree", ["disagree", "seam"]) == "In the queue because the two models disagree · also a seam spot check"
    assert M.why_line("auto-accepted", ["spot"]) == "In the queue because a random spot check of labels both models accepted"
    v = M.model_view("soniox", {"decision": "mismatch", "diff": -3, "claimed": 5, "alternative": 2, "need": 2, "needAgainst": 2, "unit": "letters",
                                "transcript": "قل هو الرحمن أحد"}, LABEL)
    assert v["name"] == "Soniox (transcribes the words)" and v["decision"] == {"key": "mismatch", "text": "doesn’t fit"}
    assert M.model_view("nemotron", {"decision": "unsure", "diff": 0.5}, LABEL)["name"] == "Nemotron (checks how well the label fits the sound)"


# ---- lookup --------------------------------------------------------------------------------------------------------

def item(triage=None, **kw):
    it = {"id": "lab-x-p01-x1", "kind": "splice", "audio": AUDIO, "text": LABEL, "label": {"type": "wrong_word", "expected": "ٱللَّهُ", "said": "ٱللَّهُ"}}
    return {**it, **({"triage": triage} if triage else {}), **kw}


def case(**kw):
    return {"id": "lab-x-p01-x1", "group": "edit", "claimed": LABEL, "alternative": ALT, "alternativeOp": "wrong word put back", "audio": AUDIO,
            "outcome": "disagree", "queued": ["disagree"], **kw}


def loader(files):
    return lambda rel: files.get(rel)


SONIOX = {"decision": "match", "reason": None, "claimed": 0, "alternative": 5, "diff": 5, "need": 2.5, "needAgainst": 2, "unit": "letters"}
FILES = {
    "triage-soniox-v1.json": {"cases": [case(nemotron={"decision": "mismatch", "claimed": -30.0, "alternative": -20.0, "diff": -10.0}, soniox=SONIOX,
                                             transcript="قل هو الله أحد")]},
    "triage-v1.json": {"cases": [case(nemotron={"decision": "mismatch", "claimed": -30.0, "alternative": -20.0, "diff": -10.0},
                                      fastconformer={"decision": "unsure", "claimed": -5.0, "alternative": -6.0, "diff": 1.0}, outcome="unsure", queued=["unsure"])]},
    "triage/asr-nemotron-3.5-transcribe.json": {"items": {"lab-x-p01-x1": {"transcript": "قل هو الرحمن أحد"}}},
    "triage/triage-v1.fastconformer.json": {"items": {"lab-x-p01-x1": {"transcript": "قل هو الله"}}},
}


def test_soniox_pairing_reads_its_case_file_and_nemotron_transcripts():
    it = item({"reasons": ["disagree"], "outcome": "disagree", "nemotron": {"decision": "mismatch", "diff": -10.0}, "soniox": {"decision": "match", "diff": 5}})
    v = M.view(it, loader(FILES))
    assert v["found"] and v["source"] == "triage-soniox-v1.json" and v["why"] == "In the queue because the two models disagree"
    assert v["alternative"] == {"text": ALT, "name": "the text without the planted mistake", "op": "wrong word put back"}
    nem, son = v["models"]
    assert (nem["model"], nem["decision"]["text"], nem["score"], nem["transcript"]) == ("nemotron", "doesn’t fit", "alternative fits better by 10.0", "قل هو الرحمن أحد")
    assert ops(nem["diff"])[2] == ("sub", "ٱللَّهُ", "الرحمن") and nem["heardNote"]
    assert (son["model"], son["decision"]["text"], son["transcript"], son["inRule"]) == ("soniox", "fits", "قل هو الله أحد", True)
    assert son["score"] == "0 letters off the label, 5 off the alternative · label closer by 5, needs 2.5"


def test_fastconformer_pairing_adds_soniox_as_not_in_the_rule():
    it = item({"reasons": ["unsure"], "outcome": "unsure", "nemotron": {"decision": "mismatch", "diff": -10.0}, "fastconformer": {"decision": "unsure", "diff": 1.0}})
    v = M.view(it, loader(FILES))
    assert v["source"] == "triage-v1.json" and [(m["model"], m["inRule"]) for m in v["models"]] == [("nemotron", True), ("fastconformer", True), ("soniox", False)]
    assert v["models"][1]["transcript"] == "قل هو الله" and v["models"][1]["score"] == "too close to call: label ahead by 1.0, needs 2"


def test_items_without_triage_are_looked_up_by_id_and_window():
    v = M.view(item(), loader(FILES))  # e.g. a v1-check item whose id and window are a triage case's
    assert v["found"] and [m["model"] for m in v["models"]] == ["nemotron", "soniox"] and v["outcome"]["text"] == "The two models disagree"
    assert v["reasons"] == [] and v["why"] == "Triage of the same window: the two models disagree"
    moved = item(audio={**AUDIO, "endS": 5.0})
    assert M.view(moved, loader(FILES)) == {"id": "lab-x-p01-x1", "found": False, "message": "No model output for this item", "models": []}
    assert M.view(item(), loader({}))["message"] == "No model output for this item"


def test_embedded_records_need_no_files():
    tri = {"reasons": ["unsure"], "outcome": "unsure", "alternative": ALT, "alternativeOp": "op",
           "nemotron": {"decision": "match", "diff": 3.0, "transcript": "قل هو الله أحد"},
           "soniox": {"decision": "unsure", "diff": 1, "claimed": 2, "alternative": 3, "need": 2, "needAgainst": 2, "unit": "letters", "reason": "too close to call",
                      "transcript": "قل هو الله احد"}}
    v = M.view(item(tri), loader({}))
    assert v["source"] == "queue" and [m["transcript"] for m in v["models"]] == ["قل هو الله أحد", "قل هو الله احد"]
    assert v["models"][1]["score"].startswith("2 letters off the label, 3 off the alternative · too close to call")


def test_by_id_is_built_once_per_document():
    doc = {"cases": [{"id": "a"}]}
    assert M.by_id(doc) is M.by_id(doc) and M.by_id(None) == {}


# ---- the triage build embeds a compact summary --------------------------------------------------------------------

def test_queue_items_embed_each_models_transcript_and_reading():
    s = session(60, seam_at=24, wrong_at=24)
    cases = T.session_cases(s, "lab-x")
    nem = {c["id"]: {"claimed": -10.0, "alternative": 0.0, "transcript": "نص"} for c in cases}
    son = {c["id"]: {**A.judge(c["claimed"], c["claimed"], c["alternative"]), "transcript": c["claimed"]} for c in cases}
    r = T.triage(cases, {"nemotron": nem, "soniox": son}, "lab-x", (0, 0), (0.0, 0), 17, models=("nemotron", "soniox"))
    q = r["queue"]
    assert validate_queue(q) == [] and q["items"]
    tri = q["items"][0]["triage"]
    assert tri["nemotron"] == {"decision": "mismatch", "diff": -10.0, "transcript": "نص"}
    assert set(tri["soniox"]) == {"decision", "diff", "claimed", "alternative", "need", "needAgainst", "unit", "transcript"}
    v = M.view(q["items"][0], loader({}))
    assert v["source"] == "queue" and v["models"][1]["diff"]["summary"] == "Every labelled word heard, nothing extra"


def test_queue_items_stay_as_before_without_transcripts():
    assert T.model_summary({"decision": "match", "claimed": -1.0, "alternative": -9.0, "diff": 8.0}) == {"decision": "match", "diff": 8.0}
    assert T.model_summary({"decision": "unsure", "claimed": None, "alternative": None, "diff": None}) == {"decision": "unsure", "diff": None}

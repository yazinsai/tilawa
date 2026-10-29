"""uv run --python 3.12 --with pytest --with numpy --with rapidfuzz python -m pytest -q tools/test_review_api.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_triage_core as T  # noqa: E402
import review_api_core as A  # noqa: E402
from lab_review_core import validate_queue  # noqa: E402
from test_lab_triage import WORDS, session  # noqa: E402

LABEL = "إِنَّهُۥ لَحَقٌّۭ مِّثْلَ قَوْمٌۭ مَا\u0653 أَنَّكُمْ"  # p15-x8 as labelled: قَوْمٌ added
ORIGINAL = "إِنَّهُۥ لَحَقٌّۭ مِّثْلَ مَا\u0653 أَنَّكُمْ"


def test_transcripts_from_vendor_responses():
    assert A.soniox_transcript({"text": " إنه لحق ", "tokens": [{"text": "x"}]}) == "إنه لحق"
    assert A.soniox_transcript({"tokens": [{"text": "إن"}, {"text": "ه"}, {"text": " لحق"}]}) == "إنه لحق"
    assert A.soniox_transcript({}) == A.speechmatics_transcript(None) == ""
    sm = {"metadata": {"language_pack_info": {"word_delimiter": " "}},
          "results": [{"type": "word", "alternatives": [{"content": "إنه"}]}, {"type": "word", "alternatives": [{"content": "لحق"}]},
                      {"type": "punctuation", "attaches_to": "previous", "alternatives": [{"content": "."}]},
                      {"type": "word", "alternatives": [{"content": "مثل"}]}, {"type": "word", "alternatives": []}]}
    assert A.speechmatics_transcript(sm) == "إنه لحق. مثل"


def test_judge_picks_the_side_of_the_midpoint_with_a_floor():
    heard = "إنه لحق مثل قوم ما أنكم"
    j = A.judge(heard, LABEL, ORIGINAL)
    assert j["decision"] == "match" and j["claimed"] == 0 and j["diff"] == j["between"] == 3 and j["need"] == 2
    # the canonical verse (planted word dropped) reads as the original: a mismatch, never a match
    assert A.judge("إنه لحق مثل ما أنكم", LABEL, ORIGINAL)["decision"] == "mismatch"
    assert A.judge("إنه لحق مثل ما أنكم", ORIGINAL, LABEL)["decision"] == "match"
    # a word heard split or joined costs nothing on the space-free skeleton; edge drops hit both texts alike
    assert A.judge("إنهلحق مثل قوم ماأنكم", LABEL, ORIGINAL)["decision"] == "match"
    assert A.judge("لحق مثل قوم ما", LABEL, ORIGINAL)["decision"] == "match"
    # half-heard planted word: too close to call
    assert A.judge("إنه لحق مثل ق ما أنكم", LABEL, ORIGINAL)["decision"] == "unsure"


def test_judge_unsure_without_a_usable_transcript_or_alternative():
    assert A.judge("", LABEL, ORIGINAL) == {"decision": "unsure", "reason": "empty transcript", "claimed": None, "alternative": None, "diff": None}
    assert A.judge("إنه", LABEL, None)["reason"] == "no alternative text"
    far = A.judge("اللهم اغفر لي ولوالدي وللمؤمنين والمؤمنات", LABEL, ORIGINAL)
    assert far["decision"] == "unsure" and far["reason"] == "transcript far from both texts"
    # a one-letter difference between the texts can never clear the 2-letter floor
    assert A.judge("ما", "مَا", "مَن")["decision"] == "unsure"


def test_readings_keep_the_ticket_12_rules_beside_the_judge():
    r = A.readings("إنه لحق مثل قوم ما أنكم", LABEL, ORIGINAL)
    assert {k: r[k]["decision"] for k in A.READINGS} == {"judge": "match", "opinion": "match", "opinionRasm": "match"}
    assert all(v["decision"] == "unsure" for v in A.readings(None, LABEL, ORIGINAL).values())


def test_edit_zone_and_bucket():
    s = session(60, seam_at=24, wrong_at=24)
    x = s["splices"][0]
    ws = s["said"][19:30]
    assert A.edit_zone(ws, x) == {4, 5, 6}
    plain = session(60)["said"][19:30]
    assert A.edit_zone(plain, {"startS": 12.0, "endS": 12.0}) == {4, 5} and A.edit_zone(ws, None) == set()
    text = " ".join(w["text"] for w in ws)
    assert text.split()[5] == "مَلِكِ"
    heard = " ".join(w if k != 5 else "ملكوت" for k, w in enumerate(text.split()))
    op = A.readings(heard, text, None)["opinion"]
    b = A.bucket(text, heard, op, zone={4, 5, 6})
    assert b["bucket"] == 3 and b["name"] == "mishearing" and b["atEdit"]
    assert A.bucket(text, heard, op, zone={0})["atEdit"] is False
    assert A.bucket(text, "اشتركوا في القناة", A.readings("اشتركوا في القناة", text, None)["opinion"])["bucket"] == 5
    canon = "إنه لحق مثل ما أنكم"
    assert A.bucket(LABEL, canon, A.readings(canon, LABEL, ORIGINAL)["opinion"], revert=ORIGINAL)["bucket"] == 4
    split = "إنهلحق مثل قوم ما أنكم"
    assert A.bucket(LABEL, split, {"ops": [], "reasons": ["wrong: x"]})["bucket"] == 1


def rows():
    """Six of your verdicts: four ok, two bad; Nemotron's decision and the API's judge per item."""
    return [{"id": "ok1", "truth": True, "nemotron": "match", "api": "match"},
            {"id": "ok2", "truth": True, "nemotron": "match", "api": "unsure"},
            {"id": "ok3", "truth": True, "nemotron": "mismatch", "api": "mismatch"},
            {"id": "ok4", "truth": True, "nemotron": "unsure", "api": "match"},
            {"id": "bad1", "truth": False, "nemotron": "match", "api": "match"},
            {"id": "bad2", "truth": False, "nemotron": "match", "api": "mismatch"}]


def test_reading_stats_and_pairing():
    st = A.reading_stats(rows(), lambda r: r["api"])
    assert st["agreement"]["k"] == 2 and st["agreement"]["n"] == 4 and st["badFlagged"]["k"] == 1 and st["badMismatch"]["k"] == 1
    assert st["okRejected"] == ["ok2", "ok3"] and st["badDecisions"] == {"bad1": "match", "bad2": "mismatch"}
    p = A.pairing(rows(), lambda r: r["api"])
    assert p["okAccepted"]["k"] == 1 and p["badAccepted"] == ["bad1"] and p["queued"] == 4 and p["items"] == 6
    assert p["bothMismatchOk"] == ["ok3"] and p["badOutcomes"] == {"bad1": "auto-accepted", "bad2": "disagree"}


def test_control_stats_count_repeat_misses():
    rs = [{"id": "a", "oneEditEdit": "repeat", "oneEditWords": 2, "d": {"oneEdit": "match", "shifted": "mismatch"}},
          {"id": "b", "oneEditEdit": "wrong_word", "oneEditWords": 1, "d": {"oneEdit": "unsure", "shifted": "mismatch"}},
          {"id": "c", "oneEditEdit": "sub", "oneEditWords": 1, "d": {"oneEdit": "match", "shifted": "match"}}]
    st = A.control_stats(rs, lambda r, k: r["d"][k])
    assert st["oneEdit"]["k"] == 1 and st["oneWord"] == {"k": 1, "n": 2, "rate": 0.5, "ci95": st["oneWord"]["ci95"]}
    assert st["repeatMisses"] == ["a"] and st["oneEditMisses"] == ["a", "c"] and st["shifted"]["k"] == 2


def test_pick_winner_is_lexicographic():
    base = {"badAccepted": 0, "okAccepted": 50, "badFlagged": 10, "oneEditFlagged": 40, "costPerHour": 0.75}
    assert A.pick_winner({"soniox": {**base, "badAccepted": 1, "okAccepted": 90}, "speechmatics": base}) == "speechmatics"
    assert A.pick_winner({"soniox": {**base, "okAccepted": 51}, "speechmatics": base}) == "soniox"
    assert A.pick_winner({"soniox": {**base, "oneEditFlagged": 39}, "speechmatics": base}) == "speechmatics"
    assert A.pick_winner({"soniox": {**base, "costPerHour": 0.1}, "speechmatics": base}) == "soniox"


def test_dedupe_keeps_the_earliest_verdict():
    r = [{"id": "v12-a", "audioSha256": "s", "claimed": "t"}, {"id": "v13-a", "audioSha256": "s", "claimed": "t"},
         {"id": "v13-b", "audioSha256": "s", "claimed": "u"}]
    kept, dropped = A.dedupe(r)
    assert [x["id"] for x in kept] == ["v12-a", "v13-b"] and dropped[0]["duplicateOf"] == "v12-a"


def test_triage_takes_a_judge_decision_as_the_second_model():
    s = session(60, seam_at=24, wrong_at=24)
    cases = T.session_cases(s, "lab-x")
    nem = {c["id"]: {"claimed": 0.0, "alternative": -10.0} for c in cases}
    api = {c["id"]: A.judge(c["claimed"] if c["group"] == "clean" else c["alternative"], c["claimed"], c["alternative"]) for c in cases}
    r = T.triage(cases, {"nemotron": nem, "soniox": api}, "lab-x", (0, 0), (0.0, 0), 17, models=("nemotron", "soniox"))
    out = {c["id"]: c["outcome"] for c in r["cases"]}
    assert out["lab-x-p01-x1"] == "disagree" and all(o == "auto-accepted" for i, o in out.items() if i != "lab-x-p01-x1")
    q = r["queue"]
    assert validate_queue(q) == [] and [it["id"] for it in q["items"]] == ["lab-x-p01-x1"]
    it = q["items"][0]
    assert it["triage"]["soniox"]["decision"] == "mismatch" and "fastconformer" not in it["triage"]
    assert it["context"].startswith("Nemotron: the label fits better than the text without the planted mistake by 10.0 nats; "
                                    "Soniox: what it heard is closer to the text without the planted mistake than to the label, by ")
    assert r["counts"]["perModel"] == {"nemotron": {"match": len(cases)}, "soniox": {"match": len(cases) - 1, "mismatch": 1}}
    cmp = T.compare_reviewed(r["cases"], {"items": [it]}, {it["id"]: {"verdict": "ok"}}, ("nemotron", "soniox"))
    assert cmp["items"][it["id"]]["soniox"] == "mismatch" and "sonioxDiff" in cmp["items"][it["id"]]
    assert WORDS  # the shared fixture


def test_default_triage_context_is_unchanged():
    s = session(60, seam_at=24, wrong_at=24)
    cases = T.session_cases(s, "lab-x")
    sc = {m: {c["id"]: {"claimed": -10.0, "alternative": 0.0} for c in cases} for m in T.MODELS}
    it = next(it for it in T.triage(cases, sc, "lab-x", (0, 0), (0.0, 0), 17)["queue"]["items"] if it["id"] == "lab-x-p01-x1")
    assert it["context"].startswith("Nemotron: the text without the planted mistake fits better than the label by 10.0; "
                                    "FastConformer: the text without the planted mistake fits better than the label by 10.0 (nats)")

"""uv run --python cpython-3.12-macos-aarch64-none --with numpy --with rapidfuzz==3.14.1 --with quran-transcript==0.6.4 --with pyarrow --with pytest pytest tools/test_lab_realuser.py"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lab_realuser_core as U  # noqa: E402
import lab_recordings_core as R  # noqa: E402
import lab_scripts as T  # noqa: E402
import lab_sources as L  # noqa: E402
from test_lab_checks import evidence  # noqa: E402

REF = L.reference()
Q = T.Quran(REF)
WORDS = Q.ref.words
word_index = lambda r: Q.ref.idx[(r["surah"], r["ayah"], r["word"])]
ordinal = lambda r: Q.ref.ayah_first[(r["surah"], r["ayah"])]
OPTIONAL = {"istiadha": T.optional(Q, "istiadha")}
BASMALA = {"text": T.BASMALA, "phonemes": Q.basmala["p"]}


def clip(cid, source, speaker, s, a, w0=1, w1=None, dur=4.0, **kw):
    n = len(Q.ref.ayah_words[f"{s}:{a}"])
    return {"id": cid, "source": source, "speaker": speaker, "from": U.at(s, a, w0), "to": U.at(s, a, w1 or n), "durationS": dur, "order": kw.pop("order", None),
            "origin": kw.pop("origin", {}), **kw}


def test_norm_merges_orthographies():
    assert U.norm("إِنَّ ٱلۡإِنسَـٰنَ لَفِی خُسۡرٍ") == U.norm("إِنَّ ٱلْإِنسَٰنَ لَفِى خُسْرٍ") == "انالانسنلفيخسر"


def test_map_text_finds_a_fragment_of_ayat_al_kursi():
    texts = U.ayah_texts(Q.ref.ayah_words)
    frm, to, d = U.map_text("لَا تَأْخُذُهُ سِنَةٌ وَلَا نَوْمٌ", "2:255", Q.ref.ayah_words, texts)
    assert (frm, to, d) == (U.at(2, 255, 8), U.at(2, 255, 12), 0.0)
    frm, to, d = U.map_text("قُلۡ أَعُوذُ بِرَبِّ ٱلنَّاسِ", 114, Q.ref.ayah_words, texts)
    assert (frm, to) == (U.at(114, 1, 1), U.at(114, 1, 4)) and d == 0.0


def test_verify_clip_statuses():
    a = "بِسمِللَاهِ"
    assert U.verify_clip(a, a, [""])["status"] == "ok"
    assert U.verify_clip("xx" + a, a, ["", "xx"])["status"] == "ok"  # an optional item said first
    v = U.verify_clip("qqqqqqqqqqq", a, [""], {"9:9": "qqqqqqqqqqq"})
    assert v["status"] == "relabel" and v["to"] == "9:9"
    assert U.verify_clip(a[:-3] + "qqq", a, [""])["status"] == "unsure"
    assert U.verify_clip("zzzzzzzzzzz", a, [""])["status"] == "drop"


def test_splits_hold_out_half_the_speakers_per_source():
    cs = [clip(f"c{i}", src, f"{src}/{i}", 1, 1) for src in U.SOURCES for i in range(7)]
    sp = U.splits(cs)
    for src in U.SOURCES:
        assert sum(v == "exam" for k, v in sp.items() if k.startswith(src)) == 3
    assert sp == U.splits(list(reversed(cs)))


def test_retasy_passes_split_repeats_and_short_passes_join_the_one_before():
    cs = [clip("a1", "retasy", "r/1", 1, 1, dur=3), clip("a2", "retasy", "r/1", 1, 2, dur=3), clip("a3", "retasy", "r/1", 1, 1, dur=2.5),
          clip("b1", "retasy", "r/2", 112, 1, dur=2)]
    ps = U.passes(cs[:3], ordinal)
    assert [[c["id"] for c in p] for p in ps] == [["a1", "a2"], ["a3"]]
    sessions, dropped = U.plan_sessions(cs, {"r/1": "practice", "r/2": "exam"}, ordinal, word_index)
    assert [s["clips"] for s in sessions] == [["a1", "a2", "a3"]]  # a3 alone is under 5 s
    assert dropped == [{"id": "b1", "source": "retasy", "kind": "short", "why": "under 5 s even joined (2.0 s)"}]


def test_tlog_sample_takes_contiguous_chains_round_robin():
    cs = [clip("t1", "tlog", "tlog/1", 100, 1, dur=3), clip("t2", "tlog", "tlog/1", 100, 2, dur=3), clip("t3", "tlog", "tlog/1", 100, 5, dur=2),
          clip("t4", "tlog", "tlog/2", 101, 1, dur=6)]
    chains = U.tlog_sample(cs, word_index, target_s=100)
    assert sorted([c["id"] for c in ch] for ch in chains) == [["t1", "t2"], ["t4"]]  # t3 alone is under 5 s


def take(clips: list[dict], sid="realuser-v1-sob-001", said_istiadha=False):
    """The script of `clips` as the teacher would hear it (optional items only where asked), timed on the join plan."""
    sc = U.script(Q.ref, sid, clips, OPTIONAL, BASMALA)
    placed = U.join_plan([c["durationS"] for c in clips])
    spoken = []
    for k, (a, b) in enumerate(placed):
        its = [x for x in sc["said"] if x["clip"] == k and (x["role"] == "ref" or (said_istiadha and k == 0 and x["role"] == "istiadha"))]
        n, t = sum(len(x["phonemes"]) for x in its), a + 0.2
        for x in its:
            d = (b - a - 0.4) * len(x["phonemes"]) / n
            spoken.append((x["phonemes"], round(t, 3), round(t + d - 0.03, 3)))
            t += d
    duration = round(placed[-1][1] + U.EDGE_S, 3)
    return sc, R.check_take(sc, evidence(spoken, duration), REF), placed


def test_joined_session_labels_validate_with_seams_repeats_and_tags():
    clips = [clip("sob-a-c0", "sobolev", "sobolev/a", 112, 1, dur=3.5, order=0),
             clip("sob-a-c1", "sobolev", "sobolev/a", 112, 2, dur=3.0, order=1),
             clip("sob-a-c2", "sobolev", "sobolev/a", 112, 2, dur=3.0, order=2, candidates=[{"word": "الصمد", "position": 2, "tags": ["Letters"]}]),
             clip("sob-a-c3", "sobolev", "sobolev/a", 112, 4, dur=4.0, order=3)]
    sc, res, placed = take(clips, said_istiadha=True)
    assert res["aligned"] and res["coverage"] == 1.0 and [sc["said"][i]["role"] for i in res["dropped"]] == ["basmala"]
    plan = {"id": "realuser-v1-sob-001", "source": "sobolev", "speaker": "sobolev/a", "split": "practice"}
    s = U.session(plan, clips, res, "sessions/realuser-v1-sob-001.phone.wav", placed, word_index)
    assert R.validate(s, {(w["s"], w["a"], w["w"]) for w in WORDS}) == []
    assert [x["kind"] for x in s["spans"]] == ["non_quran", "recitation", "recitation", "repeat", "recitation"]
    assert {n["kind"] for n in s["nonErrors"]} == {"repeat"} and len(s["nonErrors"]) == 2
    assert [x["kind"] for x in s["meta"]["seams"]] == ["contiguous", "jump", "jump"]
    assert s["meta"]["candidates"][0]["tags"] == ["Letters"] and s["events"] == []
    m = U.manifest([s])
    assert m["sessions"][0]["slices"] == {"source": "realuser", "origin": "sobolev", "clean": True}
    q = U.flagged_queue([s])
    assert [it["title"] for it in q["items"]] == ["Reviewer tag: Letters"]
    assert U.flagged_queue([{**s, "meta": {**s["meta"], "split": "exam"}}])["items"] == []


def test_separate_recordings_get_their_own_optional_items_and_partial_ayahs():
    clips = [clip("tlog-1", "tlog", "tlog/1", 2, 255, 8, 12, dur=3.0), clip("tlog-2", "tlog", "tlog/1", 2, 255, 13, 20, dur=4.0)]
    sc = U.script(Q.ref, "x", clips, OPTIONAL, BASMALA)
    assert [x["role"] for x in sc["said"] if x["clip"] == 1][:1] == ["istiadha"]
    _, res, placed = take(clips)
    s = U.session({"id": "x", "source": "tlog", "speaker": "tlog/1", "split": "exam"}, clips, res, "a.wav", placed, word_index)
    assert R.validate(s, {(w["s"], w["a"], w["w"]) for w in WORDS}) == [] and [x["kind"] for x in s["meta"]["seams"]] == ["contiguous"]


def test_join_audio_level_matches_and_lays_out_the_plan():
    import numpy as np

    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 0.1, 16000).astype(np.float32), rng.normal(0, 0.01, 8000).astype(np.float32)
    x, gains = U.join_audio([a, b])
    assert len(x) == round((U.EDGE_S * 2 + 1.0 + U.GAP_S + 0.5) * 16000) and gains == [-10.0, 10.0]
    random.seed(0)


def test_best_span_finds_the_words_a_chunk_holds():
    from per_breakdown import canon

    ids = [i for i, w in enumerate(WORDS) if w["s"] == 101]
    words = [canon(WORDS[i]["cont"]) for i in ids]
    heard = "".join(words[3:9])  # 101:3-4, a chunk labelled 101:3
    i, j, per = U.best_span(heard, words, [""])
    assert (i, j, per) == (3, 8, 0.0)
    v = U.verify_clip(heard, "".join(words[3:7]), [""], None, (ids[i], ids[j], per))
    assert v["status"] == "relabel" and v["span"] == [ids[3], ids[8]]


def test_seam_exclusions_cover_jump_seams_only():
    s = {"id": "x", "durationS": 20.0, "meta": {"seams": [{"atS": 4.0, "kind": "jump", "before": "c1"}, {"atS": 9.0, "kind": "contiguous", "before": "c2"},
                                                         {"atS": 19.5, "kind": "jump", "before": "c3"}]}}
    x = U.seam_exclusions([s], extra=[{"session": "x", "startS": 1.0, "endS": 2.0, "verdict": "bad_audio"}])
    assert x["kind"] == "lab-flags" and x["base"] == x["testSet"] == U.TEST_SET
    assert [(e["startS"], e["endS"]) for e in x["excluded"]] == [(3.0, 7.0), (18.5, 20.0), (1.0, 2.0)]


def test_a_reviewed_omission_becomes_a_mistake_in_the_gap():
    clips = [clip("sob-a-c0", "sobolev", "sobolev/a", 112, 1, dur=3.5, order=0), clip("sob-a-c1", "sobolev", "sobolev/a", 112, 2, dur=3.0, order=1)]
    _, res, placed = take(clips)
    s = U.session({"id": "x", "source": "sobolev", "speaker": "sobolev/a", "split": "practice"}, clips, res, "a.wav", placed, word_index)
    fixes = [{"session": "x", "kind": "omitted_word", "at": U.at(112, 1, 3), "item": "i1"}, {"session": "x", "kind": "omitted_word", "at": U.at(112, 1, 4), "item": "i2"}]
    c = U.correct(s, fixes, word_index)
    assert R.validate(c, {(w["s"], w["a"], w["w"]) for w in WORDS}) == []
    assert [e["location"] for e in c["events"]] == [U.at(112, 1, 3), U.at(112, 1, 4)] and not c["meta"]["clean"]
    last = c["events"][-1]
    assert last["startS"] == last["endS"] and c["spans"][0]["to"] == U.at(112, 1, 2)
    assert U.manifest([c])["sessions"][0]["slices"]["clean"] is False and s["events"] == []

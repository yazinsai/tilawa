"""uv run --python cpython-3.12-macos-aarch64-none --with numpy --with rapidfuzz==3.14.1 --with quran-transcript==0.6.4 --with pyarrow --with pytest pytest tools/test_lab_checks.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lab_checks as C  # noqa: E402
import lab_sources as L  # noqa: E402
from e1_report import TOKEN  # noqa: E402

REF = L.reference()
WORDS = C.word_phonemes(REF)
FS = 0.04


def evidence(spoken: list[tuple[str, float, float]], duration: float) -> dict:
    """Confident spikes for each word's squeezed phonemes over its span, blank elsewhere (top-5 posteriors)."""
    n = round(duration / FS)
    top = [[0, 1, 2, 3, 4] for _ in range(n)]
    probs = [[0.97, 0.01, 0.01, 0.005, 0.005] for _ in range(n)]
    phonemes = []
    for p, a, b in spoken:
        q = L.squeeze(p)
        f0, width = round(a / FS), round((b - a) / FS)
        for j, c in enumerate(q):
            f = f0 + (j * width) // len(q)
            t = TOKEN[c]
            top[f] = [t] + [x for x in (0, 1, 2, 3, 4) if x != t][:4]
            probs[f] = [0.95, 0.03, 0.01, 0.005, 0.005]
            phonemes.append({"p": c, "start": f, "end": f, "conf": 0.95})
    return {"inventory": REF["inventory"]["id"], "frameMs": 40, "nFrames": n, "phonemes": phonemes,
            "posteriors": {"k": 5, "ids": [x for r in top for x in r], "probs": [x for r in probs for x in r]}}


def session(wrong_said: str) -> tuple[dict, list[tuple[str, float, float]]]:
    """1:2 then 1:3 recited, with 1:2:3 replaced by a planted wrong word (`wrong_said`), a seam either side of it."""
    at = lambda s, a, w: {"surah": s, "ayah": a, "word": w}
    plan = [((1, 2, 1), "ref"), ((1, 2, 2), "ref"), ((1, 2, 3), "wrong"), ((1, 2, 4), "ref"), ((1, 3, 1), "ref"), ((1, 3, 2), "ref")]
    said, t = [], 1.0
    for (s, a, w), role in plan:
        p = wrong_said if role == "wrong" else WORDS[f"{s}:{a}:{w}"]
        dur = 0.12 * len(L.squeeze(p)) + 0.2
        said.append({"startS": round(t, 3), "endS": round(t + dur, 3), "at": at(s, a, w), "phonemes": p, "role": role, **({"op": "x1"} if role == "wrong" else {})})
        t += dur + 0.2
    w = said[2]
    s = {"id": "t-1", "durationS": round(t + 1.0, 3), "spans": [], "said": said, "meta": {"readId": 3, "split": "exam", "reciter": "x"},
         "segments": [{"startS": 0.0}, {"startS": w["startS"] - 0.1}, {"startS": w["endS"] + 0.1}],
         "splices": [{"id": "x1", "type": "wrong_word", "mutashabihat": False, "location": at(1, 2, 3), "startS": w["startS"], "endS": w["endS"], "removed": ["1:2:3"]}]}
    return s, [(x["phonemes"], x["startS"], x["endS"]) for x in said]


def test_planted_word_heard_passes():
    s, spoken = session(WORDS["1:3:1"])
    r = C.check_session(s, evidence(spoken, s["durationS"]), REF, "studio")
    (e,) = r["edits"]
    assert e["pass"], e
    assert e["llr"] >= C.LLR_MIN and e["editsSaid"] == 0 and e["editsOriginal"] > 0
    assert [c["pass"] for c in r["clean"]] == [True, True] and r["cleanPer"] == 0
    assert r["timing"]["pass"], r["timing"]


def test_original_heard_fails():
    s, spoken = session(WORDS["1:3:1"])
    spoken[2] = (WORDS["1:2:3"], spoken[2][1], spoken[2][2])
    r = C.check_session(s, evidence(spoken, s["durationS"]), REF, "studio")
    (e,) = r["edits"]
    assert not e["pass"]
    assert any("prefers the original" in x for x in e["reasons"]), e["reasons"]


def test_clean_stretch_misheard_fails():
    s, spoken = session(WORDS["1:3:1"])
    spoken[0] = (WORDS["1:4:1"], spoken[0][1], spoken[0][2])
    r = C.check_session(s, evidence(spoken, s["durationS"]), REF, "studio")
    assert [c["pass"] for c in r["clean"]] == [False, True]


def test_stretches_break_at_seams_and_planted_words():
    s, _ = session(WORDS["1:3:1"])
    assert C.stretches(s, C.items(s, REF)) == [[0, 1], [3, 4, 5]]


def test_reciter_check():
    reciters = {"heldOut": [{"readId": 3, "split": "exam"}], "excluded": [{"readId": 1, "reason": "excluded_teacher"}]}
    assert C.reciter_check({"meta": {"readId": "3", "split": "exam", "reciter": "a"}}, reciters)["pass"]
    assert not C.reciter_check({"meta": {"readId": 1, "split": "exam", "reciter": "b"}}, reciters)["pass"]
    assert not C.reciter_check({"meta": {"readId": 3, "split": "practice", "reciter": "a"}}, reciters)["pass"]


def test_review_queue_takes_failed_edits():
    def sess(i, types):
        sp = [{"id": f"x{k + 1}", "type": t, "mutashabihat": False, "formsOtherAyah": False, "location": {"surah": 1, "ayah": 2, "word": 1},
               "startS": 10.0 * k + 5, "endS": 10.0 * k + 6, "expected": "a", "said": "b", "donor": None,
               "seams": [{"out": {"audio": "src.wav", "srcS": 3.0}}]} for k, t in enumerate(types)]
        return {"id": i, "durationS": 100.0, "said": [], "splices": sp, "meta": {"audio": {"studio": f"sessions/{i}.studio.wav"}, "reciter": "r"}}
    sessions = {"a": sess("a", ["wrong_word", "added_word", "repeat"]), "b": sess("b", ["wrong_word", "omitted_word", "restart"])}
    item = lambda i, kind, t: {"id": i, "kind": kind, "label": {"type": t}}
    queue = {"version": 1, "testSet": "t", "items": [item("a-x1", "splice", "wrong_word"), item("a-x2", "splice", "added_word"), item("b-x2", "splice", "omitted_word"),
                                                     item("a-x3", "case", "repeat"), item("t-clean-1", "case", "clean")]}
    f = lambda i, k, t: {"id": i, "session": i[0], "splice": k, "type": t, "class": t, "reasons": ["r"]}
    q, ch = C.review_with_failures(queue, sessions, [f("b-x1", "x1", "wrong_word"), f("a-x2", "x2", "added_word"), f("b-x3", "x3", "restart")], "t")
    ids = [it["id"] for it in q["items"]]
    assert ids == ["b-x1", "a-x2", "b-x2", "b-x3", "t-clean-1"]
    assert {r["out"]: r["in"] for r in ch["replaced"]} == {"a-x1": "b-x1", "a-x3": "b-x3"}
    assert ch["alreadyQueued"] == ["a-x2"] and ch["notQueued"] == []
    assert all(it.get("check", {}).get("pass") is False for it in q["items"] if it["id"] in ("b-x1", "a-x2", "b-x3"))
    assert "check" not in q["items"][2]

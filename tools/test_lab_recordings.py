"""uv run --python cpython-3.12-macos-aarch64-none --with numpy --with rapidfuzz==3.14.1 --with quran-transcript==0.6.4 --with pyarrow --with pytest pytest tools/test_lab_recordings.py"""

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lab_recordings_core as R  # noqa: E402
import lab_review_core as V  # noqa: E402
import lab_scripts as T  # noqa: E402
import lab_sources as L  # noqa: E402
from test_lab_checks import evidence  # noqa: E402

REF = L.reference()
Q = T.Quran(REF)
REF_WORDS = {(w["s"], w["a"], w["w"]) for w in Q.ref.words}
PLANTS = {2: "wrong_word", 4: "added_word", 6: "self_correction", 8: "omitted_word"}


def fixed_script() -> dict:
    """Al-Humaza (104) with its basmala and one plant of each class, in ayahs 2, 4, 6 and 8."""
    rng = random.Random(7)
    plants = []
    for a, cls in PLANTS.items():
        (o, *_) = T.site_options(Q, 104, a, cls, rng)
        plants.append({**o, "ayahIndex": a - 1})
    return T.script(Q, "rec-test-01", "juz-amma", [(104, a) for a in range(1, 10)], False, 1, plants)


SCRIPT = fixed_script()
META = {"speaker": "test", "take": 1, "device": "synthetic", "recordedAt": "2026-10-29T00:00:00Z"}


def spoken(items: list[dict]) -> tuple[list[tuple[str, float, float]], float]:
    out, t, ayah = [], 1.0, None
    for x in items:
        key = x["at"] and (x["at"]["surah"], x["at"]["ayah"])
        if key and key != ayah:
            t, ayah = t + 0.5, key
        dur = 0.09 * len(L.squeeze(x["phonemes"])) + 0.1
        out.append((x["phonemes"], round(t, 3), round(t + dur, 3)))
        t += dur + 0.15
    return out, round(t + 1.0, 3)


def as_written(script: dict) -> list[dict]:
    """The passage recited as written: no plant performed."""
    out = []
    for x in script["said"]:
        if x["role"] in ("added", "slip") or (x["role"] == "retake" and x["at"] != next(p for p in script["plants"] if p["id"] == x["plant"])["location"]):
            continue
        if x["role"] in ("wrong", "retake"):
            w = Q.ref.words[Q.ref.idx[(x["at"]["surah"], x["at"]["ayah"], x["at"]["word"])]]
            x = {**x, "phonemes": w["cont"]}
        out.append(x)
    for p in script["plants"]:
        if p["class"] == "omitted_word":
            loc = p["location"]
            k = next(i for i, x in enumerate(out) if x["at"] == {**loc, "word": loc["word"] + 1})
            w = Q.ref.words[Q.ref.idx[(loc["surah"], loc["ayah"], loc["word"])]]
            out.insert(k, {"at": loc, "phonemes": w["cont"], "role": "ref"})
    return out


def run(items: list[dict], script: dict = SCRIPT) -> dict:
    sp, duration = spoken(items)
    return R.check_take(script, evidence(sp, duration), REF)


def test_fixed_script_has_every_class():
    assert [p["class"] for p in SCRIPT["plants"]] == list(PLANTS.values())
    assert SCRIPT["said"][0]["role"] == "basmala"


def test_take_as_scripted_is_ok():
    r = run(SCRIPT["said"])
    assert r["aligned"] and r["coverage"] == 1.0 and not r["droppedBasmala"]
    assert [p["status"] for p in r["plants"]] == ["performed"] * 4, [(p["class"], p["reasons"]) for p in r["plants"]]
    assert all(p["llr"] >= R.LLR_MIN for p in r["plants"])
    assert not any(a["flag"] for a in r["ayahs"]) and len(r["ayahs"]) == 9
    assert r["verdict"] == "ok" and not r["why"]


def test_take_as_written_is_redo():
    r = run(as_written(SCRIPT))
    assert r["aligned"]
    assert [p["status"] for p in r["plants"]] == ["not_performed"] * 4, [(p["class"], p["llr"], p["reasons"]) for p in r["plants"]]
    assert r["verdict"] == "redo" and len(r["why"]) >= 4


def test_unsaid_basmala_is_dropped():
    r = run(SCRIPT["said"][1:])
    assert r["droppedBasmala"] == [0] and r["verdict"] == "ok"
    s = R.session(SCRIPT, r, META, "recordings/sessions/x.phone.wav")
    assert not any(sp["kind"] == "non_quran" for sp in s["spans"])


def test_session_is_valid_and_labelled():
    r = run(SCRIPT["said"])
    s = R.session(SCRIPT, r, META, "recordings/sessions/rec-test-rec-test-01-t1.phone.wav")
    assert R.validate(s, REF_WORDS) == []
    assert s["id"] == "rec-test-rec-test-01-t1" and s["source"] == "recorded" and s["context"] == {"kind": "none"} and s["speaker"] == "rec-test"
    assert [sp["kind"] for sp in s["spans"]] == ["non_quran", "recitation", "repeat", "recitation"]
    assert sorted(e["errorClass"] for e in s["events"]) == ["added_word", "omitted_word", "wrong_word"]
    assert all(e["said"] for e in s["events"] if e["errorClass"] != "omitted_word")
    sc = next(p for p in SCRIPT["plants"] if p["class"] == "self_correction")
    assert [n["at"] for n in s["nonErrors"]] == sc["retake"] and all(n["kind"] == "self_correction" for n in s["nonErrors"])
    assert all(n["span"] == s["spans"][2]["id"] for n in s["nonErrors"])
    wrong = next(e for e in s["events"] if e["errorClass"] == "wrong_word")
    w = next(w for w in s["words"] if w["at"] == wrong["location"])
    assert (w["startS"], w["endS"]) == (wrong["startS"], wrong["endS"]) and w["said"]["text"] == wrong["observed"]
    assert s["meta"]["verdict"] == "ok" and not s["meta"]["clean"]


def test_clean_script_session():
    clean = T.script(Q, "rec-test-02", "juz-amma", [(104, a) for a in range(1, 10)], True, 1)
    r = run(clean["said"], clean)
    s = R.session(clean, r, META, "x.wav")
    assert r["verdict"] == "ok" and not s["events"] and not s["nonErrors"] and s["meta"]["clean"]
    assert R.validate(s, REF_WORDS) == []


def test_review_items_pass_the_queue_check():
    r = run(SCRIPT["said"])
    s = R.session(SCRIPT, r, META, "x.wav")
    its = R.review_items(s, r, SCRIPT, "recordings/test/rec-test-01/take-1.16k.wav")
    kinds = [it["label"]["type"] for it in its]
    assert kinds.count("self_correction") == 1 and kinds.count("clean") <= R.SPOT["clean"]
    assert len({it["id"] for it in its}) == len(its)
    q = R.queue([its])
    V.validate_queue(q)
    assert R.review_items(s, r, SCRIPT, "w") == R.review_items(s, r, SCRIPT, "w")


def test_unsure_plants_are_queued():
    r = run(as_written(SCRIPT))
    r["verdict"] = "listen"
    s = R.session(SCRIPT, r, META, "x.wav")
    its = R.review_items(s, r, SCRIPT, "w")
    assert sum(it["title"].endswith(R.STATUS["not_performed"]) for it in its) == 4


def test_paths_and_names():
    f = R.take_files("ali", "rec-v1-01", 2, "webm")
    assert (f["upload"], f["meta"], f["result"]) == ("recordings/ali/rec-v1-01/take-2.webm", "recordings/ali/rec-v1-01/take-2.json", "recordings/ali/rec-v1-01/take-2.result.json")
    assert f["wav"] == "recordings/sessions/rec-ali-rec-v1-01-t2.phone.wav" and f["session"] == "recordings/sessions/rec-ali-rec-v1-01-t2.json"
    assert f["evidence"] == "recordings/sessions/rec-ali-rec-v1-01-t2.teacher.phone.evidence.json"
    assert R.next_take(["take-1.webm", "take-1.json", "take-3.wav", "take-x.webm", "notes.txt"]) == 4 and R.next_take([]) == 1
    assert R.ext_for("audio/webm;codecs=opus") == "webm" and R.ext_for("audio/mp4") == "mp4"
    for bad in ("", "../x", "a/b", "-x", "x" * 65):
        with pytest.raises(ValueError):
            R.safe(bad, "speaker")
    with pytest.raises(ValueError):
        R.ext_for("video/quicktime")


def test_take_meta_checks_the_script_copy():
    m = R.take_meta({"speaker": "ali", "scriptVersion": "1", "scriptSha": SCRIPT["sha"], "device": "MacBook mic", "durationS": "212.5"}, SCRIPT, 1, "webm", "t", 10)
    assert m["speaker"] == "ali" and m["scriptSha"] == SCRIPT["sha"] and m["clientDurationS"] == 212.5 and m["recordedAt"] == "t"
    with pytest.raises(ValueError, match="old copy"):
        R.take_meta({"speaker": "ali", "scriptSha": "stale"}, SCRIPT, 1, "webm", "t", 10)


def test_verdicts():
    assert R.decide(5.0, 0, 3, 0.0) == ("performed", [])
    assert R.decide(-5.0, 3, 0, 0.0)[0] == "not_performed"
    assert R.decide(1.0, 1, 2, 0.0)[0] == "unsure"
    p = lambda st: {"n": 1, "ayah": "1:1", "status": st, "reasons": ["x"]}
    a = lambda f: {"key": "1:1", "flag": f, "per": 0.3}
    assert R.take_verdict(True, [p("performed")], [a(False)], 1.0) == ("ok", [])
    assert R.take_verdict(True, [p("unsure")], [a(False)], 1.0)[0] == "listen"
    assert R.take_verdict(True, [p("performed")], [a(True)] * 2, 1.0)[0] == "listen"
    assert R.take_verdict(True, [p("performed")], [a(True)] * 3, 1.0)[0] == "redo"
    assert R.take_verdict(True, [p("not_performed")], [], 1.0)[0] == "redo"
    assert R.take_verdict(True, [], [], 0.5)[0] == "redo"
    assert R.take_verdict(False, [], [], 0.0)[0] == "redo"


def test_current_takes_and_manifest_join():
    rs = [{"speaker": "a", "script": "s1", "take": 1, "verdict": "ok"}, {"speaker": "a", "script": "s1", "take": 2, "verdict": "redo"},
          {"speaker": "a", "script": "s1", "take": 3, "verdict": "listen"}, {"speaker": "b", "script": "s1", "take": 1, "verdict": "redo"}]
    assert [(r["speaker"], r["take"]) for r in R.current_takes(rs)] == [("a", 3)]
    r = run(SCRIPT["said"])
    s = R.session(SCRIPT, r, META, "sessions/rec-test-rec-test-01-t1.phone.wav")
    m = R.manifest([s], {"test": "exam"})
    (e,) = m["sessions"]
    assert m["variants"] == ["phone"] and e["split"] == "exam" and e["tags"] == ["recorded"] and e["slices"]["source"] == "recorded"
    assert "said" not in e["session"] and "meta" not in e["session"] and e["audio"]["phone"] == s["audio"]
    base = {"version": 1, "testSet": "lab-v1.4", "variants": ["studio", "phone"], "sessions": [{"id": "t-1", "session": {}}]}
    j = R.join(base, m, "recorded/")
    assert len(j["sessions"]) == 2 and j["sessions"][1]["audio"] == {"phone": "recorded/rec-test-rec-test-01-t1.phone.wav"}
    assert j["sessions"][1]["session"]["audio"] == "recorded/rec-test-rec-test-01-t1.phone.wav" and j["recorded"]["sessions"] == 1
    with pytest.raises(ValueError):
        R.join(j, m, "recorded/")


CLI = Path(__file__).resolve().parents[1] / "packages/core/dist/bench/lab-validate-cli.js"


@pytest.mark.skipif(not CLI.exists() or not shutil.which("node"), reason="build packages/core first")
def test_core_validate_session(tmp_path):
    r = run(SCRIPT["said"])
    s = R.session(SCRIPT, r, META, "sessions/rec-test-rec-test-01-t1.phone.wav")
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / f"{s['id']}.json").write_text(json.dumps(s))
    (tmp_path / "sessions" / f"{s['id']}.phone.wav").write_bytes(b"")
    (tmp_path / "manifest-v1.json").write_text(json.dumps(R.manifest([s], {})))
    out = subprocess.run(["node", str(CLI), "--dir", str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert json.loads(out.stdout)["manifestSessions"] == 1

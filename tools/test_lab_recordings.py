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
    assert [x["role"] for x in SCRIPT["said"][:2]] == ["istiadha", "basmala"] and SCRIPT["said"][-1]["role"] == "closing"


def test_take_as_scripted_is_ok():
    r = run(SCRIPT["said"])
    assert r["aligned"] and r["coverage"] == 1.0 and not r["dropped"]
    assert [p["status"] for p in r["plants"]] == ["performed"] * 4, [(p["class"], p["reasons"]) for p in r["plants"]]
    assert all(p["llr"] >= R.LLR_MIN for p in r["plants"])
    assert not any(a["flag"] for a in r["ayahs"]) and len(r["ayahs"]) == 9
    assert r["verdict"] == "ok" and not r["why"]


def test_take_as_written_is_redo():
    r = run(as_written(SCRIPT))
    assert r["aligned"]
    assert [p["status"] for p in r["plants"]] == ["not_performed"] * 4, [(p["class"], p["llr"], p["reasons"]) for p in r["plants"]]
    assert r["verdict"] == "redo" and len(r["why"]) >= 4


def test_silent_take_is_redo():
    _, duration = spoken(SCRIPT["said"])
    r = R.check_take(SCRIPT, evidence([], duration), REF)
    assert r["verdict"] == "redo" and "hears almost none" in r["why"][0] and r["heard"] == 0


def test_unsaid_optional_items_are_dropped():
    last = len(SCRIPT["said"]) - 1
    r = run(SCRIPT["said"][1:-1])
    assert r["dropped"] == [0, last] and r["verdict"] == "ok", r["why"]
    r = run(SCRIPT["said"][2:-1])
    assert r["dropped"] == [0, 1, last] and r["verdict"] == "ok", r["why"]
    s = R.session(SCRIPT, r, META, "recordings/sessions/x.phone.wav")
    assert not any(sp["kind"] == "non_quran" for sp in s["spans"])


def test_session_is_valid_and_labelled():
    r = run(SCRIPT["said"])
    s = R.session(SCRIPT, r, META, "recordings/sessions/rec-test-rec-test-01-t1.phone.wav")
    assert R.validate(s, REF_WORDS) == []
    assert s["id"] == "rec-test-rec-test-01-t1" and s["source"] == "recorded" and s["context"] == {"kind": "none"} and s["speaker"] == "rec-test"
    assert [sp.get("label", sp["kind"]) for sp in s["spans"]] == ["istiadha", "basmala", "recitation", "repeat", "recitation", "closing"]
    assert sorted(e["errorClass"] for e in s["events"]) == ["added_word", "omitted_word", "wrong_word"]
    assert all(e["said"] for e in s["events"] if e["errorClass"] != "omitted_word")
    sc = next(p for p in SCRIPT["plants"] if p["class"] == "self_correction")
    assert [n["at"] for n in s["nonErrors"]] == sc["retake"] and all(n["kind"] == "self_correction" for n in s["nonErrors"])
    assert all(n["span"] == s["spans"][3]["id"] for n in s["nonErrors"])
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
    with pytest.raises(ValueError):
        R.take_dir("sessions", "rec-v1-01")


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
    j = R.join(base, m, "sessions/")
    assert len(j["sessions"]) == 2 and j["sessions"][1]["audio"] == {"phone": "sessions/rec-test-rec-test-01-t1.phone.wav"}
    assert j["sessions"][1]["session"]["audio"] == "sessions/rec-test-rec-test-01-t1.phone.wav" and j["recorded"]["sessions"] == 1
    with pytest.raises(ValueError):
        R.join(j, m, "sessions/")


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


def test_resume_points_are_kept_sorted_and_bad_input_dropped():
    assert R.resumes("[31.5, 12.25, -1, \"x\"]") == [12.25, 31.5]
    assert R.resumes("") == [] and R.resumes("not json") == [] and R.resumes("{}") == []


# ---- fixing a passage ---------------------------------------------------------------------------------------------

def energy(said: list[dict], duration: float, speech: float = -20.0, quiet: float = -70.0, loud: dict | None = None) -> list[float]:
    """10 ms frame dB: `speech` over every timed item, `quiet` elsewhere; `loud` {(a, b): dB} overrides a stretch."""
    db = [quiet] * int(duration * 100)
    for t in said:
        for i in range(int(t["startS"] * 100), min(len(db), int(t["endS"] * 100) + 1)):
            db[i] = speech
    for (a, b), v in (loud or {}).items():
        for i in range(int(a * 100), int(b * 100)):
            db[i] = v
    return db


def take_and_patch(a: str, b: str, items: list[dict] | None = None):
    """A take as scripted but mistake 2 (the added word in 104:4) left out, and a patch of a..b performing it."""
    take_items = items or [x for x in SCRIPT["said"] if x.get("plant") != "m2"]
    sp, dur = spoken(take_items)
    r = R.check_take(SCRIPT, evidence(sp, dur), REF)
    sub = R.fix_script(SCRIPT, a, b)
    psp, pdur = spoken(sub["said"])
    pr = R.check_take(sub, evidence(psp, pdur), REF)
    return r, dur, sub, pr, pdur


FIX = {"base": 1, "from": "104:4", "to": "104:4"}


def test_said_ayahs_put_added_words_and_the_basmala_in_their_ayah():
    ks = R.said_ayahs(SCRIPT)
    roles = [x["role"] for x in SCRIPT["said"]]
    assert ks[0] is None and ks[-1] is None  # isti'adha, closing
    assert ks[roles.index("basmala")] == "104:1" and ks[roles.index("added")] == "104:4"
    idx = R.fix_items(SCRIPT, "104:4", "104:5")
    assert idx == list(range(idx[0], idx[-1] + 1)) and {ks[i] for i in idx} == {"104:4", "104:5"}
    assert roles.index("added") in idx
    assert SCRIPT["said"][R.fix_items(SCRIPT, "104:1", "104:1")[0]]["role"] == "basmala"
    for a, b in (("104:5", "104:4"), ("104:0", "104:2"), ("2:1", "2:2")):
        with pytest.raises(ValueError):
            R.fix_items(SCRIPT, a, b)


def test_fix_ranges_from_the_checks():
    r = run(as_written(SCRIPT))
    assert r["verdict"] == "redo"
    fails = R.failed_ayahs(SCRIPT, r)
    assert list(fails) == ["104:2", "104:4", "104:6", "104:8"] and fails["104:2"] == ["mistake 1 not heard"]
    (one,) = R.fix_ranges(SCRIPT, r)  # two ayahs apart: one passage
    assert (one["from"], one["to"]) == ("104:2", "104:8") and list(one["ayahs"]) == list(fails)
    plants = [p if p["ayah"] in ("104:2", "104:8") else {**p, "status": "performed"} for p in r["plants"]]
    gap = {**r, "plants": plants, "ayahs": [{**a, "flag": False} for a in r["ayahs"]], "said": [{**t, "endS": t["startS"]} if t["at"] and t["at"]["ayah"] == 9 and t["at"]["word"] == 1 else t for t in r["said"]]}
    assert [(x["from"], x["to"]) for x in R.fix_ranges(SCRIPT, gap)] == [("104:2", "104:2"), ("104:8", "104:9")]
    assert R.fix_ranges(SCRIPT, gap)[1]["ayahs"]["104:9"] == ["1 word not found"]
    assert R.fix_ranges(SCRIPT, run(SCRIPT["said"])) == []  # ok
    assert R.fix_ranges(SCRIPT, {**r, "aligned": False}) == [] and R.fix_ranges(SCRIPT, {**r, "verdict": "error"}) == []


def test_fix_request_refusals():
    r = run(as_written(SCRIPT))
    m = {"take": 2, "scriptSha": SCRIPT["sha"]}
    assert R.fix_request(SCRIPT, m, r, "104:2", "104:3") == {"base": 2, "from": "104:2", "to": "104:3"}
    for meta, res, a, b, msg in ((None, r, "104:2", "104:2", "doesn't exist"), (m, None, "104:2", "104:2", "no checked alignment"),
                                 (m, {**r, "verdict": "error"}, "104:2", "104:2", "no checked alignment"),
                                 ({**m, "scriptSha": "old"}, r, "104:2", "104:2", "older version"), (m, r, "104:3", "104:2", "isn't a passage")):
        with pytest.raises(R.Refused, match=msg):
            R.fix_request(SCRIPT, meta, res, a, b)


def test_patch_is_checked_against_its_ayahs_only():
    r, _, sub, pr, _ = take_and_patch("104:4", "104:4")
    assert next(p for p in r["plants"] if p["id"] == "m2")["status"] == "not_performed" and r["verdict"] == "redo"
    assert [p["id"] for p in sub["plants"]] == ["m2"] and all("src" in x for x in sub["said"]) and sub["fix"] == ["104:4", "104:4"]
    assert pr["aligned"] and pr["verdict"] == "ok" and [p["status"] for p in pr["plants"]] == ["performed"], pr["why"]
    assert [t["src"] for t in pr["said"]] == R.fix_items(SCRIPT, "104:4", "104:4")
    s = R.patch_summary(pr)
    assert s["verdict"] == "ok" and s["plants"][0]["status"] == "performed" and s["flagged"] == []
    words = R.fix_script(SCRIPT, "104:4", "104:4")["said"]
    psp, pdur = spoken([x for x in words if x["role"] != "added"])
    assert R.check_take(R.fix_script(SCRIPT, "104:4", "104:4"), evidence(psp, pdur), REF)["verdict"] == "redo"


def test_a_patch_checks_plants_at_its_own_ends():
    """A patch of one ayah starts and ends inside the script: a plant on its first or last word is checked against
    the word on its one side, not left unsure."""
    edges = 0
    for p in SCRIPT["plants"]:
        sub = R.fix_script(SCRIPT, p["ayah"], p["ayah"])
        mine = [k for k, x in enumerate(sub["said"]) if x.get("plant") == p["id"]]
        edges += bool(mine) and (mine[0] == 0 or mine[-1] == len(sub["said"]) - 1)
        psp, pdur = spoken(sub["said"])
        (c,) = R.check_take(sub, evidence(psp, pdur), REF)["plants"]
        assert c["status"] == "performed", (p["class"], p["location"], c["reasons"])
    assert edges >= 1
    assert R.plant_window([{"plant": "m"}], {"id": "m"}) is None and R.plant_window([{"plant": "m"}], {"id": "m"}, open_ends=True) == (None, None, [0])


def test_pauses_and_cuts_on_frame_energy():
    db = [-20.0] * 100 + [-70.0] * 30 + [-20.0] * 50 + [-70.0] * 5 + [-20.0] * 15
    assert R.pauses(db, -50.0, 0.0, 2.0) == [(1.0, 1.3)]  # the 50 ms dip is too short
    assert R.pauses(db, -50.0, 1.1, 2.0) == [(1.1, 1.3)] and R.pauses(db, -50.0, 1.25, 2.0) == []
    assert R.pauses(db, -75.0, 0.0, 2.0) == []
    assert R.cut_in((1.0, 1.3), "before") == 1.15 and R.cut_in((1.0, 3.0), "before") == 1.4 and R.cut_in((1.0, 3.0), "after") == 2.6
    assert R.cut_in((1.0, 1.08), "after") == 1.04
    assert R.quiet_floor(db) == -50.0  # 30 dB under speech: the room (-70 dB) is quieter than that
    assert R.quiet_floor([-20.0] * 60 + [-45.0] * 40) == pytest.approx(-42.0)  # room tone + 3 dB
    assert R.speech_level([-20.0] * 50 + [-70.0] * 50) == pytest.approx(-20.0)


def test_plan_fix_finds_both_joins():
    r, dur, _, pr, pdur = take_and_patch("104:4", "104:4")
    plan = R.plan_fix(SCRIPT, FIX, r["said"], dur, energy(r["said"], dur), pr["said"], pdur, energy(pr["said"], pdur))
    assert plan["ok"], plan["why"]
    idx = R.fix_items(SCRIPT, "104:4", "104:4")
    a = next(t for t in r["said"] if t["item"] == idx[0])["startS"]
    z = next(t for t in r["said"] if t["item"] == idx[-1])["endS"]
    prev = next(t for t in reversed(r["said"]) if t["item"] < idx[0])["endS"]
    nxt = next(t for t in r["said"] if t["item"] > idx[-1])["startS"]
    (c1, c2), (p0, p1) = plan["take"], plan["patch"]
    assert prev < c1 < a and z < c2 < nxt
    assert 0 < p0 < pr["said"][0]["startS"] and pr["said"][-1]["endS"] < p1 < pdur
    assert pr["said"][0]["startS"] - p0 <= R.JOIN_KEEP_S + 0.02 and c1 - prev <= R.JOIN_KEEP_S + 0.02  # a pause of normal length
    assert plan["gainDb"] == 0.0 and plan["joinsS"] == [c1, round(c1 + p1 - p0, 3)]
    assert plan["durationS"] == pytest.approx(c1 + (p1 - p0) + dur - c2, abs=1e-3)
    j0, j1 = plan["joins"]
    assert j0["before"] == "104:4" and j1["after"] == "104:4" and j0["silenceS"]["take"] >= R.JOIN_SILENCE_S and 0.3 < j0["pauseS"] < 1.0


def test_plan_fix_refusals():
    r, dur, _, pr, pdur = take_and_patch("104:4", "104:4")
    idx = R.fix_items(SCRIPT, "104:4", "104:4")
    a = next(t for t in r["said"] if t["item"] == idx[0])["startS"]
    prev = next(t for t in reversed(r["said"]) if t["item"] < idx[0])["endS"]
    z = next(t for t in r["said"] if t["item"] == idx[-1])["endS"]
    nxt = next(t for t in r["said"] if t["item"] > idx[-1])["startS"]
    pdb = energy(pr["said"], pdur)
    plan = lambda tdb, p=pdb: R.plan_fix(SCRIPT, FIX, r["said"], dur, tdb, pr["said"], pdur, p)
    # the reciter ran 104:3 into 104:4: the gap is only 20 dB down, not silence
    x = plan(energy(r["said"], dur, loud={(prev, a): -40.0}))
    assert not x["ok"] and x["why"] == ["no clean pause before 104:4 in take 1: record again from 104:3"] and x["suggest"] == {"from": "104:3", "to": "104:4"}
    x = plan(energy(r["said"], dur, loud={(prev, a): -40.0, (z, nxt): -40.0}))
    assert len(x["why"]) == 2 and x["suggest"] == {"from": "104:3", "to": "104:5"}
    # the reciter started at once and stopped at once: the patch has no silence at its ends
    p_first, p_last = pr["said"][0]["startS"], pr["said"][-1]["endS"]
    x = plan(energy(r["said"], dur), energy(pr["said"], pdur, loud={(0.0, p_first): -25.0}))
    assert not x["ok"] and "no silence before 104:4 at the start of your fix" in x["why"][0] and x["suggest"] == {"from": "104:4", "to": "104:4"}
    x = plan(energy(r["said"], dur), energy(pr["said"], pdur, loud={(p_last, pdur): -25.0}))
    assert "no silence after 104:4 at the end of your fix" in x["why"][0]
    # level: 4 dB quieter is matched, 10 dB is refused
    x = plan(energy(r["said"], dur), energy(pr["said"], pdur, speech=-24.0))
    assert x["ok"] and x["gainDb"] == 4.0
    x = plan(energy(r["said"], dur), energy(pr["said"], pdur, speech=-30.0, quiet=-80.0))
    assert not x["ok"] and x["why"][0].startswith("your fix is 10.0 dB quieter than take 1") and x["gainDb"] == 10.0
    # a take with no pause anywhere (a noisy room: its room tone is its speech level) has no silence, not silence everywhere
    flat = [-20.0] * int(dur * 100)
    x = R.plan_fix(SCRIPT, {**FIX, "from": "104:1", "to": "104:1"}, r["said"], dur, flat, pr["said"], pdur, pdb)
    assert x["suggest"] is None and x["why"] == ["no clean pause before 104:1 in take 1: record the whole take again",
                                                 "no clean pause after 104:1 in take 1: record again up to 104:2"]
    x = R.plan_fix(SCRIPT, {**FIX, "from": "104:9", "to": "104:9"}, r["said"], dur, flat, pr["said"], pdur, pdb)
    assert x["suggest"] is None and x["why"][1] == "no clean pause after 104:9 in take 1: record the whole take again"


def test_stitch_and_its_timeline():
    np = pytest.importorskip("numpy")
    plan = {"take": [1.0, 2.5], "patch": [0.5, 1.75], "gainDb": 20 * np.log10(2), "joinsS": [1.0, 2.25]}
    take, patch = np.full(16000 * 4, 0.1, dtype=np.float32), np.full(16000 * 2, 0.2, dtype=np.float32)
    y = R.stitch(take, patch, plan)
    h = round(R.JOIN_XFADE_S * 16000 / 2)
    assert len(y) == 16000 + 20000 + (64000 - 40000)
    assert np.allclose(y[: 16000 - h], 0.1) and np.allclose(y[16000 + h : 36000 - h], 0.4) and np.allclose(y[36000 + h :], 0.1)
    assert 0.1 < y[16000] < 0.4 and np.all(np.diff(y[16000 - h : 16000 + h]) >= 0)  # the crossfade rises from take to patch
    with pytest.raises(ValueError, match="out of range"):
        R.stitch(take, patch, {**plan, "patch": [0.5, 1.999]})
    assert R.fix_resumes([0.4, 1.8, 3.1], [0.2, 1.0], plan) == [0.4, 1.0, 1.5, 2.25, 2.85]


def test_fix_takes_keep_their_number_and_record_the_patch():
    assert R.next_take(["take-1.webm", "take-1.json", "take-2.json", "take-2.patch.webm"]) == 3
    fix = {"base": 1, "from": "104:4", "to": "104:4"}
    m = R.take_meta({"speaker": "ali", "scriptSha": SCRIPT["sha"], "resumesS": "[2.5]", "durationS": "9.5"}, SCRIPT, 2, "webm", "t", 10, fix)
    assert m["ext"] == "wav" and m["resumesS"] == [] and m["clientDurationS"] == 9.5
    assert m["fix"] == {**fix, "patch": "take-2.patch.webm", "patchExt": "webm", "resumesS": [2.5]}
    f = R.fix_files("ali", "rec-v1-01", 2, "webm")
    assert f["patch"] == "recordings/ali/rec-v1-01/take-2.patch.webm" and f["patchWav"] == "recordings/sessions/rec-ali-rec-v1-01-t2-patch.phone.wav"
    assert not R.TAKE.match("take-2.patch.webm") and R.take_files("ali", "rec-v1-01", 2, m["ext"])["upload"].endswith("take-2.wav")
    rs = [{"speaker": "a", "script": "s", "take": 1, "verdict": "redo"}, {"speaker": "a", "script": "s", "take": 2, "verdict": "refused"},
          {"speaker": "a", "script": "s", "take": 3, "verdict": "ok"}]
    assert [r["take"] for r in R.current_takes([r for r in rs if r["verdict"] in ("ok", "listen")])] == [3]


def test_composed_take_session_carries_its_joins():
    r = run(SCRIPT["said"])
    meta = {**META, "take": 3, "resumesS": [12.0, 30.5], "joinsS": [12.0, 30.5], "fix": {"base": 2, "from": "104:4", "to": "104:5", "patch": "x"}}
    s = R.session(SCRIPT, r, meta, "x.wav")
    assert s["meta"]["joinsS"] == [12.0, 30.5] and s["meta"]["fix"] == {"base": 2, "from": "104:4", "to": "104:5"} and s["meta"]["resumesS"] == [12.0, 30.5]
    its = R.review_items(s, r, SCRIPT, "w")
    joins = [it for it in its if "-join-" in it["id"]]
    assert len(joins) == 2 and all(it["label"] == {"type": "clean"} and it["audio"]["startS"] < t < it["audio"]["endS"] for it, t in zip(joins, [12.0, 30.5]))
    assert V.validate_queue(R.queue([its])) == []
    assert "joinsS" not in R.session(SCRIPT, r, META, "x.wav")["meta"]

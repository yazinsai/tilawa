# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with numpy --with scipy --with pytest pytest tools/test_lab_sessions.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_sessions as S  # noqa: E402
from lab_review_core import validate_queue  # noqa: E402

REF = S.Ref(json.loads((S.ROOT / "packages/core/assets/ref-v2.json").read_text()))
SR = S.SR


def word(s, a, w):
    x = REF.words[REF.idx[(s, a, w)]]
    return x["cont"], x["iso"], x["text"]


# ---- boundary kinds -------------------------------------------------------------------------------------------------

def test_tail_and_head_kinds():
    assert S.tail_kind(*word(1, 2, 1)) == ("vowel", "د")  # ٱلْحَمْدُ: pause form plus the case vowel
    assert S.tail_kind(*word(112, 1, 1)) == ("closed", "ل")  # قُلْ
    assert S.tail_kind(*word(112, 1, 2)) is None  # هُوَ: the pause form lengthens it
    assert S.head_kind(*word(112, 1, 2)[:2]) == ("firm", "ه")
    assert S.head_kind(*word(2, 2, 2)[:2]) == ("wasl", "ل")  # ٱلْكِتَـٰبُ drops its connecting hamza
    assert S.tail_kind("", "x") is None and S.head_kind("", "x") is None


def test_joins_rules():
    pause, link = {"kind": "pause"}, lambda tail, head: {"kind": "link", "tail": tail, "head": head}
    firm, wasl = ("firm", "ق"), ("wasl", "ل")
    assert S.joins(pause, pause) and S.joins(pause, link(None, firm)) and not S.joins(pause, link(None, wasl))
    assert not S.joins(link(("vowel", "\u064e"), firm), pause)
    assert S.joins(link(("vowel", "\u064e"), firm), link(None, firm))
    assert not S.joins(link(("vowel", "\u064e"), firm), link(None, wasl))  # the tail was said before a firm head
    assert not S.joins(link(("long", "ا"), firm), link(None, ("firm", "ء")))  # madd before hamza lengthens
    assert S.joins(link(("closed", "ن"), ("firm", "ع")), link(None, ("firm", "ح")))  # izhar only before throat letters
    assert not S.joins(link(("closed", "ن"), ("firm", "ع")), link(None, ("firm", "ق")))
    assert not S.joins(link(("closed", "م"), firm), link(None, ("firm", "ب")))  # ikhfa shafawi
    assert not S.joins(link(("closed", "ل"), firm), link(None, ("firm", "ل")))  # idgham of like letters
    assert S.joins(link(("vowel", "\u064e"), firm), pause, restart=True) and not S.joins(pause, link(None, firm), restart=True)


def test_twin_words_are_symmetric_one_word_swaps():
    twins = S.twin_words(REF)
    assert twins
    for a, bs in list(twins.items())[:2000]:
        wa = REF.words[a]
        ka = REF.ayah_words[f"{wa['s']}:{wa['a']}"]
        for b in bs:
            wb = REF.words[b]
            kb = REF.ayah_words[f"{wb['s']}:{wb['a']}"]
            assert a in twins[b] and wa["w"] == wb["w"] and len(ka) == len(kb)
            from help_import import skeleton

            assert sum(skeleton(x) != skeleton(y) for x, y in zip(ka, kb)) == 1


def test_forms_other_ignores_identical_refrains():
    refrain = REF.ayah_words["55:13"]
    assert REF.ayah_words["55:16"] == refrain
    other_word = REF.ayah_words["55:1"][0]
    said = refrain[:1] + [other_word] + refrain[2:]
    from help_import import forms_other_ayah

    assert forms_other_ayah(REF.ayah_words, ["55:16"], said)  # the raw rule counts the identical 55:13 as another ayah
    assert not S.forms_other(REF, ["55:16"], said)
    twins = S.twin_words(REF)
    a = next(i for i, bs in twins.items() if REF.words[i]["s"] == 26)
    w, b = REF.words[a], REF.words[twins[a][0]]
    host = REF.ayah_words[f"{w['s']}:{w['a']}"]
    swapped = host[: w["w"] - 1] + [b["text"]] + host[w["w"]:]
    assert S.forms_other(REF, [f"{w['s']}:{w['a']}"], swapped)


# ---- a synthetic recording of Al-Ikhlas, every word a tone -------------------------------------------------------

def ikhlas(inner_gap=0.35, inner_db=-20.0, ayah_extra=None, refuse=None):
    t, words, ayahs = 1.0, [], []
    for a in range(1, 5):
        n = len(REF.ayah_words[f"112:{a}"])
        ayahs.append({"ayah": a, "excluded": False, "levelDb": -20.0, **(ayah_extra or {}).get(a, {})})
        for i in range(1, n + 1):
            words.append({"surah": 112, "ayah": a, "word": i, "startS": round(t, 3), "endS": round(t + 0.5, 3), "score": 0.95})
            gap = 0.8 if i == n else inner_gap
            words[-1]["cutAfterS"], words[-1]["cutDb"] = round(t + 0.5 + gap / 2, 3), -20.0 if i == n else inner_db
            t += 0.5 + gap
    words[-1]["cutAfterS"] = None
    dur = t + 2
    align = {"readId": 7, "surah": 112, "split": "practice", "reciter": "Test", "audio": "sources/test/112.wav", "durationS": dur,
             "basmala": {"present": False}, "ayahs": ayahs, "words": words}
    x = np.zeros(int(dur * SR), dtype=np.float32)
    for k, w in enumerate(words):
        a, b = round(w["startS"] * SR), round(w["endS"] * SR)
        x[a:b] = 0.3 * np.sin(2 * np.pi * freq(k) * np.arange(b - a) / SR)
    return S.Rec(align, REF, refuse), x


def freq(k):
    return 300.0 + 125.0 * k


def tone_at(y, a, b):
    seg = y[round(a * SR): round(b * SR)].astype(np.float64)
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=1 << 16))
    return np.argmax(spec) * SR / (1 << 16)


def build(edits=None):
    rec, x = ikhlas()
    b = lambda k: rec.boundary(k, REF)
    if edits is None:
        edits = [
            {"cls": "repeat", "k0": 2, "k1": 3, "seam": (b(3), b(1))},
            {"cls": "omitted_word", "k0": 7, "k1": 7, "seam": (b(6), b(7)), "remove": [7], "other": False},
            {"cls": "wrong_word", "k0": 12, "k1": 12, "seam": (b(11), b(12)), "remove": [12], "donor": (rec, 1, b(0), b(1)), "other": False},
        ]
    last = len(rec.ayahs) - 1
    win = {"id": "lab-v1-t01", "rec": rec.key, "split": "practice", "kind": "ayah1", "dis": 0, "startAyah": 1, "endAyah": 4, "first": 0, "last": last,
           "startS": S.start_cut(rec, 0), "endS": S.end_cut(rec, last), "clean": False}
    toks = S.tokens_for(win, rec, edits)
    recs = {rec.key: rec}
    pieces = S.pieces_of(toks, recs, win["startS"], win["endS"], rec)
    s = S.label(win, rec, edits, toks, pieces, recs, REF, {"snrDb": 15.0})
    return rec, x, pieces, s


def test_boundaries_on_synthetic_recording():
    rec, _ = ikhlas()
    assert all(rec.boundary(k, REF)["kind"] == "pause" for k in range(len(rec.words) - 1))
    assert rec.boundary(len(rec.words) - 1, REF) is None
    assert S.start_cut(rec, 0) == pytest.approx(0.85)
    assert S.end_cut(rec, 3) == pytest.approx(rec.words[-1]["endS"] + 0.6)


def test_labels_validate_and_map_into_the_output_timeline():
    rec, x, pieces, s = build()
    assert S.validate(s, REF) == []
    y = S.render(pieces, lambda audio: x)
    assert len(y) == sum(p["len"] for p in pieces) == round(s["durationS"] * SR)
    for t in s["said"]:
        k = rec.pos[(t["from"]["ayah"], t["from"]["word"])]
        assert tone_at(y, t["startS"] + 0.05, t["endS"] - 0.05) == pytest.approx(freq(k), abs=2), t
        src = rec.words[k]
        assert t["endS"] - t["startS"] == pytest.approx(src["endS"] - src["startS"], abs=1e-3)
    ats = [(w["at"]["ayah"], w["at"]["word"]) for w in s["words"]]
    assert (3, 2) not in ats  # omitted words are absent
    assert ats.count((1, 3)) == 2 and ats.count((1, 4)) == 2  # the repeat says them twice
    wrong = next(w for w in s["words"] if "said" in w)
    assert wrong["at"] == {"surah": 112, "ayah": 4, "word": 3} and wrong["said"]["from"] == {"surah": 112, "ayah": 1, "word": 2}
    om = next(e for e in s["events"] if e["errorClass"] == "omitted_word")
    before = next(w for w in s["words"] if w["at"]["ayah"] == 3 and w["at"]["word"] == 1)
    after = next(w for w in s["words"] if w["at"]["ayah"] == 3 and w["at"]["word"] == 3)
    assert om["location"] == {"surah": 112, "ayah": 3, "word": 2} and om["startS"] == before["endS"] and om["endS"] == after["startS"]
    assert [n["kind"] for n in s["nonErrors"]] == ["repeat", "repeat"] and {sp["kind"] for sp in s["spans"]} == {"recitation", "repeat"}
    assert s["meta"]["counts"] == {"omitted_word": 1, "repeat": 1, "wrong_word": 1}
    sp = {x["type"]: x for x in s["splices"]}
    assert sp["omitted_word"]["removed"] == ["112:3:2"] and len(sp["omitted_word"]["seams"]) == 1
    seam = sp["wrong_word"]["seams"][0]
    assert seam["xfadeS"] == 0.01 and seam["out"]["boundary"] == "pause" and seam["in"]["srcS"] == pytest.approx(rec.words[0]["cutAfterS"])


def test_render_crossfade_is_equal_gain_and_pieces_are_exact():
    x = np.full(SR * 4, 0.5, dtype=np.float32)
    pieces = [{"audio": "a", "src": [8000, 24000], "out": 0, "len": 16000, "gainDb": 0.0},
              {"audio": "a", "src": [32000, 40000], "out": 16000, "len": 8000, "gainDb": 0.0}]
    y = S.render(pieces, lambda a: x)
    assert len(y) == 24000
    assert np.allclose(y[S.EDGE_FADE: -S.EDGE_FADE], 0.5, atol=1e-6)  # constant through the seam
    assert y[0] == 0 and abs(y[-1]) < 0.01


def test_degrade_keeps_length_and_timing():
    rng = np.random.default_rng(0)
    x = np.zeros(SR * 3, dtype=np.float32)
    x[SR: SR + 800] = rng.standard_normal(800) * 0.3
    rir = np.zeros(4000)
    rir[100], rir[300:4000] = 1.0, rng.standard_normal(3700) * 0.02 * np.exp(-np.arange(3700) / 800)
    noise = [rng.standard_normal(SR) * 0.1 for _ in range(4)]
    a, b = S.degrade(x, rir, noise, 20.0), S.degrade(x, rir, noise, 20.0)
    assert len(a) == len(x) and np.array_equal(a, b) and np.abs(a).max() <= 0.89 + 1e-6
    cc = np.correlate(a[SR - 200: SR + 1000].astype(np.float64), x[SR: SR + 800].astype(np.float64), "valid")
    assert int(np.argmax(cc)) == 200  # no delay: the RIR starts at its direct path


def test_pcm16_is_deterministic():
    x = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
    a = S.pcm16(x)
    assert a == S.pcm16(x) and a[:4] == b"RIFF" and len(a) == 44 + 32000


def test_validate_catches_overlaps():
    _, _, _, s = build()
    s = json.loads(json.dumps(s))
    s["words"][3]["startS"] = s["words"][2]["startS"]
    assert S.validate(s, REF)


def test_phone_params_and_build_are_deterministic():
    bank = {"rirs": [{"name": f"r{i}"} for i in range(10)], "noises": [{"name": f"n{i}", "durationS": 7.0} for i in range(20)]}
    a = S.phone_params("lab-v1-p01", 200.0, bank)
    assert a == S.phone_params("lab-v1-p01", 200.0, bank) and a != S.phone_params("lab-v1-p02", 200.0, bank)
    assert sum(bank["noises"][i]["durationS"] for i in a["noise"]) >= 201 and S.SNR_DB[0] <= a["snrDb"] <= S.SNR_DB[1]
    assert json.dumps(build()[3]) == json.dumps(build()[3])


def test_said_form_takes_the_pause_form_where_the_reciter_stopped():
    w = lambda s, a, i: REF.words[REF.idx[(s, a, i)]]
    q = w(41, 15, 12)  # قُوَّةً, mid-ayah
    assert S.said_form(q, False, False) == "قُووَتَن" and S.said_form(q, False, True) == "قُووَه"
    h = w(41, 15, 7)  # ٱلْحَقِّ: connecting hamza, said only when starting after a stop
    assert S.said_form(h, False, True) == "لحَققڇ"
    assert S.said_form(h, True, False) == "ءَلحَققِ" and S.said_form(h, True, True) == h["iso"]
    last, first = w(41, 15, 21), w(41, 15, 1)  # ayah edges are pause forms in the continuous phonemes already
    assert S.said_form(last, False, True) == last["cont"] and S.said_form(first, True, False) == first["cont"]
    changed = next(x for x in REF.words if 1 < x["w"] < x["n"] and S.head_kind(x["cont"], x["iso"]) is None)
    assert S.said_form(changed, False, False) == changed["cont"] and S.said_form(changed, False, True) is None


def test_labels_use_pause_forms_at_source_pauses():
    _, _, _, s = build()  # every inner gap is a pause here
    wrong = next(e for e in s["events"] if e["errorClass"] == "wrong_word")
    huwa = REF.words[REF.idx[(112, 1, 2)]]
    assert wrong["said"] == huwa["iso"] != huwa["cont"]
    assert next(t for t in s["said"] if t["role"] == "wrong")["phonemes"] == huwa["iso"]
    rec, _ = ikhlas(inner_gap=0.05, inner_db=-10.0)  # linked inside ayahs: continuous forms
    assert not rec.pause_after(1) and rec.pause_after(3) and rec.said(1, REF) == huwa["cont"]


def test_long_or_loose_ayahs_and_check_failures_are_refused():
    assert S.refused_ayah({"excluded": False, "lenRatio": 1.06, "per": 0.0}) == "long"
    assert S.refused_ayah({"excluded": False, "lenRatio": 1.0, "per": 0.12}) == "per"
    assert S.refused_ayah({"excluded": False, "lenRatio": 1.04, "per": 0.1}) is None
    assert S.refused_ayah({"excluded": True, "lenRatio": 2.0, "per": 0.5}) is None
    rec, _ = ikhlas(ayah_extra={2: {"lenRatio": 1.08}}, refuse={3})
    assert rec.refused == {2: "long", 3: "check"}
    assert S.runs(rec) == [[0], [3]]  # windows never span a refused ayah
    assert all(rec.boundary(k, REF) is None for k, w in enumerate(rec.words) if w["ayah"] in (2, 3))


def test_linked_cuts_refuse_weak_onsets():
    rec, _ = ikhlas(inner_gap=0.05, inner_db=-10.0)
    assert REF.words[rec.words[1]["ref"]]["cont"][0] == "ه"
    assert rec.boundary(0, REF) is None  # قُلْ | هُوَ: closed l before a soft h
    k = rec.pos[(3, 1)]
    assert rec.boundary(k, REF)["kind"] == "link"  # لَمْ | يَلِدْ
    paused, _ = ikhlas()
    assert paused.boundary(0, REF)["kind"] == "pause"  # the same head is fine after a pause


def test_donors_skip_refused_words():
    rec, _ = ikhlas()
    all_ = S.Donors([rec], REF)
    some = S.Donors([rec], REF, {(rec.key, 1, 2)})
    assert any(k == 1 for _, k, _, _ in all_.items) and not any(k == 1 for _, k, _, _ in some.items)
    assert len(some.items) == len(all_.items) - 1


def test_refusals_from_checks_are_cumulative_and_in_source_terms():
    _, _, _, s = build()
    sid = s["id"]
    wrong = next(x for x in s["splices"] if x["type"] == "wrong_word")
    checks = {"testSet": "t", "failed": [{"id": f"{sid}-{wrong['id']}", "session": sid, "splice": wrong["id"], "reasons": ["local PER 0.3"]}],
              "failedClean": [{"id": f"{sid}-clean-0", "session": sid, "from": "112:1:1", "to": "112:2:2", "reasons": ["PER 0.09"]}],
              "failedTiming": [{"session": sid, "off": [{"at": "112:4", "startS": 9.0, "dStartS": 0.0, "dEndS": 0.5}, {"at": "added", "startS": 3.0, "dStartS": 1, "dEndS": 0}]}]}
    doc, new = S.refusals_from_checks(checks, {sid: s})
    assert doc["sites"] == [{"rec": "7/112", "ayah": 4, "word": 3, "class": "wrong_word", "reason": "local PER 0.3", "from": f"{sid}-{wrong['id']}", "iteration": 1}]
    assert [(d["rec"], d["ayah"], d["word"]) for d in doc["donors"]] == [("7/112", 1, 2)]
    assert [x["ayah"] for x in doc["ayahs"]] == [1, 2, 4] and len(doc["unmapped"]) == 1 and len(new) == 6
    again, new2 = S.refusals_from_checks(checks, {sid: s}, doc, iteration=2)
    assert new2 == [] and again == doc


def test_review_queue_puts_linked_splices_first_with_plain_titles():
    seam = lambda a, b: {"out": {"boundary": a}, "in": {"boundary": b}}
    assert S.linked({"seams": [seam("pause", "link")]}) and not S.linked({"seams": [seam("pause", "pause")]})
    _, _, _, s = build()
    q = S.review_queue([s], REF)
    titles = {i["title"] for i in q["items"]}
    assert "A wrong word put in (joined at a pause)" in titles and q["testSet"] == s["meta"]["testSet"]


def test_manifest_and_review_queue():
    _, _, _, s = build()
    m = S.manifest([s])
    e = m["sessions"][0]
    assert m["version"] == 1 and e["session"]["audio"] == "sessions/lab-v1-t01.phone.wav" and "said" not in e["session"]
    assert e["evidence"] == "sessions/lab-v1-t01.{listener}.phone.evidence.json"
    q = S.review_queue([s], REF)
    assert validate_queue(q) == [] and len({i["id"] for i in q["items"]}) == len(q["items"])
    assert {i["kind"] for i in q["items"]} == {"splice", "case"}


def test_review_clips_play_every_word_they_show():
    _, _, _, s = build()
    for i in S.review_queue([s], REF)["items"]:
        a, b = i["audio"]["startS"], i["audio"]["endS"]
        shown = [w for w in s["said"] if w["endS"] > a and w["startS"] < b]
        assert shown and all(a <= w["startS"] and w["endS"] <= b for w in shown), i["id"]
        assert i["text"] == " ".join(w["text"] for w in shown)

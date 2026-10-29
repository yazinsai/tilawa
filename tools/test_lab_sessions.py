# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with numpy --with scipy --with pytest pytest tools/test_lab_sessions.py"""

import json
from collections import Counter
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
    assert S.joins(pause, pause, donor=True) and not S.joins(pause, link(None, firm), donor=True)  # a pause-form donor lands on a pause
    assert S.joins(link(("vowel", "\u064e"), firm), link(None, firm), donor=True)


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

def ikhlas(inner_gap=0.35, inner_db=-20.0, ayah_extra=None, refuse=None, energy=True, noise=None, cut_db=None, surah=112):
    """noise: {word k: (from s, to s) after its end} filled with low noise; cut_db: {word k: cutDb} overrides."""
    t, words, ayahs = 1.0, [], []
    for a in range(1, 1 + sum(k.startswith(f"{surah}:") for k in REF.ayah_words)):
        n = len(REF.ayah_words[f"{surah}:{a}"])
        ayahs.append({"ayah": a, "excluded": False, "levelDb": -20.0, **(ayah_extra or {}).get(a, {})})
        for i in range(1, n + 1):
            words.append({"surah": surah, "ayah": a, "word": i, "startS": round(t, 3), "endS": round(t + 0.5, 3), "score": 0.95})
            gap = 0.8 if i == n else inner_gap
            words[-1]["cutAfterS"], words[-1]["cutDb"] = round(t + 0.5 + gap / 2, 3), -20.0 if i == n else inner_db
            t += 0.5 + gap
    words[-1]["cutAfterS"] = None
    for k, db in (cut_db or {}).items():
        words[k]["cutDb"] = db
    dur = t + 2
    align = {"readId": 7, "surah": surah, "split": "practice", "reciter": "Test", "audio": f"sources/mp3quran/7/{surah}.wav", "durationS": dur,
             "basmala": {"present": False}, "ayahs": ayahs, "words": words}
    x = np.zeros(int(dur * SR), dtype=np.float32)
    for k, w in enumerate(words):
        a, b = round(w["startS"] * SR), round(w["endS"] * SR)
        x[a:b] = 0.3 * np.sin(2 * np.pi * freq(k) * np.arange(b - a) / SR)
    rng = np.random.default_rng(1)
    for k, (a, b) in (noise or {}).items():
        i, j = round((words[k]["endS"] + a) * SR), round((words[k]["endS"] + b) * SR)
        x[i:j] = 0.01 * rng.standard_normal(j - i)
    return S.Rec(align, REF, refuse, S.frame_db(x) if energy else None), x


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


def test_linked_cuts_need_depth_and_more_after_a_vowel():
    closed, vowel = (3, 1), (2, 1)  # لَمْ | يَلِدْ ends closed; ٱللَّهُ | ٱلصَّمَدُ ends in its case vowel
    for db, closed_ok, vowel_ok in ((-5.0, False, False), (-7.0, True, False), (-9.0, True, True)):
        rec, _ = ikhlas(inner_gap=0.05, inner_db=db)
        bc, bv = rec.boundary(rec.pos[closed], REF), rec.boundary(rec.pos[vowel], REF)
        assert (bc is not None, bv is not None) == (closed_ok, vowel_ok), db
        assert bv is None or bv["tail"][0] == "vowel"
    assert S.link_limit(("long", "ا")) == S.link_limit("vowel") == S.LINK_TAIL_DB < S.link_limit(("closed", "م")) == S.LINK_DB


def test_pause_cuts_before_weak_onsets_need_measured_silence():
    rec, _ = ikhlas()  # silent gaps: every pause is verified as it is
    assert REF.words[rec.words[1]["ref"]]["iso"][0] == "ه"  # قُلْ | هُوَ
    assert not rec.boundary(0, REF).get("weak") and "cutMovedS" not in rec.words[0] and S.start_cut(rec, 1) is not None
    bare, _ = ikhlas(energy=False)  # nothing measured: every pause before a weak onset is unverified
    b = bare.boundary(0, REF)
    assert b["kind"] == "pause" and b["weak"] and not S.joins({"kind": "pause"}, b) and not S.joins(b, b, restart=True)
    assert S.joins(b, bare.boundary(bare.pos[(3, 1)], REF))  # a weak boundary still ends a piece; يَلِدْ starts firm
    assert S.start_cut(bare, 1) is None  # ayah 2 starts with ٱللَّهُ, a hamza after the stop
    moved, _ = ikhlas(noise={0: (0.165, 0.35)})  # sound from 10 ms before the cut to the next word: the cut moves into the silence
    assert moved.words[0]["cutMovedS"] == pytest.approx(0.03) and moved.words[0]["cutAfterS"] == pytest.approx(1.645)
    assert not moved.boundary(0, REF).get("weak") and moved.boundary(0, REF)["cut"] == pytest.approx(1.645)
    loud, _ = ikhlas(noise={0: (0.0, 0.35)})  # sound over the whole gap
    assert loud.words[0].get("weakOnset") and loud.boundary(0, REF)["weak"]


def test_silence_floor_follows_the_room():
    quiet = [-20.0] * 10 + [-60.0] * 10 + [-20.0] * 80
    assert S.onset_floor(S.room_tone(quiet), -20.0) == S.ONSET_FLOOR_DB  # a quiet room: 30 dB below speech
    assert S.silence_before(quiet, 0.19, -20.0, -30.0) == pytest.approx(0.09)
    assert S.silence_before(quiet, 0.20, -20.0, -30.0) == 0.0  # the crossfade after the cut reaches the sound
    assert S.silent_cut(quiet, 0.0, 0.25, -20.0) == pytest.approx(0.19) and S.silent_cut(quiet, 0.0, 0.17, -20.0) is None
    reverb = [-20.0] * 10 + [-34.0] * 10 + [-20.0] * 80  # room tone only 14 dB below speech
    floor = S.onset_floor(S.room_tone(reverb), -20.0)
    assert floor == pytest.approx(-34.0 + S.ROOM_MARGIN_DB + 20.0) and S.silence_before(reverb, 0.19, -20.0, floor) == pytest.approx(0.09)


def test_short_donors_are_refused():
    rec, _ = ikhlas()
    assert 1 in {k for _, k, _, _ in S.Donors([rec], REF).items}
    rec.words[1]["endS"] = rec.words[1]["startS"] + 0.24
    assert 1 not in {k for _, k, _, _ in S.Donors([rec], REF).items}


def gate_checks(s, **seam):
    base = {"beforeDb": -20.0, "afterDb": -20.0, "srcBeforeDb": -20.0, "srcAfterDb": -20.0, "clickRatio": 1.0, "seamDb": -20.0,
            "outCutDb": -40.0, "inCutDb": -40.0, "inSilenceS": 0.3}
    n = len(s["said"])
    return {"seams": [{**base, "outS": m["outS"], **seam} for x in s["splices"] for m in x["seams"]],
            "words": {"words": n, "found": n, "clipped": 0, "worst": []}}


def test_seam_gate_measures_link_depth_onsets_and_donor_length():
    lab = {"out": {"boundary": "link", "cutDb": -9.0, "tail": "vowel"}, "in": {"boundary": "pause", "cutDb": -30.0}}
    ok = {"beforeDb": -20, "afterDb": -20, "srcBeforeDb": -20, "srcAfterDb": -20, "clickRatio": 1.0, "seamDb": -20, "outCutDb": -9.0,
          "inCutDb": -30.0, "inSilenceS": 0.1}
    assert S.seam_failures(ok, label=lab, head="ءَللَااه") == []
    assert S.seam_failures({**ok, "outCutDb": -7.0}, label=lab)[0].startswith("linked cut before the seam at -7.0 dB (vowel tail")
    assert S.seam_failures({**ok, "outCutDb": -7.0}, label={**lab, "out": {**lab["out"], "tail": "closed"}}) == []
    assert S.seam_failures({**ok, "inSilenceS": 0.05}, label=lab, head="ءَللَااه")[0].startswith("weak onset ء")
    assert S.seam_failures({**ok, "inSilenceS": 0.05}, label=lab, head="يَلِدڇ") == []
    _, _, _, s = build()
    g = S.seam_gate(s, gate_checks(s))
    assert g["failed"] == [] and g["donors"] == 1 and g["donorsShort"] == 0
    first = s["splices"][0]["seams"][0]["outS"]
    g = S.seam_gate(s, {**gate_checks(s), "seams": [{**m, "inSilenceS": 0.0} if m["outS"] == first else m for m in gate_checks(s)["seams"]]})
    head = next(t["phonemes"] for t in sorted(s["said"], key=lambda t: t["startS"]) if t["startS"] >= first)
    assert head[0] == "ء" and [f["splice"] for f in g["failed"]] == [s["splices"][0]["id"]]  # the repeat restarts at ٱللَّهُ
    short = json.loads(json.dumps(s))
    t = next(t for t in short["said"] if t["role"] == "wrong")
    t["endS"] = round(t["startS"] + 0.2, 4)
    g = S.seam_gate(short, gate_checks(short))
    wrong = next(x["id"] for x in s["splices"] if x["type"] == "wrong_word")
    assert [(f["splice"], f["reasons"][0][:10]) for f in g["failed"]] == [(wrong, "wrong word")] and g["donorsShort"] == 1


def test_rule_audit_names_the_rules_an_old_splice_breaks():
    rec, _, _, s = build()
    assert all(not r["refused"] and not r["moved"] for r in S.rule_audit(s, {rec.key: rec}, REF).values())
    bare, _ = ikhlas(energy=False)
    got = S.rule_audit(s, {bare.key: bare}, REF)
    rep = next(x["id"] for x in s["splices"] if x["type"] == "repeat")
    assert "onset" in got[rep]["refused"]  # the repeat starts again at ٱللَّهُ after a pause, unverified without energy
    wrong = next(x for x in s["splices"] if x["type"] == "wrong_word")
    wrong["donor"]["srcEndS"] = wrong["donor"]["srcStartS"] + 0.2
    assert S.rule_audit(s, {rec.key: rec}, REF)[wrong["id"]]["refused"] == {"donorShort": ["112:1:2 0.20 s"]}


def test_pause_form_donors_only_land_on_a_host_pause():
    import random

    rec, _ = ikhlas(inner_gap=0.05, inner_db=-12.0, surah=107)  # in-ayah joins linked, ayah ends paused
    d = S.Donors([rec], REF)
    links = [b for k in range(len(rec.words) - 1) if (b := rec.boundary(k, REF)) and b["kind"] == "link"]
    old = [(b, it) for b in links for it in d.items if S.joins(b, it[2]) and S.joins(it[3], b) and it[3]["kind"] == "pause"]
    assert old  # ayah-final donors (pause after them) the old rule put into linked host speech
    got = [x for b in links for seed in range(20) if (x := d.find(b, b, rec, 0.0, (1e6, 1e6), lambda r, k: True, random.Random(seed)))]
    assert got and all(x[3]["kind"] == "link" for x in got)
    pause = next(b for k in range(len(rec.words) - 1) if (b := rec.boundary(k, REF)) and b["kind"] == "pause")
    x = d.find(pause, pause, rec, 0.0, (1e6, 1e6), lambda r, k: True, random.Random(1))
    assert x and x[3]["kind"] == "pause"  # at a host pause a pause-form donor is what fits
    rec, _ = ikhlas(inner_gap=0.05, inner_db=-12.0, surah=114)
    k = rec.pos[(3, 1)]
    spans = S.Donors([rec], REF).twin_spans(rec, k, S.twin_words(REF)[rec.words[k]["ref"]])
    assert spans and all(x["seam"][2]["kind"] == x["seam"][3]["kind"] or x["seam"][2]["kind"] == "link" for x in spans)


def test_donors_come_from_the_host_recording_only():
    import random

    host, _ = ikhlas()
    other, _ = ikhlas(surah=114)
    d = S.Donors([host, other], REF)
    assert {r.key for r, *_ in d.items} == {host.key, other.key}
    b = host.boundary(5, REF)
    got = {d.find(b, b, host, 0.0, (1e6, 1e6), lambda r, k: True, random.Random(seed))[0].key for seed in range(20)}
    assert got == {host.key}
    twins = S.twin_words(REF)
    deep, _ = ikhlas(inner_gap=0.05, inner_db=-12.0, surah=114)
    k = deep.pos[(3, 1)]
    assert S.Donors([deep], REF).twin_spans(other, k, twins[deep.words[k]["ref"]]) == []  # the twin is in another recording


def test_repeats_and_restarts_break_off_at_a_pause():
    for gap, db in ((0.05, -12.0), (0.35, -20.0)):
        rec, _ = ikhlas(inner_gap=gap, inner_db=db, surah=87)
        last = len(rec.ayahs) - 1
        for n in range(6):
            win = {"id": f"lab-v1-t{n}", "rec": rec.key, "split": "practice", "kind": "normal", "dis": 0, "startAyah": 1, "endAyah": last + 1,
                   "first": 0, "last": last, "startS": S.start_cut(rec, 0), "endS": S.end_cut(rec, last), "clean": True}
            edits = S.plan_edits(win, {rec.key: rec}, REF, S.Donors([rec], REF), {}, [], seed=n)
            assert len(edits) == 1 and edits[0]["cls"] in ("repeat", "restart")
            e = edits[0]
            assert e["seam"][0]["kind"] == "pause" and rec.pause_after(e["k1"])
            assert e["seam"][1]["kind"] == "pause" and rec.pause_after(e["k0"] - 1)  # and goes back to one
            if gap < 0.3:  # nowhere to stop inside an ayah: the whole ayah said twice
                assert e["cls"] == "repeat" and rec.words[e["k1"] + 1]["ayah"] != rec.words[e["k1"]]["ayah"]
                assert rec.words[e["k0"]]["word"] == 1


def test_added_words_and_skips_are_capped_at_a_share_of_the_mistakes(monkeypatch):
    monkeypatch.setattr(S, "forms_other", lambda *a: False)  # a scan of every ayah per site, not what this tests
    total = Counter()
    for surah in (78, 87, 89):
        rec, _ = ikhlas(surah=surah)
        last = len(rec.ayahs) - 1
        for n in range(4):
            win = {"id": f"lab-v1-c{surah}-{n}", "rec": rec.key, "split": "practice", "kind": "normal", "dis": 0, "startAyah": 1,
                   "endAyah": last + 1, "first": 0, "last": last, "startS": S.start_cut(rec, 0), "endS": S.end_cut(rec, last), "clean": False}
            got = Counter(e["cls"] for e in S.plan_edits(win, {rec.key: rec}, REF, S.Donors([rec], REF), {}, [], seed=n)
                          if e["cls"] not in ("repeat", "restart"))
            for c, share in S.CAPPED.items():
                assert got[c] <= int(share * sum(got.values())) + 1
            total += got
    assert all(0 < total[c] <= 0.33 * sum(total.values()) for c in S.CAPPED) and total["wrong_word"] + total["omitted_word"] > 0


def test_seam_gate_holds_the_lab_v1_2_rules():
    def splice(kind, seams, donor=None):
        side = lambda audio, name, kind, piece, which: {"audio": audio, "boundary": kind, "piece": piece, ("after" if which == "out" else "before"): name, "cutDb": -20.0}
        return {"id": "x1", "type": kind, "host": {"audio": "h.wav"}, "donor": donor,
                "seams": [{"outS": 10.0 + i, "out": side(*o, "out"), "in": side(*n, "in")} for i, (o, n) in enumerate(seams)]}

    swap = splice("wrong_word", [(("h.wav", "1:1:3", "link", "host"), ("h.wav", "1:5:4", "link", "donor")),
                                 (("h.wav", "1:5:4", "pause", "donor"), ("h.wav", "1:1:4", "link", "host"))]) | {"mutashabihat": True}
    assert S.splice_rule_failures(swap) == ["pause-form donor 1:5:4 runs on into 1:1:4, which the host says linked"]
    swap["seams"][1]["in"]["boundary"] = "pause"
    assert S.splice_rule_failures(swap) == []  # a twin span keeps the lab-v1.3 rule: linked seams allowed
    # re-read from the source alignment: 1:5:4 ends its ayah, 1:1:3 | 1:1:4 is a short gap (linked)
    words = {(1, 1, w): {"startS": w * 1.0, "endS": w * 1.0 + 0.9, "cutDb": -20.0} for w in range(1, 6)} | {(1, 5, w): {"startS": 50.0 + w, "endS": 50.9 + w, "cutDb": -20.0} for w in range(1, 5)}
    assert S.splice_rule_failures(swap, lambda audio: words)[0].startswith("pause-form donor")
    # inserted audio (a wrong or added word): every seam a pause on both sides, label or source
    added = splice("added_word", [(("h.wav", "1:1:3", "pause", "host"), ("h.wav", "1:5:4", "link", "donor")),
                                  (("h.wav", "1:5:4", "pause", "donor"), ("h.wav", "1:1:4", "pause", "host"))])
    assert S.splice_rule_failures(added) == ["inserted audio joins 1:1:3 (pause) to 1:5:4 (link), not pause to pause"]
    assert S.rule_of(S.splice_rule_failures(added)[0]) == "insertLinked"
    added["seams"][0]["in"]["boundary"] = "pause"
    assert S.splice_rule_failures(added) == []
    assert S.splice_rule_failures(added, lambda audio: words) == ["inserted audio joins 1:1:3 (link) to 1:5:4 (link), not pause to pause",
                                                                  "inserted audio joins 1:5:4 (pause) to 1:1:4 (link), not pause to pause"]
    words[(1, 1, 4)]["startS"] = words[(1, 5, 4)]["startS"] = 60.0  # a pause before 1:1:4 in the host and before 1:5:4 in the donor
    assert S.splice_rule_failures(added, lambda audio: words) == []
    other = splice("wrong_word", [(("h.wav", "1:1:2", "pause", "host"), ("o.wav", "2:1:1", "pause", "donor")),
                                  (("o.wav", "2:1:1", "pause", "donor"), ("h.wav", "1:1:4", "pause", "host"))])
    assert S.splice_rule_failures(other) == ["piece from another recording (o.wav)"]
    restart = splice("restart", [(("h.wav", "1:1:3", "link", "host"), ("h.wav", "1:1:1", "pause", "host"))])
    assert S.splice_rule_failures(restart) == ["restart breaks off after 1:1:3 mid-phrase, not at a pause"]
    assert S.rule_of(S.splice_rule_failures(restart)[0]) == "breakLinked" and S.rule_of(S.splice_rule_failures(other)[0]) == "otherRecording"
    _, _, _, s = build()
    wrong = next(x for x in s["splices"] if x["type"] == "wrong_word")
    assert [(m["out"]["piece"], m["in"]["piece"]) for m in wrong["seams"]] == [("host", "donor"), ("donor", "host")]
    rep = next(x for x in s["splices"] if x["type"] == "repeat")
    assert rep["seams"][0]["out"]["piece"] == rep["seams"][0]["in"]["piece"] == "host"
    bad = json.loads(json.dumps(s))
    r = next(x for x in bad["splices"] if x["type"] == "restart" or x["type"] == "repeat")
    r["seams"][0]["out"]["boundary"] = "link"
    g = S.seam_gate(bad, gate_checks(bad))
    assert [f["splice"] for f in g["failed"]] == [r["id"]] and g["spliceRules"] == {r["id"]: ["repeat breaks off after 112:1:4 mid-phrase, not at a pause"]}
    assert S.seam_gate_summary([g], "t")["summary"]["spliceRules"] == {"pauseForm": 0, "otherRecording": 0, "breakLinked": 1, "insertLinked": 0, "reenterLinked": 0}
    back = json.loads(json.dumps(s))
    r = next(x for x in back["splices"] if x["type"] == "repeat")
    r["seams"][0]["in"]["boundary"] = "link"
    assert S.splice_rule_failures(r) == [f"repeat re-enters before {r['seams'][0]['in']['before']} mid-phrase, not at a pause"]
    assert S.rule_of(S.splice_rule_failures(r)[0]) == "reenterLinked"


def test_rule_audit_names_the_lab_v1_2_rules():
    rec, _, _, s = build()
    wrong = next(x for x in s["splices"] if x["type"] == "wrong_word")
    linked, _ = ikhlas(cut_db={12: -10.0})  # the host runs on from 112:4:3 into 112:4:4, where the donor's out seam lands
    got = S.rule_audit(s, {rec.key: linked}, REF)[wrong["id"]]["refused"]
    assert got == {"insertLinked": ["112:1:2 (pause) to 112:4:4 (link)"]}
    wrong["mutashabihat"] = True  # a twin span keeps the lab-v1.3 rule
    got = S.rule_audit(s, {rec.key: linked}, REF)[wrong["id"]]["refused"]
    assert list(got) == ["pauseForm"] and got["pauseForm"][0].startswith("112:1:2 said before a pause")
    other, _ = ikhlas(surah=114)
    moved = json.loads(json.dumps(s))
    w = next(x for x in moved["splices"] if x["type"] == "wrong_word")
    w["seams"][0]["in"]["audio"] = w["seams"][1]["out"]["audio"] = w["donor"]["audio"] = other.audio
    assert "otherRecording" in S.rule_audit(moved, {rec.key: rec, other.key: other}, REF)[w["id"]]["refused"]


def window_of(rec, sid="lab-v1-t03"):
    last = len(rec.ayahs) - 1
    return {"id": sid, "rec": rec.key, "split": "practice", "kind": "normal", "dis": 0, "startAyah": 1, "endAyah": last + 1, "first": 0, "last": last,
            "startS": S.start_cut(rec, 0), "endS": S.end_cut(rec, last), "clean": False}


def pause_after(rec, k):
    """Turns the linked join after word k into an in-ayah pause."""
    w, n = rec.words[k], rec.words[k + 1]
    n["startS"], w["cutDb"] = round(w["endS"] + 0.35, 3), -20.0
    rec._b.clear()


def test_wrong_words_run_pause_to_pause_in_host_and_donor():
    rec, _ = ikhlas(inner_gap=0.05, inner_db=-12.0, surah=114)  # every join inside an ayah linked (cuttable, but not a pause)
    k = rec.pos[(3, 1)]  # إِلَـٰهِ ٱلنَّاسِ
    got = S.Donors([rec], REF).pause_spans(rec, k, "wrong_word", lambda r, kk: True)
    # مَلِكِ ٱلنَّاسِ in for it, on to the ayah end: the only donor stretch that reads as the host's but for one word
    assert [(x["a"], x["b"], x["dk"], x["span"][2:]) for x in got] == [(k, k + 1, rec.pos[(2, 1)], (rec.pos[(2, 1)], rec.pos[(2, 2)]))]
    assert all(b["kind"] == "pause" for b in got[0]["seam"])
    assert S.Donors([rec], REF).pause_spans(rec, k, "wrong_word", lambda r, kk: True, window=(0.0, 1e9)) == []  # too near the plant
    assert S.Donors([rec], REF).pause_spans(rec, rec.pos[(4, 2)], "wrong_word", lambda r, kk: True) == []  # inside a linked phrase


def test_an_added_word_alone_between_pauses():
    rec, _ = ikhlas(surah=114)  # every join a pause
    k = rec.pos[(4, 2)]
    got = S.Donors([rec], REF).pause_spans(rec, k, "added_word", lambda r, kk: kk == rec.pos[(1, 2)])
    assert [(x["a"], x["b"], x["span"][2:]) for x in got][0] == (k, k - 1, (rec.pos[(1, 2)],) * 2)  # nothing shared
    e = {"cls": "added_word", "k0": k, "k1": k, "at": k, "seam": got[0]["seam"], "remove": [], "before": k, "other": False,
         "donor": (rec, got[0]["dk"], None, None), "span": got[0]["span"]}
    win = window_of(rec)
    toks = S.tokens_for(win, rec, [e])
    assert [(t["k"], t["role"], "seam" in t) for t in toks[k - 1: k + 2]] == [(k - 1, "ref", False), (rec.pos[(1, 2)], "added", True), (k, "ref", True)]


def test_an_added_word_takes_shared_words_to_reach_pauses():
    rec, x = ikhlas(inner_gap=0.05, inner_db=-12.0, surah=114)
    pause_after(rec, rec.pos[(2, 1)])  # مَلِكِ | ٱلنَّاسِ
    k = rec.pos[(2, 2)]
    dx = rec.pos[(3, 1)]  # إِلَـٰهِ ٱلنَّاسِ: the donor's ٱلنَّاسِ reads as the host's
    got = [g for g in S.Donors([rec], REF).pause_spans(rec, k, "added_word", lambda r, kk: kk == dx)]
    assert [(g["a"], g["b"], g["span"][2:]) for g in got] == [(k, k, (dx, dx + 1))]
    assert all(b["kind"] == "pause" for b in got[0]["seam"])
    e = {"cls": "added_word", "k0": k, "k1": k, "at": k, "seam": got[0]["seam"], "remove": [], "before": k, "other": False,
         "donor": (rec, dx, rec.boundary(dx - 1, REF), rec.boundary(dx, REF)), "span": got[0]["span"]}
    win = window_of(rec)
    toks = S.tokens_for(win, rec, [e])
    assert [(t["k"], t["role"], t["at"]) for t in toks[k - 1: k + 3]] == [
        (k - 1, "ref", rec.words[k - 1]["ref"]), (dx, "added", None), (dx + 1, "ref", rec.words[k]["ref"]), (k + 1, "ref", rec.words[k + 1]["ref"])]
    pieces = S.pieces_of(toks, {rec.key: rec}, win["startS"], win["endS"], rec)
    s = S.label(win, rec, [e], toks, pieces, {rec.key: rec}, REF, {"snrDb": 15.0})
    assert S.validate(s, REF) == []
    (ev,) = s["events"]  # only the word put in is a mistake
    assert ev["errorClass"] == "added_word" and ev["location"] == {"surah": 114, "ayah": 2, "word": 2} and ev["observed"] == REF.ayah_words["114:3"][0]
    sp = s["splices"][0]
    assert [(m["out"]["boundary"], m["in"]["boundary"]) for m in sp["seams"]] == [("pause", "pause")] * 2
    assert sp["donor"]["shared"] == 1 and S.splice_rule_failures(sp) == []
    y = S.render(pieces, lambda audio: x)
    t = next(t for t in s["said"] if t["role"] == "added")
    assert tone_at(y, t["startS"] + 0.05, t["endS"] - 0.05) == pytest.approx(freq(dx), abs=2)
    item = S.review_splice(s, sp)
    assert item["audio"]["startS"] <= sp["seams"][0]["outS"] - 1.0 and item["audio"]["endS"] >= sp["seams"][-1]["outS"] + 0.9


def test_planted_inserted_audio_meets_the_host_only_at_pauses(monkeypatch):
    monkeypatch.setattr(S, "forms_other", lambda *a: False)
    got = Counter()
    for surah in (78, 87, 89, 114):
        for gap, db in ((0.05, -12.0), (0.35, -20.0)):
            rec, _ = ikhlas(inner_gap=gap, inner_db=db, surah=surah)
            win = window_of(rec, f"lab-v1-i{surah}")
            edits = S.plan_edits(win, {rec.key: rec}, REF, S.Donors([rec], REF), {}, [], seed=surah)
            toks = S.tokens_for(win, rec, edits)
            pieces = S.pieces_of(toks, {rec.key: rec}, win["startS"], win["endS"], rec)
            s = S.label(win, rec, edits, toks, pieces, {rec.key: rec}, REF, {"snrDb": 15.0})
            assert S.validate(s, REF) == []
            for sp in s["splices"]:
                assert S.splice_rule_failures(sp) == [], sp["id"]
                if S.inserted(sp):
                    got[(sp["type"], gap)] += 1
                    assert all(m[side]["boundary"] == "pause" for m in sp["seams"] for side in ("out", "in"))
                if sp["type"] in ("repeat", "restart"):
                    assert sp["seams"][0]["in"]["boundary"] == "pause"
    assert got[("wrong_word", 0.35)] and got[("added_word", 0.35)]  # and fewer, or none, where reciters link


def test_twin_spans_take_shared_words_to_reach_a_clean_seam():
    twins = S.twin_words(REF)
    shallow, _ = ikhlas(inner_gap=0.05, inner_db=-5.0, surah=114)  # every join inside an ayah linked and shallow
    k = shallow.pos[(3, 1)]  # إِلَـٰهِ ٱلنَّاسِ, the twin of مَلِكِ ٱلنَّاسِ
    refs = twins[shallow.words[k]["ref"]]
    got = S.Donors([shallow], REF).twin_spans(shallow, k, refs)
    assert [(x["a"], x["b"], x["dk"]) for x in got] == [(k, k + 1, shallow.pos[(2, 1)])]  # on to the ayah end, a pause both sides
    assert all(b["kind"] == "pause" for b in got[0]["seam"])
    deep, _ = ikhlas(inner_gap=0.05, inner_db=-12.0, surah=114)
    got = S.Donors([deep], REF).twin_spans(deep, k, refs)
    assert [(x["a"], x["b"]) for x in got] == [(k, k), (k, k + 1)]  # the one-word swap first
    assert S.Donors([deep], REF).twin_spans(deep, k, refs, window=(0.0, 1e9)) == []  # the twin sits inside the session, too near


def test_twin_span_labels_the_shared_words_as_recited():
    rec, _ = ikhlas(inner_gap=0.05, inner_db=-5.0, surah=114)
    k = rec.pos[(3, 1)]
    x = S.Donors([rec], REF).twin_spans(rec, k, S.twin_words(REF)[rec.words[k]["ref"]])[0]
    e = {"cls": "mutashabihat", "k0": x["a"], "k1": x["b"], "at": k, "seam": x["seam"], "remove": [k], "other": True,
         "donor": (x["rec"], x["dk"], rec.boundary(x["dk"] - 1, REF), rec.boundary(x["dk"], REF)), "span": (x["a"], x["b"], x["a"] + x["off"], x["b"] + x["off"])}
    last = len(rec.ayahs) - 1
    win = {"id": "lab-v1-t02", "rec": rec.key, "split": "practice", "kind": "muta", "dis": 0, "startAyah": 1, "endAyah": 6, "first": 0, "last": last,
           "startS": S.start_cut(rec, 0), "endS": S.end_cut(rec, last), "clean": False}
    toks = S.tokens_for(win, rec, [e])
    assert [(t["k"], t["role"]) for t in toks[k - 1: k + 3]] == [(k - 1, "ref"), (rec.pos[(2, 1)], "wrong"), (rec.pos[(2, 2)], "ref"), (k + 2, "ref")]
    pieces = S.pieces_of(toks, {rec.key: rec}, win["startS"], win["endS"], rec)
    assert [[t["k"] for t in p["tokens"]][:2] for p in pieces] == [[0, 1], [rec.pos[(2, 1)], rec.pos[(2, 2)]], [k + 2, k + 3]]
    s = S.label(win, rec, [e], toks, pieces, {rec.key: rec}, REF, {"snrDb": 15.0})
    assert S.validate(s, REF) == []
    sp = s["splices"][0]
    assert sp["mutashabihat"] and sp["location"] == {"surah": 114, "ayah": 3, "word": 1} and (sp["said"], sp["expected"]) == (REF.ayah_words["114:2"][0], REF.ayah_words["114:3"][0])
    assert sp["donor"]["span"] == ["114:2:1", "114:2:2"] and sp["donor"]["shared"] == 1 and len(sp["seams"]) == 2
    assert all(m["out"]["boundary"] == m["in"]["boundary"] == "pause" for m in sp["seams"])
    said = s["said"][k: k + 2]
    assert [w["text"] for w in said] == REF.ayah_words["114:2"] and [w["role"] for w in said] == ["wrong", "ref"]
    assert said[1]["at"] == {"surah": 114, "ayah": 3, "word": 2} and said[1]["from"] == {"audio": rec.audio, "surah": 114, "ayah": 2, "word": 2}


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

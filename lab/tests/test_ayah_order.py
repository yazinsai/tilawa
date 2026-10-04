"""Ayah-order check on synthetic token streams built from the corpus text."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB / "scripts"))

corpus_path = LAB / "data" / "zipformer" / "quran.json"
pytestmark = pytest.mark.skipif(not corpus_path.is_file(), reason="zipformer corpus not fetched")

@pytest.fixture(scope="module")
def ao():
    import ayah_order

    return ayah_order


@pytest.fixture(scope="module")
def corpus(ao):
    return ao.Corpus()


def tokens_for(corpus, pieces: list, gap: int = 20) -> list:
    """pieces: (surah, ayah) or a raw phoneme string; a pause of ``gap`` frames after each."""
    toks, f = [], 10
    for piece in pieces:
        text = "".join(corpus.ph[piece]) if isinstance(piece, tuple) else piece
        for ch in text:
            toks.append([ch, f, f * 0.04 + 0.5])
            f += 2
        f += gap
    return toks


def row_for(toks, expected=None, verses=None):
    return {"tokens": toks, "expected_verses": [{"surah": s, "ayah": a} for s, a in (expected or [])],
            "verses": verses or [], "duration_s": toks[-1][1] * 0.04}


def flags(ao, corpus, row, mode):
    return ao.flags_of(ao.detect_take(corpus, row, mode, ao.DEFAULTS), ao.RULE)


def test_segments_cut_at_pauses(ao):
    toks = [["a", 0, 0.1], ["b", 2, 0.1], ["c", 30, 1.3], ["d", 32, 1.3]]
    segs = ao.segments(toks, gap=12, min_chars=1)
    assert [s["text"] for s in segs] == ["ab", "cd"]


def test_skipped_middle_ayah_is_flagged(ao, corpus):
    exp = [(78, 13), (78, 14), (78, 15)]
    row = row_for(tokens_for(corpus, [(78, 13), (78, 15)]), expected=exp)
    out = flags(ao, corpus, row, "expected")
    assert [(f["surah"], f["ayah"]) for f in out] == [(78, 14)]


def test_in_order_take_is_not_flagged(ao, corpus):
    exp = [(78, 13), (78, 14), (78, 15)]
    row = row_for(tokens_for(corpus, exp), expected=exp)
    assert flags(ao, corpus, row, "expected") == []


def test_restart_and_preamble_are_not_skips(ao, corpus):
    exp = [(78, 13), (78, 14), (78, 15)]
    row = row_for(tokens_for(corpus, [ao.ISTIADHA, ao.BASMALA, (78, 13), (78, 13), (78, 14), (78, 15)]), expected=exp)
    assert flags(ao, corpus, row, "expected") == []


def test_no_passage_window_from_tracker(ao, corpus):
    row = row_for(tokens_for(corpus, [(78, 13), (78, 15)]), verses=[[78, 13], [78, 14]])
    out = flags(ao, corpus, row, "tracker")
    assert [(f["surah"], f["ayah"]) for f in out] == [(78, 14)]


def test_skip_read_later_is_not_flagged(ao, corpus):
    exp = [(78, 13), (78, 14), (78, 15)]
    row = row_for(tokens_for(corpus, [(78, 13), (78, 15), (78, 14)]), expected=exp)
    assert flags(ao, corpus, row, "expected") == []


def test_restart_and_garble_guards(ao):
    base = {"surah": 89, "ayah": 12, "fit0": 0.1, "fit1": 0.1, "between": 0, "read_later": False, "at": 3.0,
            "post_chars": 20, "pre_chars": 10, "pre_restart": False, "pre_on_skipped": 0.6, "skip_chars": 30}
    rule = {**ao.RULE, "no_restart_pre": True, "restart_alt": 0.3, "between_rel": 0.2}

    def n(**kw):
        return len(ao.flags_of({"margin": 5.0, "jumps": [{**base, **kw}]}, rule))

    assert n() == 1
    assert n(pre_restart=True) == 1                       # restart that doesn't fit the skipped ayah's ending
    assert n(pre_restart=True, pre_on_skipped=0.2) == 0   # rhyming restart: could be the skipped ayah's tail
    assert n(between=6) == 1
    assert n(between=8) == 0                              # 8 undecoded chars for a 30-char ayah


def test_identical_text_guard(ao, corpus):
    assert ao.ident_guarded(corpus, 109, 4)
    assert not ao.ident_guarded(corpus, 109, 3)
    assert not ao.ident_guarded(corpus, 94, 5)
    exp = [(109, 3), (109, 4), (109, 5)]
    row = row_for(tokens_for(corpus, [(109, 3), (109, 5)]), expected=exp)
    assert flags(ao, corpus, row, "expected") == []
    exp = [(94, 4), (94, 5), (94, 6)]
    row = row_for(tokens_for(corpus, [(94, 4), (94, 6)]), expected=exp)
    assert [(f["surah"], f["ayah"]) for f in flags(ao, corpus, row, "expected")] == [(94, 5)]

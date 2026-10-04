"""Acted-mistake scorer and label mapping. Synthetic rows only; no audio."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


ev = _load("acted_eval")
ce = _load("correction_eval")


def _label(kind: str, word: int | None = 2) -> dict:
    return {"id": "c1", "split": "test", "label_kind": kind, "surah": 2, "ayah": 5,
            "word_index": 0 if word is None else word, "mapped": True, "span_s": [1.0, 1.5]}


def test_hits_window_and_skip_ayah():
    lab = _label("skip_word")
    assert ev.hits({"surah": 2, "ayah": 5, "word": 3}, lab)
    assert not ev.hits({"surah": 2, "ayah": 5, "word": 4}, lab)
    assert not ev.hits({"surah": 2, "ayah": 6, "word": 2}, lab)
    skip = _label("skip_ayah", None)
    assert ev.hits({"surah": 2, "ayah": 5, "word": 7}, skip)


def test_clip_stats_kind_latency_and_notes():
    lab = _label("repeat")
    row = {"issues": [{"kind": "possible_substitution", "surah": 2, "ayah": 5, "word": 3, "atSeconds": 3.0},
                      {"kind": "possible_omission", "surah": 2, "ayah": 5, "word": 9, "atSeconds": 4.0}],
           "notes": [{"kind": "possible_repetition", "surah": 2, "ayah": 5, "word": 2, "atSeconds": 2.0}]}
    st = ev.acted_clip_stats(lab, row)
    assert st["caught"] and not st["kind_ok"] and not st["exact"] and st["note_caught"]
    assert st["latency"] == 2.0
    agg = ev.aggregate([{**st, "speaker": "s", "minutes": 0.1}], [{"flags": [("possible_vowel", False)], "notes": [], "minutes": 1.0, "speaker": "t"}])
    assert agg["all"]["flags"] == 3 and agg["all"]["tp"] == 1
    assert ev.kind_prf(agg, "repeat")["r"] == 1.0


def test_acted_word_index_basmala_and_mismatch():
    words = ["قل", "هو", "الله", "أحد"]
    assert ce.acted_word_index(3, "الله", words, 4) == 2
    # Label counted on basmala-prefixed text: re-found by the expected word.
    assert ce.acted_word_index(7, "الله", words, 4) == 2
    assert ce.acted_word_index(3, "الله", words, 5) is None
    assert ce.acted_word_index(3, "غير", words, 4) is None


def test_locate_summary_has_no_ids():
    rows = [{"id": "secret", "label_kind": "vowel", "split": "dev", "mapped": True, "span_s": [0, 1],
             "confirmed_any": True, "confirmed_all": False, "exact_any": True}]
    assert "secret" not in json.dumps(ce.acted_locate_summary(rows))

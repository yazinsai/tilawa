"""Similar-verse check: index, variants and detection on text only (no audio, no ONNX)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sv = _load("similar_verse")
sve = _load("similar_verse_eval")


class _Corpus(sv.Corpus):
    def __init__(self, ayahs: dict[tuple[int, int], list[str]]):
        self.ph = {k: list(v) for k, v in ayahs.items()}
        self.keys = {k: list(v) for k, v in ayahs.items()}


E = (1, 1)
M = (2, 1)
CORPUS = _Corpus({
    E: ["فَءِننَ", "لجَننَةَ", "هِيَ", "لمَءوَاا", "لِلمُتتَقِۦۦن"],
    M: ["فَءِننَ", "لجَحِۦۦمَ", "هِيَ", "لمَءوَاا", "لِلكَاافِرِۦۦن"],
    (3, 1): ["قُل", "هُوَ", "للَااهُ", "ءَحَد"],
})


def _row(words: list[str], passage=E) -> dict:
    text = "".join(words)
    return {"id": "x", "duration_s": 4.0, "tokens": [[ch, i, 0.1 * i] for i, ch in enumerate(text)],
            "expected_verses": [{"surah": passage[0], "ayah": passage[1]}], "verses": [list(passage)]}


def test_cost_mirrors_sdk():
    assert sv.char_cost("ا", "ا") == 0
    assert sv.char_cost("ۦ", "ي") == 0
    assert sv.char_cost("َ", "ُ") == 0.1
    assert sv.char_cost("أ", "ء") == 0.1
    assert sv.char_cost("د", "ذ") == 0.25
    assert sv.char_cost("ب", "ل") == 1.0
    q, r = sv.encode("ءَحَد"), sv.encode("ءَحَد")
    assert sv.align_cost(q, r, sv.COST, 0.5) == 0


def test_index_pairs_lookalikes_and_skips_unrelated():
    idx = sv.build_index(CORPUS, min_shared=3, min_frac=0.5)
    assert [c["m"] for c in idx["1:1"]] == [[2, 1]]
    assert "3:1" not in idx
    tags = {r[0] for r in idx["1:1"][0]["regions"]}
    assert tags == {"replace"}


def test_lookalike_swap_flags_divergence_word():
    idx = sv.build_index(CORPUS)
    heard = CORPUS.ph[E][:1] + CORPUS.ph[M][1:2] + CORPUS.ph[E][2:]
    cands = sv.detect_take(CORPUS, idx, _row(heard), "expected")
    rule = {"families": ["lookalike", "drop"], "e": {"default": 3.5, "drop": 10}, "max_span": 1,
            "loc": 0.25, "ayah_d": 0.5}
    picked = sve.pick(cands, rule)
    assert len(picked) == 1
    assert (picked[0]["word"], picked[0]["kind"], picked[0]["from"]) == (1, "possible_substitution", [2, 1])
    assert picked[0]["at"] is not None


def test_clean_reading_and_pure_lookalike_are_not_flagged():
    idx = sv.build_index(CORPUS)
    rule = {"families": ["lookalike", "drop"], "e": {"default": 3.5, "drop": 10}, "max_span": 1,
            "loc": 0.25, "ayah_d": 0.5}
    assert sve.pick(sv.detect_take(CORPUS, idx, _row(CORPUS.ph[E]), "expected"), rule) == []
    # No passage: reading M whole is a correct reading of M, whatever the tracker locked.
    row = _row(CORPUS.ph[M], passage=E)
    row["expected_verses"] = []
    assert sve.pick(sv.detect_take(CORPUS, idx, row, "tracker"), {**rule, "all": True}) == []


def test_drop_variant_flags_skipped_word():
    idx = sv.build_index(CORPUS)
    heard = CORPUS.ph[E][:2] + CORPUS.ph[E][3:]
    cands = sv.detect_take(CORPUS, idx, _row(heard), "expected")
    drops = [c for c in cands if c["family"] == "drop" and c["word"] == 2 and c["span"] == 1]
    assert drops and drops[0]["margin"] > 0 and drops[0]["kind"] == "possible_omission"

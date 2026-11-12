"""Self-repair miner. Synthetic token ids only; no audio and no clip ids."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

_PATH = ROOT / "scripts" / "self_repair.py"
_SPEC = importlib.util.spec_from_file_location("self_repair", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
sr = importlib.util.module_from_spec(_SPEC)
sys.modules["self_repair"] = sr
_SPEC.loader.exec_module(sr)

TOK = ["a", "b", "c", "d", "e", "f", "g", "z", "بَ", "بُ", "<blank>"]


def _mine(words, hyp):
    return sr.mine_collapsed(words, hyp, TOK)


def test_restart_with_one_bad_word_is_a_letter_slip():
    # w0 ab, w1 cd, w2 e, w3 fg. First pass says cz for w1, then restarts at w0.
    words = [[0, 1], [2, 3], [4], [5, 6]]
    hyp = [0, 1, 2, 7, 0, 1, 2, 3, 4, 5, 6]
    rows = _mine(words, hyp)
    assert len(rows) == 1
    assert rows[0].word_index == 1
    assert rows[0].kind == "letter"
    assert rows[0].rewind >= 1
    assert rows[0].hyp_indexes  # the bad token, not the repaired one


def test_exact_reread_is_not_a_slip():
    words = [[0, 1], [2, 3], [4], [5, 6]]
    hyp = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]
    assert _mine(words, hyp) == []


def test_single_word_stutter_is_not_a_slip():
    words = [[0, 1], [2, 3], [4], [5, 6]]
    hyp = [0, 1, 7, 7, 0, 1, 2, 3, 4, 5, 6]
    assert _mine(words, hyp) == []


def test_internal_skip_then_restart():
    # Said w0, skipped w1, said w2, then restarted the span correctly.
    words = [[0, 1], [2, 3], [4], [5, 6]]
    hyp = [0, 1, 4, 0, 1, 2, 3, 4, 5, 6]
    rows = _mine(words, hyp)
    assert [row.word_index for row in rows] == [1]
    assert rows[0].kind == "skip"


def test_whole_word_swap_is_substitution():
    # Two neighbouring words match, so the replaced word is a restart of the span.
    words = [[0, 1], [2, 3, 4], [5], [6]]
    hyp = [0, 1, 7, 7, 7, 5, 0, 1, 2, 3, 4, 5, 6]
    rows = _mine(words, hyp)
    assert len(rows) == 1
    assert rows[0].word_index == 1
    assert rows[0].kind == "substitution"


def test_trailing_scrap_after_a_clean_word_is_not_a_slip():
    words = [[0, 1], [2, 3], [4, 5], [6]]
    hyp = [0, 1, 2, 3, 7, 0, 1, 2, 3, 4, 5, 6]
    assert _mine(words, hyp) == []


def test_vowel_only_change():
    assert sr.classify_deviation(["بَ"], ["بُ"]) == "vowel"
    assert sr.classify_deviation(["بَ", "a"], ["بُ", "a"]) == "vowel"
    assert sr.classify_deviation(["a", "b"], ["a", "z"]) == "letter"
    assert sr.classify_deviation(["a", "b", "c"], []) == "skip"
    assert sr.classify_deviation(["a", "b", "c"], ["z", "z", "z"]) == "substitution"


def test_more_than_three_deviant_words_drops_the_clip():
    # Each word keeps its first token and flips the second, so the attempt is
    # recognisable, and four slips on one clip are dropped.
    words = [[0, 1], [2, 3], [4, 5], [6, 0]]
    hyp = [0, 7, 2, 7, 4, 7, 6, 7, 0, 1, 2, 3, 4, 5, 6, 0]
    assert _mine(words, hyp) == []


def test_madd_length_alone_is_not_a_slip():
    tokens = ["ا", "ب", "<blank>"]
    # Reference word is a madd run. Both renditions differ only in length.
    words_ph = ["اا", "ب"]
    from locate_slips import encode_tokens

    hyp = encode_tokens("اب" + "ااب", tokens)
    rows = sr.mine_self_repairs(words_ph, hyp, tokens)
    assert rows == []


def test_coincide_word_and_clip():
    candidates = [
        {"id": "c1", "surah": 2, "ayah": 5, "word_index": 3},
        {"id": "c2", "surah": 2, "ayah": 6, "word_index": 1},
    ]
    issues = [
        {"id": "c1", "surah": 2, "ayah": 5, "word": 3},
        {"id": "c1", "surah": 2, "ayah": 5, "word": 8},
        {"id": "c9", "surah": 2, "ayah": 5, "word": 3},
    ]
    exact = sr.coincide(candidates, issues, tol=0)
    assert exact["coincide_word"] == 1
    assert exact["coincide_clip"] == 1
    near = sr.coincide(candidates, [{"id": "c2", "surah": 2, "ayah": 6, "word": 2}], tol=1)
    assert near["coincide_word"] == 1
    assert sr.coincide(candidates, [{"id": "c2", "surah": 2, "ayah": 6, "word": 2}], tol=0)["coincide_word"] == 0

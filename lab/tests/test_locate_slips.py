"""Word-level slip alignment on hand-made token sequences. No audio."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_PATH = ROOT / "scripts" / "locate_slips.py"
_SPEC = importlib.util.spec_from_file_location("locate_slips", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
loc = importlib.util.module_from_spec(_SPEC)
sys.modules["locate_slips"] = loc
_SPEC.loader.exec_module(loc)

# Single-character inventory plus madd runs, so longest-match is obvious.
TOKENS = ["ب", "ت", "م", "ل", "ن", "ر", "ا", "ۥ", "ۦ", "اا", "اااا"]
B, T, M, L, N, R, ALEF, WAW, YAA, ALEF2, ALEF4 = range(11)


def _slips(words, hyp):
    return loc.locate_word_slips(words, hyp, TOKENS)


def _one(words, hyp):
    slips = _slips(words, hyp)
    assert len(slips) == 1, [(s.word_index, s.kind, s.extent, s.ops) for s in slips]
    return slips[0]


def test_identical_sequence_has_no_slip():
    assert _slips([[B, T], [M, L], [N]], [B, T, M, L, N]) == []


def test_madd_length_difference_is_not_a_slip():
    # ب + اااااا vs ب + اا. Length-only madd, same letter.
    words = [[B, ALEF4], [T]]
    hyp = [B, ALEF2, T]
    assert _slips(words, hyp) == []
    # A run of single-alef tokens collapses the same way.
    assert _slips([[B, ALEF, ALEF, ALEF], [T]], [B, ALEF, T]) == []


def test_madd_letter_change_is_a_partial_substitution():
    slip = _one([[B, ALEF], [T]], [B, WAW, T])
    assert (slip.word_index, slip.kind, slip.extent) == (0, "substituted", "partial")
    assert slip.ops == [{"op": "replace", "ref": "ا", "hyp": "ۥ"}]


def test_whole_word_omitted():
    slip = _one([[B, T], [M, L], [N, R]], [B, T, N, R])
    assert (slip.word_index, slip.kind, slip.extent) == (1, "omitted", "whole")
    assert {op["op"] for op in slip.ops} == {"delete"}


def test_partial_word_omitted():
    slip = _one([[B, T, M], [L]], [B, T, L])
    assert (slip.word_index, slip.kind, slip.extent) == (0, "omitted", "partial")


def test_whole_word_substituted():
    slip = _one([[B, T], [M, L], [N]], [B, T, R, N])
    assert (slip.word_index, slip.kind, slip.extent) == (1, "substituted", "whole")
    assert any(op["op"] == "replace" for op in slip.ops)


def test_partial_word_substituted():
    slip = _one([[B, T, M], [L]], [B, R, M, L])
    assert (slip.word_index, slip.kind, slip.extent) == (0, "substituted", "partial")


def test_whole_word_repeated():
    slip = _one([[B, T], [M, L], [N]], [B, T, M, L, M, L, N])
    assert (slip.word_index, slip.kind, slip.extent) == (1, "repeated", "whole")
    assert slip.word_end == 2


def test_phrase_repeated_stays_a_repeat():
    # Echo of the last two words, not a jump back to the start.
    slip = _one([[B], [T], [M], [L]], [B, T, M, T, M, L])
    assert (slip.word_index, slip.kind, slip.extent) == (1, "repeated", "whole")
    assert slip.word_end == 3


def test_partial_word_repeated():
    slip = _one([[B, T, M, L]], [B, T, M, T, M, L])
    assert (slip.word_index, slip.kind, slip.extent) == (0, "repeated", "partial")


def test_restart_from_the_first_word():
    slip = _one([[B], [T], [M], [L]], [B, T, M, B, T, M, L])
    assert (slip.word_index, slip.kind, slip.extent) == (0, "restarted", "whole")
    assert slip.word_end == 3


def test_restart_from_an_earlier_word():
    # After word 3, the reciter goes back to word 1 (two words back).
    slip = _one([[B], [T], [M], [L], [N]], [B, T, M, L, T, M, L, N])
    assert (slip.word_index, slip.kind, slip.extent) == (1, "restarted", "whole")


def test_two_slips_in_one_ayah():
    # Word 1 omitted, word 3 substituted.
    slips = _slips([[B], [T], [M], [L], [N]], [B, M, R, N])
    got = [(s.word_index, s.kind) for s in slips]
    assert got == [(1, "omitted"), (3, "substituted")]


def test_span_seconds_uses_40ms_frames():
    frames = [(0, 4), (5, 9), (10, 12)]
    assert loc.span_seconds(frames, [1]) == [0.2, 0.4]
    assert loc.span_seconds(frames, [0, 2]) == [0.0, 0.52]
    assert loc.span_seconds(frames, []) is None


def test_clamp_span_drops_encoder_drain():
    assert loc.clamp_span([1.0, 9.0], 4.2) == [1.0, 4.2]


def test_one_phone_edit_is_not_a_review_candidate():
    partial = _one([[B, T, M], [L]], [B, R, M, L])
    assert partial.n_edits == 1
    assert loc.is_review_slip(partial) is False
    half = _one([[B, T, M, L]], [R, N, M, L])
    assert half.extent == "partial" and half.n_edits == 2
    assert loc.is_review_slip(half) is True
    assert loc.is_review_slip(_one([[B, T], [M, L], [N]], [B, T, N])) is True


def test_confidence_ranks_a_clean_whole_word_edit_highest():
    high = loc.confidence_score(
        kind_agree=True, extent="whole", edit_tokens=4, word_tokens=4, neighbours_clean=True
    )
    small = loc.confidence_score(
        kind_agree=True, extent="partial", edit_tokens=1, word_tokens=6, neighbours_clean=False
    )
    assert high >= loc.HIGH_CONFIDENCE
    assert small < loc.HIGH_CONFIDENCE
    assert high > small
    # A long unmatched insertion glued onto a short word is not one-word evidence.
    glued = loc.confidence_score(
        kind_agree=True, extent="partial", edit_tokens=40, word_tokens=4, neighbours_clean=True
    )
    assert glued < loc.HIGH_CONFIDENCE
    assert glued < high


def test_passage_slip_maps_onto_the_ayah_of_its_first_word():
    # Ayah 1:1 is ب ت, ayah 1:2 is م ل. The hypothesis drops م.
    slips = loc.locate_clip(["ب", "ت", "م", "ل"], [B, T, L], TOKENS)
    assert len(slips) == 1 and slips[0].word_index == 2
    index = [(1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1)]
    key, local = loc.ayah_local_slip(slips[0], index)
    assert key == (1, 2)
    assert (local.word_index, local.kind) == (0, "omitted")
    # A slip that starts in ayah 1 stays there, even if the span is passage-indexed.
    first = loc.locate_clip(["ب", "ت", "م", "ل"], [T, M, L], TOKENS)
    assert first and first[0].word_index == 0
    key, local = loc.ayah_local_slip(first[0], index)
    assert key == (1, 1)
    assert local.word_index == 0


def test_agreement_requires_the_same_word_index():
    left = _slips([[B, T], [M, L], [N]], [B, T, N])
    right = _slips([[B, T], [M, L], [N]], [B, T, M, L, R])
    # left omits word 1; right substitutes word 2. No shared location.
    merged = loc.merge_agreed(left, right, n_words=3, left_name="v3", right_name="a0w-ep1-a0.5")
    assert merged == []
    both = _slips([[B, T], [M, L], [N]], [B, T, N])
    merged = loc.merge_agreed(left, both, n_words=3, left_name="v3", right_name="a0w-ep1-a0.5")
    assert [row["word_index"] for row in merged] == [1]
    assert merged[0]["kind"] == "omitted"
    assert merged[0]["confidence"] >= loc.HIGH_CONFIDENCE


def test_same_ayah_suspect_filter():
    dev = {"keep-me-out"}
    assert loc.same_ayah_suspect(
        {"id": "a", "bucket": "suspect", "per": 0.2, "surah": 2, "ayah": 5, "alt_key": "2:5"}, dev
    )
    assert not loc.same_ayah_suspect(
        {"id": "keep-me-out", "bucket": "suspect", "per": 0.2, "surah": 2, "ayah": 5, "alt_key": "2:5"},
        dev,
    )
    assert not loc.same_ayah_suspect(
        {"id": "b", "bucket": "suspect", "per": 0.3, "surah": 2, "ayah": 5, "alt_key": "3:1"}, dev
    )
    assert not loc.same_ayah_suspect(
        {"id": "c", "bucket": "mislabel", "per": 0.5, "surah": 2, "ayah": 5, "alt_key": "2:5"}, dev
    )

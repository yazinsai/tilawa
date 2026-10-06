# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --with pytest pytest tools/test_help_import.py"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from help_import import WordCountMismatch, apply_swaps, label, site_counts, split_speakers, word_ref  # noqa: E402

QURAN = {
    "112:1": {"text_clean": "بسم الله الرحمن الرحيم قل هو الله احد"},
    "112:2": {"text_clean": "الله الصمد ۚ"},
    "2:181": {"text_clean": "فمن بدلهۥ بعدما سمعهۥ"},
}
REF = {"112:1": ["قُلْ", "هُوَ", "ٱللَّهُ", "أَحَدٌ"], "112:2": ["ٱللَّهُ", "ٱلصَّمَدُ"], "2:181": ["فَمَنۢ", "بَدَّلَهُۥ", "بَعْدَ", "مَا", "سَمِعَهُۥ"]}


def test_word_index_maps_onto_reference_word_with_waqf_marks_dropped():
    assert word_ref(QURAN, REF, "112:2", 2) == {"surah": 112, "ayah": 2, "word": 2}


def test_basmala_prefixed_first_ayah_shifts_by_four():
    assert word_ref(QURAN, REF, "112:1", 5) == {"surah": 112, "ayah": 1, "word": 1}


def test_word_inside_the_basmala_prefix_fails():
    with pytest.raises(WordCountMismatch):
        word_ref(QURAN, REF, "112:1", 2)


def test_word_count_disagreement_fails_loudly():
    with pytest.raises(WordCountMismatch):
        word_ref(QURAN, REF, "2:181", 1)


def clip(kind, mistakes, ref="112:1-112:2", **extra):
    return {"id": "c1", "speaker": "s1", "prompt_version": 2, "ref": ref, "kind": kind, "mistakes": mistakes, "extra_mistake": False, **extra}


def test_substitution_is_a_wrong_word_at_its_reference_location():
    m = {"kind": "substitution", "ayah": "112:2", "word": 2, "expected": "الصمد", "said": "لكنا", "said_uthmani": "لَّٰكِنَّا"}
    lab = label(QURAN, REF, clip("substitution", [m]))
    assert lab["events"] == [{"errorClass": "wrong_word", "state": "confirmed", "location": {"surah": 112, "ayah": 2, "word": 2}, "expected": "الصمد", "observed": "لَّٰكِنَّا"}]
    assert lab["scored"] and lab["ayahs"] == ["112:1", "112:2"]


def test_repeat_is_not_an_error():
    lab = label(QURAN, REF, clip("repeat", [{"kind": "repeat", "ayah": "112:1", "word": 6, "expected": "هو"}]))
    assert lab["events"] == []
    assert lab["non_errors"] == [{"kind": "repeat", "location": {"surah": 112, "ayah": 1, "word": 2}}]


def test_skipped_ayah_is_a_suspected_omission_of_the_whole_ayah():
    lab = label(QURAN, REF, clip("skip_ayah", [{"kind": "skip_ayah", "ayah": "112:2"}]))
    assert lab["events"] == [{"errorClass": "omitted_word", "state": "suspected", "location": {"surah": 112, "ayah": 2, "word": 1}, "words": 2}]
    lab = label(QURAN, REF, clip("skip_ayah", [{"kind": "skip_ayah", "ayah": "112:1"}]))
    assert lab["events"][0]["errorClass"] == "skipped_passage"


def test_pronunciation_rules_map_to_core_classes_and_ghunna_is_unsupported():
    def cls(m):
        return [e["errorClass"] for e in label(QURAN, REF, clip(m["kind"], [m]))["events"]]

    at = {"ayah": "112:2", "word": 2, "expected": "الصمد"}
    assert cls({"kind": "vowel", **at, "letter": 1, "from": "fatha", "to": "kasra"}) == ["short_vowel"]
    assert cls({"kind": "tajweed", "rule": "light_emphatic", "letter": "ص", **at}) == ["consonant_sad_sin"]
    assert cls({"kind": "tajweed", "rule": "no_shadda", **at}) == ["missing_shadda"]
    assert cls({"kind": "tajweed", "rule": "madd_short", **at}) == ["madd_natural"]
    lab = label(QURAN, REF, clip("tajweed", [{"kind": "tajweed", "rule": "no_ghunna", **at}]))
    assert lab["events"] == [] and lab["unsupported"] == [{"rule": "no_ghunna", "location": {"surah": 112, "ayah": 2, "word": 2}}]


def test_version_swap_relabels_to_the_shown_v2_prompt_and_flags_it():
    stored = clip("none", [], ref="112:1", prompt_version=1)
    swaps = {"c1": {"kind": "skip_word", "ref": "112:2", "mistakes": [{"kind": "skip_word", "ayah": "112:2", "word": 1, "expected": "الله"}]}}
    (r,) = apply_swaps([stored], swaps)
    lab = label(QURAN, REF, r)
    assert lab["ayahs"] == ["112:2"] and lab["relabel"] == "version_swap"
    assert lab["events"][0]["errorClass"] == "omitted_word"
    assert "relabel" not in label(QURAN, REF, stored)


def labs(speaker, kinds):
    return [{"id": f"{speaker}-{i}", "speaker": speaker, "kind": k} for i, k in enumerate(kinds)]


def test_split_keeps_heavy_speakers_in_train_and_stratifies_kinds_over_held_out_speakers():
    rows = labs("heavy", ["vowel"] * 16 + ["none"] * 4) + labs("h2", ["repeat"] * 16)
    for n in range(8):
        rows += labs(f"s{n}", ["vowel", "repeat", "skip_word", "none"] if n % 2 else ["skip_word", "none"])
    split = split_speakers(rows)
    assert split["heavy"] == split["h2"] == "train"
    held = {s for s, v in split.items() if v == "eval"}
    assert held and held < {f"s{n}" for n in range(8)}
    for kind in ("vowel", "repeat", "skip_word", "none"):
        assert {split[r["speaker"]] for r in rows if r["kind"] == kind} == {"train", "eval"}, kind


def test_same_speaker_on_same_ayah_under_different_seeds_is_one_site():
    def lab(speaker, ayah, word, scored=True):
        return {"speaker": speaker, "scored": scored, "events": [{"errorClass": "wrong_word", "location": {"surah": 112, "ayah": ayah, "word": word}}]}

    rows = [lab("a", 1, 1), lab("a", 1, 3), lab("a", 2, 1), lab("b", 1, 1), lab("c", 1, 1, scored=False)]
    assert site_counts(rows) == {"wrong_word": 3}


def test_extra_mistake_clips_are_dropped_from_scoring():
    assert not label(QURAN, REF, clip("none", [], extra_mistake=True))["scored"]

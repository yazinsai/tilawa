"""Pure helpers for the help-recording pull. Synthetic rows only; no network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "pull_help_recordings.py"
_SPEC = importlib.util.spec_from_file_location("pull_help_recordings", _PATH)
pull = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(pull)


def _index(*refs: tuple[int, int]) -> pull.VerseIndex:
    return pull.VerseIndex(list(refs))


def test_parse_ref_single_and_span():
    assert pull.parse_ref("2:183") == ((2, 183), None)
    assert pull.parse_ref(" 83:25-83:26 ") == ((83, 25), (83, 26))
    assert pull.parse_ref("1:7-2:2") == ((1, 7), (2, 2))


@pytest.mark.parametrize(
    "text",
    ["", "2", "2:", ":1", "a:1", "2:1-3", "2:1-", "-2:1", "2:1-2", "s:a-s:b"],
)
def test_parse_ref_rejects_malformed(text: str):
    with pytest.raises(ValueError):
        pull.parse_ref(text)


def test_expected_verses_expands_in_corpus_order():
    index = _index((1, 7), (2, 1), (2, 2), (2, 3), (83, 25), (83, 26), (83, 34), (83, 35), (83, 36))
    single = pull.expected_verses({"ref": "83:25", "ayahs": ["83:25"]}, index)
    assert single == [{"surah": 83, "ayah": 25}]
    run = pull.expected_verses(
        {"ref": "83:34-83:36", "ayahs": ["83:34", "83:35", "83:36"]},
        index,
    )
    assert run == [
        {"surah": 83, "ayah": 34},
        {"surah": 83, "ayah": 35},
        {"surah": 83, "ayah": 36},
    ]
    cross = pull.expected_verses({"ref": "1:7-2:2"}, index)
    assert cross == [
        {"surah": 1, "ayah": 7},
        {"surah": 2, "ayah": 1},
        {"surah": 2, "ayah": 2},
    ]


def test_expected_verses_rejects_unknown_reversed_and_disagree():
    index = _index((2, 1), (2, 2))
    with pytest.raises(ValueError):
        pull.expected_verses({"ref": "9:1"}, index)
    with pytest.raises(ValueError):
        pull.expected_verses({"ref": "2:2-2:1"}, index)
    with pytest.raises(ValueError):
        pull.expected_verses({"ref": "2:1", "ayahs": ["2:2"]}, index)


def test_kind_filter_counts_acted_rows_and_drops_them():
    acted = {
        "id": "acted-row",
        "kind": "acted",
        "file": "audio/acted-row.webm",
        "mistakes": ["do-not-keep"],
    }
    missing_kind = {"id": "no-kind", "file": "audio/no-kind.wav"}
    clean = {"id": "clean-row", "kind": "none", "ref": "1:1"}
    other_clean = {"id": "also-clean", "kind": "none"}
    kept, excluded = pull.filter_kind_none([clean, acted, missing_kind, other_clean])
    assert excluded == 2
    assert [row["id"] for row in kept] == ["clean-row", "also-clean"]
    assert all(row["kind"] == "none" for row in kept)
    blob = json.dumps(kept)
    assert "do-not-keep" not in blob
    assert "acted-row" not in blob


def test_speaker_split_is_disjoint_deterministic_and_near_40_60():
    seconds = {f"s{i:02d}": float((i % 5) + 1) for i in range(30)}
    seconds["s00"] = 40.0
    first = pull.assign_speaker_splits(seconds)
    second = pull.assign_speaker_splits(dict(reversed(list(seconds.items()))))
    assert first == second
    assert set(first) == set(seconds)
    assert set(first.values()) <= {"dev", "test"}
    dev_speakers = {speaker for speaker, split in first.items() if split == "dev"}
    test_speakers = {speaker for speaker, split in first.items() if split == "test"}
    assert dev_speakers
    assert test_speakers
    assert dev_speakers.isdisjoint(test_speakers)
    dev_seconds = sum(seconds[speaker] for speaker in dev_speakers)
    total = sum(seconds.values())
    assert abs(dev_seconds / total - 0.4) < 0.08

    clips = []
    for speaker, n_clips in (("s00", 3), ("s01", 2), ("s02", 4)):
        for _ in range(n_clips):
            clips.append({"speaker": speaker, "split": first[speaker]})
    by_speaker: dict[str, set[str]] = {}
    for clip in clips:
        by_speaker.setdefault(clip["speaker"], set()).add(clip["split"])
    assert all(len(splits) == 1 for splits in by_speaker.values())


def test_slip_use_is_not_clean():
    index = _index((2, 1), (2, 2))
    base = {
        "id": "clip_1",
        "file": "audio/clip_1.wav",
        "speaker": "a" * 20,
        "prompt_version": 4,
        "kind": "none",
        "ref": "2:1-2:2",
        "ayahs": ["2:1", "2:2"],
        "passage": "run",
        "device": "phone",
        "gender": None,
        "level": None,
    }
    clean = pull.prepare_row({**base, "extra_mistake": False}, index)
    slip = pull.prepare_row({**base, "extra_mistake": True}, index)
    assert clean["use"] == "clean"
    assert slip["use"] == "slip"
    assert slip["use"] != "clean"
    assert clean["n_ayahs"] == 2
    assert clean["ayah_end"] == 2
    assert clean["expected_verses"] == [{"surah": 2, "ayah": 1}, {"surah": 2, "ayah": 2}]


def test_merge_keeps_prior_speakers_and_splits_new_ones():
    prior = {"old_dev": "dev", "old_test": "test"}
    seconds = {"old_dev": 10.0, "old_test": 10.0, **{f"n{i}": 5.0 for i in range(10)}}
    merged = pull.merge_speaker_splits(prior, seconds)
    assert merged["old_dev"] == "dev" and merged["old_test"] == "test"
    assert set(merged) == set(seconds)
    new_dev = [sp for sp, split in merged.items() if sp.startswith("n") and split == "dev"]
    assert 0 < len(new_dev) < 10


def test_acted_row_keeps_label_and_validates_ayah():
    index = _index((2, 1), (2, 2))
    base = {
        "id": "clip_2",
        "file": "audio/clip_2.wav",
        "speaker": "b" * 20,
        "prompt_version": 2,
        "kind": "skip_word",
        "ref": "2:1-2:2",
        "extra_mistake": False,
        "mistakes": [{"kind": "skip_word", "ayah": "2:2", "word": 3, "expected": "x"}],
    }
    spec = pull.prepare_row(base, index)
    assert spec["use"] == "acted"
    assert spec["mistake"] == {"kind": "skip_word", "surah": 2, "ayah": 2, "word": 3, "expected": "x"}
    skip = pull.prepare_row({**base, "kind": "skip_ayah", "mistakes": [{"kind": "skip_ayah", "ayah": "2:1"}]}, index)
    assert skip["mistake"]["word"] is None
    with pytest.raises(pull.RowError):
        pull.prepare_row({**base, "mistakes": [{"kind": "skip_word", "ayah": "3:1", "word": 0}]}, index)
    clean = pull.prepare_row({**base, "kind": "none", "mistakes": []}, index)
    assert clean["use"] == "clean" and clean["mistake"] is None

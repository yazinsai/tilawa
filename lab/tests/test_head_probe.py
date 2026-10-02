"""Deterministic checks for the slip-head probe helpers. No audio, no ids."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "head_probe"))
from common import (  # noqa: E402
    fit_splice, group_overlap, hits_label, k_for_rate, skeleton, threshold_at_k, units_from_track,
)


def test_skeleton_drops_vowels_only():
    bare = skeleton("لعَاالَمِۦۦن")
    assert "َ" not in bare and "ۦ" not in bare and "ِ" not in bare
    assert bare.startswith("لع") and bare.endswith("ن")
    assert skeleton("بَ") != skeleton("بِ") or skeleton("بَ") == "ب"
    assert skeleton("كتب") == "كتب"


def test_units_split_a_skipped_gap():
    # words 0 and 3 tracked, 1 and 2 skipped across frames 0..4 and 10..12
    track = [
        ["a", 0, 2, 5, 0],
        ["b", 4, 2, 5, 0],
        ["c", 10, 2, 5, 3],
        ["d", 12, 2, 5, 3],
    ]
    units = units_from_track(track)
    by_word = {u["word"]: u for u in units}
    assert set(by_word) == {0, 1, 2, 3}
    assert by_word[0]["skipped"] is False and by_word[1]["skipped"] is True
    assert by_word[1]["b"] <= by_word[2]["a"]
    assert by_word[1]["a"] >= 5 and by_word[2]["b"] <= 10


def test_threshold_flags_only_the_top_k():
    scores = np.array([0.9, 0.5, 0.4, 0.1])
    thr0 = threshold_at_k(scores, 0)
    assert np.sum(scores >= thr0) == 0
    thr1 = threshold_at_k(scores, 1)
    assert np.sum(scores >= thr1) == 1
    assert k_for_rate(0.0, 100) == 0
    assert k_for_rate(0.1, 33.7) == 3


def test_hit_window_and_skip_ayah():
    lab = {"mapped": True, "surah": 2, "ayah": 5, "word_index": 4, "label_kind": "substitution"}
    assert hits_label(2, 5, 5, lab) and not hits_label(2, 5, 6, lab)
    assert not hits_label(3, 5, 4, lab)
    skip = {"mapped": True, "surah": 2, "ayah": 5, "word_index": 0, "label_kind": "skip_ayah"}
    assert hits_label(2, 5, 9, skip)
    assert not hits_label(2, 6, 0, skip)


def test_splice_keeps_length_and_crossfades_edges():
    host = np.linspace(-1, 1, 1600, dtype=np.float32)
    donor = np.ones(800, dtype=np.float32)
    out = fit_splice(host, 400, 1200, donor, 0, 800, fade_s=0.01, sr=16000)
    assert out is not None and len(out) == len(host)
    assert out[0] == host[0] and out[-1] == host[-1]
    # The middle of the slot is the donor, not the host line.
    assert abs(float(out[800]) - 1.0) < 1e-5


def test_group_overlap_is_the_leak_check():
    assert group_overlap({"a", "b"}, {"c"}) == 0
    assert group_overlap({"a"}, {"a", "b"}) == 1

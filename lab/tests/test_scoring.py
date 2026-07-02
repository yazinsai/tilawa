import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.runner import score_sequence


def test_perfect_match():
    expected = [{"surah": 103, "ayah": 1}, {"surah": 103, "ayah": 2}, {"surah": 103, "ayah": 3}]
    predicted = [{"surah": 103, "ayah": 1}, {"surah": 103, "ayah": 2}, {"surah": 103, "ayah": 3}]
    s = score_sequence(expected, predicted)
    assert s["recall"] == 1.0
    assert s["precision"] == 1.0
    assert s["sequence_accuracy"] == 1.0


def test_partial_match():
    expected = [{"surah": 103, "ayah": 1}, {"surah": 103, "ayah": 2}, {"surah": 103, "ayah": 3}]
    predicted = [{"surah": 103, "ayah": 1}, {"surah": 103, "ayah": 3}]  # missed ayah 2
    s = score_sequence(expected, predicted)
    assert s["recall"] == 2 / 3
    assert s["precision"] == 1.0
    assert s["sequence_accuracy"] == 0.0


def test_wrong_verse():
    expected = [{"surah": 1, "ayah": 1}]
    predicted = [{"surah": 2, "ayah": 1}]
    s = score_sequence(expected, predicted)
    assert s["recall"] == 0.0
    assert s["precision"] == 0.0
    assert s["sequence_accuracy"] == 0.0


def test_single_verse_correct():
    expected = [{"surah": 2, "ayah": 255}]
    predicted = [{"surah": 2, "ayah": 255}]
    s = score_sequence(expected, predicted)
    assert s["recall"] == 1.0
    assert s["precision"] == 1.0
    assert s["sequence_accuracy"] == 1.0


def test_empty_predicted():
    expected = [{"surah": 1, "ayah": 1}]
    predicted = []
    s = score_sequence(expected, predicted)
    assert s["recall"] == 0.0
    assert s["precision"] == 0.0
    assert s["sequence_accuracy"] == 0.0


def test_extra_predictions():
    expected = [{"surah": 1, "ayah": 1}]
    predicted = [{"surah": 1, "ayah": 1}, {"surah": 1, "ayah": 2}]
    s = score_sequence(expected, predicted)
    assert s["recall"] == 1.0
    assert s["precision"] == 0.5
    assert s["sequence_accuracy"] == 0.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["rapidfuzz==3.14.1", "pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with rapidfuzz==3.14.1 --with pytest pytest tools/test_gold_set.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold_set import per_bar, score  # noqa: E402

def test_emphatic_substitution_counts_against_its_group_only():
    s = score([{"id": "a", "spliced": "صَبَرَ", "device": "phone"}], {"a": "سَبَرَ"})
    assert s["overall"]["canon"] == 1 / 6
    assert s["groups"]["emphatics"] == {"per": 1.0, "edits": 1, "phonemes": 1}
    assert s["groups"]["short_vowels"] == {"per": 0.0, "edits": 0, "phonemes": 3}


def test_canon_ignores_nasal_spelling_but_raw_does_not():
    s = score([{"id": "a", "spliced": "مِںكَ", "device": "phone"}], {"a": "مِنكَ"})
    assert s["overall"]["canon"] == 0
    assert s["overall"]["raw"] == 1 / 5


def test_split_by_device_and_inserted_vowel_counts_for_short_vowels():
    rows = [{"id": "a", "spliced": "بَب", "device": "phone"}, {"id": "b", "spliced": "بَب", "device": "tlog"}]
    s = score(rows, {"a": "بَب", "b": "بَبِ"})
    assert s["byDevice"]["phone"]["canon"] == 0
    assert s["byDevice"]["tlog"]["canon"] == 1 / 3
    assert s["groups"]["short_vowels"]["edits"] == 1
    assert s["clips"] == 2


def test_bar_is_the_per_where_the_fitted_metric_drops_two_points_below_the_teacher():
    pts = [(0.03, 0.90), (0.05, 0.80), (0.07, 0.70)]
    assert abs(per_bar(pts, teacher_per=0.03, drop=0.02) - 0.034) < 1e-9


def test_no_bar_when_the_metric_does_not_fall_with_per():
    assert per_bar([(0.03, 0.80), (0.05, 0.81), (0.07, 0.82)], teacher_per=0.03, drop=0.02) is None

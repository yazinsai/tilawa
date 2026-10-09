# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest"]
# ///
"""uv run --with pytest pytest tools/test_leadin_build.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leadin_build import Source, choose_lead  # noqa: E402

KEYS = ["26:121", "26:122", "26:123", "26:124", "26:125", "27:1", "27:2"]


def src(kind, id, ayahs, speaker="s1"):
    return Source(kind=kind, id=id, ayahs=ayahs, speaker=speaker)


def test_prefers_the_same_speakers_clean_clip_of_the_preceding_ayah():
    pool = [src("tlog-heldout", "t", ["26:122", "26:123"]), src("help", "other", ["26:123"], "s2"), src("help", "mine", ["26:123"])]
    assert choose_lead(KEYS, ["26:124"], "s1", pool).id == "mine"


def test_falls_back_to_another_help_speaker_then_held_out_tlog_then_tlog_train():
    assert choose_lead(KEYS, ["26:124"], "s1", [src("tlog-heldout", "t", ["26:123"]), src("help", "o", ["26:123"], "s2")]).kind == "help-other-speaker"
    assert choose_lead(KEYS, ["26:124"], "s1", [src("tlog-train", "x", ["26:123"]), src("tlog-heldout", "t", ["26:123"])]).id == "t"


def test_lead_must_end_on_the_ayah_before_and_cover_at_most_two():
    pool = [src("tlog-heldout", "late", ["26:124"]), src("tlog-heldout", "long", ["26:121", "26:122", "26:123"]), src("tlog-heldout", "gap", ["26:122"])]
    assert choose_lead(KEYS, ["26:124"], "s1", pool) is None


def test_prefers_two_ayahs_of_lead_in_over_one_within_a_source_kind():
    pool = [src("tlog-heldout", "one", ["26:123"]), src("tlog-heldout", "two", ["26:122", "26:123"])]
    assert choose_lead(KEYS, ["26:124"], "s1", pool).id == "two"


def test_no_lead_in_for_a_surahs_first_ayah():
    assert choose_lead(KEYS, ["27:1"], "s1", [src("tlog-heldout", "t", ["26:125"])]) is None

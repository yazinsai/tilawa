"""Scoring for correction-mode eval. Hand-made issues and slips; no audio."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_PATH = ROOT / "scripts" / "correction_eval.py"
_SPEC = importlib.util.spec_from_file_location("correction_eval", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
ev = importlib.util.module_from_spec(_SPEC)
sys.modules["correction_eval"] = ev
_SPEC.loader.exec_module(ev)


def _slip(**kwargs):
    base = {
        "id": "clip",
        "surah": 2,
        "ayah": 5,
        "word_index": 3,
        "kind": "omitted",
        "span_s": [1.0, 1.4],
    }
    base.update(kwargs)
    return base


def _issue(**kwargs):
    base = {
        "surah": 2,
        "ayah": 5,
        "word": 3,
        "wordIndex": 99,
        "kind": "possible_omission",
        "atSeconds": 1.5,
    }
    base.update(kwargs)
    return base


def test_catch_uses_ayah_word_not_global_index():
    hit = ev.match_slip(_slip(), [_issue(word=4, wordIndex=0)])
    assert hit["caught"] is True
    assert hit["exact_word"] is False
    assert hit["kind_correct"] is True
    assert hit["latency_s"] == 0.5
    # wordIndex is ignored: a global index of 3 does not catch when `word` is two away.
    miss = ev.match_slip(_slip(), [_issue(word=5, wordIndex=3)])
    assert miss == {"caught": False, "kind_correct": False, "exact_word": False, "latency_s": None}


def test_different_ayah_does_not_catch():
    miss = ev.match_slip(_slip(), [_issue(ayah=6)])
    assert miss["caught"] is False
    miss = ev.match_slip(_slip(), [_issue(surah=3)])
    assert miss["caught"] is False


def test_kind_correct_mapping():
    same = _issue()
    omitted = _slip(kind="omitted")
    assert ev.match_slip(omitted, [_issue(kind="possible_substitution")])["caught"] is True
    assert ev.match_slip(omitted, [_issue(kind="possible_substitution")])["kind_correct"] is False
    assert ev.match_slip(omitted, [same])["kind_correct"] is True

    substituted = _slip(kind="substituted")
    assert ev.match_slip(substituted, [_issue(kind="possible_substitution")])["kind_correct"] is True
    assert ev.match_slip(substituted, [_issue(kind="possible_vowel")])["kind_correct"] is False
    assert ev.match_slip(substituted, [_issue(kind="possible_omission")])["kind_correct"] is False

    for slip_kind in ("repeated", "restarted"):
        slip = _slip(kind=slip_kind)
        for kind in ("possible_omission", "possible_substitution", "possible_vowel", "unclear_ayah"):
            assert ev.match_slip(slip, [_issue(kind=kind)])["kind_correct"] is True
        skipped = ev.match_slip(slip, [_issue(kind="possible_skipped_ayah", word=3)])
        assert skipped["caught"] is True
        assert skipped["kind_correct"] is False


def test_exact_word_and_earliest_latency():
    slip = _slip(span_s=[2.0, 2.4])
    issues = [
        _issue(word=4, kind="possible_vowel", atSeconds=3.5),
        _issue(word=3, kind="possible_omission", atSeconds=2.25),
    ]
    hit = ev.match_slip(slip, issues)
    assert hit["caught"] and hit["exact_word"] and hit["kind_correct"]
    assert hit["latency_s"] == 0.25
    # No span: still a catch, latency stays unset.
    bare = ev.match_slip(_slip(span_s=None), [_issue()])
    assert bare["caught"] is True and bare["latency_s"] is None


def test_score_slips_groups_kinds_and_skips_missing_clips():
    slips = [
        _slip(id="a", kind="omitted", span_s=[1.0, 2.0]),
        _slip(id="b", surah=1, ayah=2, word_index=0, kind="substituted", span_s=[0.0, 1.0]),
        _slip(id="missing", kind="repeated"),
    ]
    issues = {
        "a": [_issue(surah=2, ayah=5, word=3, kind="possible_omission", atSeconds=1.2)],
        "b": [],
    }
    scored = ev.score_slips(slips, issues)
    assert scored["unscored"] == 1
    assert scored["all"]["n"] == 2
    assert scored["all"]["recall"] == 0.5
    assert scored["all"]["kind_recall"] == 0.5
    assert scored["all"]["exact_rate"] == 0.5
    assert abs(scored["all"]["median_latency_s"] - 0.2) < 1e-9
    assert scored["by_kind"]["omitted"]["recall"] == 1
    assert scored["by_kind"]["substituted"]["n"] == 1
    assert scored["by_kind"]["substituted"]["recall"] == 0
    assert scored["by_kind"]["substituted"]["median_latency_s"] is None


def test_false_flags_per_clean_minute_and_slices():
    clips = [
        {"id": "a", "duration_s": 60, "issues": [{}, {}], "device": "phone", "gender": "male",
         "level": "beginner", "ayah_span": "single", "split": "dev"},
        {"id": "b", "duration_s": 60, "issues": [], "device": "laptop", "gender": None,
         "level": None, "ayah_span": "multi", "split": "test"},
        {"id": "c", "duration_s": 30, "issues": [{}], "device": "phone", "gender": "female",
         "level": "hafiz", "ayah_span": "single", "split": "dev"},
    ]
    stats = ev.false_flag_stats(clips)
    assert stats["issues"] == 3
    assert stats["minutes"] == 2.5
    assert abs(stats["per_minute"] - 1.2) < 1e-9
    assert stats["flagged_clips"] == 2
    assert stats["clip_fraction"] == 2 / 3
    slices = ev.slice_false_flags(clips)
    assert slices["device"]["phone"]["per_minute"] == 2.0
    assert slices["device"]["phone"]["minutes"] == 1.5
    assert slices["gender"]["unknown"]["issues"] == 0
    assert slices["gender"]["unknown"]["clips"] == 1
    assert slices["ayah_span"]["multi"]["clip_fraction"] == 0
    assert slices["level"]["hafiz"]["issues"] == 1
    assert slices["split"]["dev"]["issues"] == 3
    assert ev.false_flag_stats([])["per_minute"] is None


def test_edge_words_cannot_have_both_neighbours():
    assert ev.position_class(0, 10) == "edge"
    assert ev.position_class(9, 10) == "edge"
    assert ev.position_class(4, 10) == "middle"
    assert ev.position_class(1, 2) == "edge"
    assert ev.position_class(0, 1) == "edge"
    assert ev.position_class(1, None) == "unknown"
    slips = [
        _slip(id="e", word_index=0, n_words=8, span_s=[0.0, 0.2]),
        _slip(id="m", word_index=3, n_words=8, span_s=[1.0, 1.2]),
    ]
    issues = {
        "e": [],
        "m": [_issue(word=3, kind="possible_omission", atSeconds=1.1)],
    }
    pos = ev.position_recall(slips, issues, n_words_of=None)
    assert pos["edge"]["recall"] == 0
    assert pos["middle"]["recall"] == 1


def test_substituted_sample_is_seeded_and_keeps_every_other_kind():
    rows = [
        {"id": f"s{i:02d}", "kind": "substituted", "surah": 1, "ayah": 1, "word_index": i}
        for i in range(10)
    ]
    rows.append({"id": "o", "kind": "omitted", "surah": 1, "ayah": 2, "word_index": 0})
    rows.append({"id": "r", "kind": "repeated", "surah": 1, "ayah": 3, "word_index": 1})
    rows.append({"id": "z", "kind": "restarted", "surah": 1, "ayah": 4, "word_index": 0})
    first = ev.select_tlog_slips(rows, n_sub=3, seed=0)
    second = ev.select_tlog_slips(list(reversed(rows)), n_sub=3, seed=0)

    def _subs(picked):
        return [row["id"] for row in picked if row["kind"] == "substituted"]

    assert _subs(first) == _subs(second)
    assert {row["id"] for row in first if row["kind"] != "substituted"} == {"o", "r", "z"}
    kinds = [row["kind"] for row in first]
    assert kinds.count("omitted") == 1
    assert kinds.count("repeated") == 1
    assert kinds.count("restarted") == 1
    assert kinds.count("substituted") == 3
    other = ev.select_tlog_slips(rows, n_sub=3, seed=1)
    assert [row["id"] for row in first if row["kind"] == "substituted"] != [
        row["id"] for row in other if row["kind"] == "substituted"
    ]

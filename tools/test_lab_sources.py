# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pytest", "rapidfuzz==3.14.1"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with numpy --with pytest --with rapidfuzz==3.14.1 pytest tools/test_lab_sources.py"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_sources as L  # noqa: E402

WORDS = L.surah_words(L.reference())


def test_pool_required_size_and_determinism():
    timed = list(range(1, 115))
    a = L.pool(3, timed, WORDS)
    assert a == L.pool(3, timed, WORDS)
    assert set(L.REQUIRED) <= set(a) and len(a) == L.POOL and a == sorted(set(a))
    assert sum(160 <= WORDS[s] < 450 for s in a if s not in L.REQUIRED) >= 8
    assert all(WORDS[s] >= L.MIN_WORDS for s in a)
    assert len({L.juz_of(s) for s in a}) >= 12
    assert L.pool(6, timed, WORDS) != a


def test_pool_only_timed():
    timed = [s for s in range(1, 115) if s not in (26, 54)]
    a = L.pool(7, timed, WORDS)
    assert set(a) <= set(timed) and {55, 77} <= set(a) and len(a) == L.POOL


def test_juz_of():
    assert [L.juz_of(s) for s in (1, 2, 3, 18, 55, 67, 77, 78, 114)] == [1, 1, 3, 15, 27, 29, 29, 30, 30]


def rows(*spans, pre=None):
    out = [{"ayah": 0, "start_time": pre[0], "end_time": pre[1]}] if pre else []
    return out + [{"ayah": i + 1, "start_time": s, "end_time": e, "polygon": ""} for i, (s, e) in enumerate(spans)]


def test_parse_timing():
    ayat, pre, bad = L.parse_timing(rows((1000, 2500), (2500, 4000), pre=(0, 1000)), 2)
    assert pre == {"startS": 0.0, "endS": 1.0} and not bad
    assert ayat == [{"ayah": 1, "startS": 1.0, "endS": 2.5}, {"ayah": 2, "startS": 2.5, "endS": 4.0}]
    assert L.parse_timing(rows((1000, 2500)), 2)[2]
    assert L.parse_timing(rows((1000, 2600), (2500, 4000)), 2)[2]
    assert L.parse_timing(rows((1000, 1000), (1000, 4000)), 2)[2]
    assert not L.parse_timing(rows((1000, 2500), (2500, 400_000)), 2, 450.0)[2]
    assert L.parse_timing(rows((1000, 2500), (2500, 97_000)), 2, 519.0)[2] == ["timing ends at 97 s of 519 s audio"]
    assert L.parse_timing(rows((1000, 2500), (2500, 97_000)), 2, 90.0)[2]


@pytest.mark.parametrize("seconds", [3, 30, 31, 55.5, 3600.3])
def test_windows_cover_every_frame_once(seconds):
    n = int(seconds * L.SR)
    plan = L.windows(n)
    assert plan[0][0] == 0 and all(s % L.FRAME == 0 for s, _, _ in plan)
    assert plan[-1][0] + L.WINDOW_S * L.SR >= n
    covered = []
    for s, lo, hi in plan:
        local = min(L.WINDOW_S * L.SR, n - s) // L.FRAME
        hi = local if hi < 0 else hi
        assert lo < hi <= local
        covered += range(s // L.FRAME + lo, s // L.FRAME + hi)
    assert covered == list(range(len(covered))) and len(covered) >= n // L.FRAME - len(plan)


def test_greedy():
    ids = np.array([0, 3, 3, 0, 3, 5, 5, 0])
    conf = np.array([0.9, 0.5, 0.8, 0.9, 0.6, 0.7, 0.4, 0.9])
    assert L.greedy(ids, conf) == [(3, 1, 2, 0.8), (3, 4, 4, 0.6), (5, 5, 6, 0.7)]


def test_word_spans_with_merged_word():
    spans = [(0, 2), (3, 4), (6, 8), (9, 10)]
    tp = [0.9, 0.7, 0.5, 0.3]
    assert L.word_spans([0, 2, 2], spans, tp) == [(0, 4, pytest.approx(0.8)), (4, 4, None), (6, 10, pytest.approx(0.4))]


def test_mismatches_and_verdict():
    ayat = [{"ayah": 1, "startS": 1.0, "endS": 3.0}, {"ayah": 2, "startS": 3.0, "endS": 5.0}, {"ayah": 3, "startS": 5.0, "endS": 7.0}]
    assert L.mismatches(ayat, [(1.2, 2.7), (3.3, 4.6), (5.2, 6.9)]) == [0, 0, 0]
    # ayah 2's speech starts 0.5 s before its window: both ayah 1 (holds foreign speech) and ayah 2 (cut off) disagree.
    m = L.mismatches(ayat, [(1.2, 2.4), (2.5, 4.6), (5.2, 6.9)])
    assert m[0] == pytest.approx(0.5) and m[1] == pytest.approx(0.5) and m[2] == 0
    assert L.mismatches(ayat, [(1.2, 2.4), None, (5.2, 6.9)]) == [0, float("inf"), 0]
    assert L.verdict(0.29, 0.1, 0.5) == []
    assert L.verdict(0.31, 0.2, 0.1) == ["window-mismatch 0.31s", "per 0.20", "word-score 0.10"]
    assert L.verdict(float("inf"), None, None) == ["align-failed"]


def test_cut_finds_quiet_gap():
    x = np.concatenate([np.full(L.SR // 2, 0.5), np.full(L.SR // 10, 0.001), np.full(L.SR // 2, 0.5)]).astype(np.float32)
    db = L.rms_db(x)
    t, d = L.cut(db, 0.49, 0.61)
    assert 0.5 <= t <= 0.6 and d < -50


def test_garbage_blank():
    lp = np.log(np.array([[0.9, 0.05, 0.05], [1e-6, 0.01, 0.99]]))
    out = L.garbage(lp)
    assert out[0, 0] == lp[0, 0] and out[1, 0] == pytest.approx(np.log(0.99) + L.GARBAGE_LOGP) and (out[:, 1:] == lp[:, 1:]).all()


def test_collapse_keeps_word_starts():
    # Two words; the second starts inside a run of the same phoneme (a merged noon), so the run splits there.
    assert L.collapse("aabbnnnncc", [0, 6]) == ("abnnc", [0, 3])
    assert L.collapse("abc", [0, 3]) == ("abc", [0, 3])
    assert L.squeeze("رَحمَاااان") == "رَحمَان" and L.per("رَحمَاان", "رَحمَاااان") == 0


def test_extend_over_repeat_spikes():
    top1 = np.array([0, 5, 0, 5, 0, 5, 0, 7, 7, 0])
    assert L.extend([(1, 2), (7, 9)], [5, 7], top1, 10) == [(1, 6), (7, 9)]
    far = np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 5])
    assert L.extend([(1, 2)], [5], far, 10) == [(1, 2)]

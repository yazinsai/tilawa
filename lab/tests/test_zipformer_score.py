import math

import numpy as np
import pytest

from shared.zipformer_score import (
    BLANK,
    bucket,
    ctc_forced_logprob,
    edit_counts,
    greedy_ids,
    score_clip,
)


def _lp(path, v=251, p=0.97):
    """Log-probs that put mass p on each frame's token in `path`."""
    lp = np.full((len(path), v), math.log((1 - p) / (v - 1)), dtype=np.float64)
    for t, u in enumerate(path):
        lp[t, u] = math.log(p)
    return lp


def test_greedy_collapses_repeats_and_blanks():
    assert greedy_ids(_lp([5, 5, BLANK, 5, 7, BLANK, BLANK, 7])) == [5, 5, 7, 7]


def test_edit_counts_split():
    ec = edit_counts([1, 2, 3, 4], [1, 9, 3, 4, 5])
    assert (ec.sub, ec.ins, ec.dele, ec.ref_len) == (1, 1, 0, 4)
    ec = edit_counts([1, 2, 3, 4], [1, 4])
    assert (ec.sub, ec.ins, ec.dele) == (0, 0, 2)
    assert ec.per == pytest.approx(0.5)


def test_forced_logprob_prefers_true_label():
    lp = _lp([BLANK, 3, 3, BLANK, 4, BLANK])
    good = ctc_forced_logprob(lp, [3, 4])
    bad = ctc_forced_logprob(lp, [3, 5])
    assert good > bad
    assert good <= 0


def test_forced_logprob_matches_bruteforce_small():
    rng = np.random.default_rng(0)
    v, T = 4, 5
    logits = rng.normal(size=(T, v))
    lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    blank = 3
    ref = [1, 1]

    def collapse(path):
        out, prev = [], None
        for u in path:
            if u != blank and u != prev:
                out.append(u)
            prev = u
        return out

    import itertools

    tot = -np.inf
    for path in itertools.product(range(v), repeat=T):
        if collapse(path) == ref:
            tot = np.logaddexp(tot, sum(lp[t, u] for t, u in enumerate(path)))
    assert ctc_forced_logprob(lp, ref, blank=blank) == pytest.approx(tot, abs=1e-9)


def test_forced_infeasible_when_too_short():
    assert ctc_forced_logprob(_lp([1, 2]), [1, 2, 3, 4]) == -np.inf


@pytest.mark.parametrize("per,b", [(0.0, "clean"), (0.10, "clean"), (0.2, "suspect"), (0.35, "suspect"), (0.36, "mislabel")])
def test_bucket_thresholds(per, b):
    assert bucket(per) == b


def test_score_clip_json_roundtrip():
    lp = _lp([BLANK, 3, BLANK, 4, BLANK])
    s = score_clip(lp, [3, 4], clip_id="x", surah=1, ayah=2, duration=1.0).to_json()
    assert s["per"] == 0 and s["bucket"] == "clean" and s["frames"] == 5
    assert s["alt_per"] is None


def test_viterbi_spans_follow_the_peaks():
    from shared.zipformer_score import ctc_viterbi_spans

    lp = _lp([BLANK, 3, 3, BLANK, BLANK, 4, BLANK, 4, 4, BLANK])
    assert ctc_viterbi_spans(lp, [3, 4, 4]) == [(1, 2), (5, 5), (7, 8)]
    assert ctc_viterbi_spans(_lp([1, 2]), [1, 2, 3, 4]) is None

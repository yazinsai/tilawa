"""Training-only relative phone timing views for a frozen listener.

CTC forward/backward retains a distribution over occurrence boundaries. Only
the conservative interior common to that distribution can be stretched; blank
intervals, vowels, uncertain boundaries and recording edges are copied. This
does not align or transform inference audio.
"""
import hashlib
import numpy as np

CONSONANTS = frozenset("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي")
FRAME_SAMPLES = 640  # The installed 40 ms / 16 kHz output clock.


def speaker_partition(speakers):
    if any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Every recording needs an identified training speaker")
    speakers = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(speakers) < 5:
        raise ValueError("Need at least five training speakers for grouped selection")
    return {s: "selection" if i % 5 == 0 else "fit" for i, s in enumerate(speakers)}


def admitted_none(labels, excluded=()):
    excluded = set(excluded) | {r.get("speaker") for r in labels if r.get("split") != "train"}
    selected, seen = [], set()
    for r in labels:
        if r.get("split") != "train" or r.get("kind") != "none":
            continue
        if not r.get("speaker") or r["speaker"] in excluded:
            continue
        if (r.get("scored") is not True or any(r.get(k) != [] for k in
                ("events", "non_errors", "unsupported")) or r.get("excluded") or r.get("exclude")):
            continue
        if not r.get("id") or not r.get("wav") or not r.get("ayahs"):
            raise ValueError("None take lacks audio identity or canonical transcript")
        if r["id"] in seen:
            raise ValueError("Duplicate none take")
        seen.add(r["id"])
        selected.append(r)
    return selected


def occupancy(log_probs, target):
    """Return CTC state, entry and last-occupation marginals, plus log Z.

    Columns distinguish transcript occurrences, including repeated labels. No
    hard Viterbi path or argmax timing is substituted for boundary uncertainty.
    """
    lp, target = np.asarray(log_probs, dtype=np.float64), np.asarray(target)
    if (lp.ndim != 2 or not np.isfinite(lp).all() or target.ndim != 1 or not len(target)
            or not np.issubdtype(target.dtype, np.integer) or np.any(target <= 0)
            or np.any(target >= lp.shape[1])):
        raise ValueError("Need finite full-vocabulary evidence and nonblank target ids")
    if len(lp) < len(target) + int(np.sum(target[1:] == target[:-1])):
        raise ValueError("Impossible CTC transcript")
    states = np.zeros(2 * len(target) + 1, dtype=np.int64)
    states[1::2] = target
    predecessors = []
    for s, label in enumerate(states):
        predecessors.append([s] + ([s - 1] if s else []) +
                            ([s - 2] if s > 1 and label and label != states[s - 2] else []))
    successors = [[j for j, pred in enumerate(predecessors) if s in pred] for s in range(len(states))]
    alpha = np.full((len(lp), len(states)), -np.inf)
    alpha[0, :2] = lp[0, states[:2]]
    for t in range(1, len(lp)):
        for s, pred in enumerate(predecessors):
            alpha[t, s] = np.logaddexp.reduce(alpha[t - 1, pred]) + lp[t, states[s]]
    z = np.logaddexp.reduce(alpha[-1, -2:])
    if not np.isfinite(z):
        raise ValueError("No finite CTC path")
    beta = np.full_like(alpha, -np.inf)
    beta[-1, -2:] = 0.
    for t in range(len(lp) - 2, -1, -1):
        for s, succ in enumerate(successors):
            beta[t, s] = np.logaddexp.reduce(lp[t + 1, states[succ]] + beta[t + 1, succ])
    entry, exit = np.zeros_like(alpha), np.zeros_like(alpha)
    entry[0, 1] = np.exp(alpha[0, 1] + beta[0, 1] - z)
    for s in range(1, len(states), 2):
        pred = [p for p in predecessors[s] if p != s]
        succ = [p for p in successors[s] if p != s]
        for t in range(1, len(lp)):
            entry[t, s] = np.exp(np.logaddexp.reduce(alpha[t - 1, pred]) +
                                 lp[t, states[s]] + beta[t, s] - z)
        for t in range(len(lp) - 1):
            exit[t, s] = np.exp(alpha[t, s] + np.logaddexp.reduce(
                lp[t + 1, states[succ]] + beta[t + 1, succ]) - z)
        if s == len(states) - 2:
            exit[-1, s] = np.exp(alpha[-1, s] - z)
    return np.exp(alpha + beta - z), entry[:, 1::2], exit[:, 1::2], float(z)


def safe_intervals(log_probs, phones, target, samples):
    """90% onset / 10% offset cores, leaving 10 ms boundary guards.

    The constants specify a conservative training alignment policy, never an
    inference tolerance. Short or diffuse cores stay identity views.
    """
    if len(phones) != len(target) or samples <= 0:
        raise ValueError("Phone occurrences or audio length differ")
    _, entry, exit, z = occupancy(log_probs, target)
    result = []
    for i, phone in enumerate(phones):
        if phone not in CONSONANTS:
            continue
        def quantile(p, q):
            if not np.isclose(p.sum(), 1., atol=1e-6):
                raise ValueError("Incomplete occurrence boundary distribution")
            return int(np.searchsorted(np.cumsum(p / p.sum()), q))
        start = quantile(entry[:, i], .9) * FRAME_SAMPLES + 160
        end = min(samples, (quantile(exit[:, i], .1) + 1) * FRAME_SAMPLES) - 160
        if end - start >= 640:
            result.append(dict(occurrence=i, phone=phone, start=start, end=end))
    if any(a["end"] > b["start"] for a, b in zip(result, result[1:])):
        raise ValueError("Uncertain phone interiors overlap")
    return result, z


def wsola(wave, samples):
    """Pitch-preserving waveform-similarity overlap/add at an exact length.

    Source samples are copied into 20 ms grains, shifted by correlation rather
    than resampled. Identity-length synthesis takes the same overlap/add path.
    The first and final grains stay at the original recording boundaries.
    """
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not np.isfinite(wave).all() or len(wave) < 320 or samples < 320:
        raise ValueError("WSOLA needs finite audio and at least one full grain")
    width, hop, search = 320, 80, 80
    positions = list(range(0, samples - width + 1, hop))
    if positions[-1] != samples - width:
        positions.append(samples - width)
    output, weights = np.zeros(samples, dtype=np.float64), np.zeros(samples)
    window = np.maximum(np.hanning(width), 1e-3)
    previous = -1
    for j, pos in enumerate(positions):
        expected = int(round(pos * (len(wave) - width) / max(1, samples - width)))
        source = expected
        if samples != len(wave) and j not in (0, len(positions) - 1):
            overlap = min(width, max(0, positions[j - 1] + width - pos))
            if overlap:
                reference = output[pos:pos + overlap] / np.maximum(weights[pos:pos + overlap], 1e-12)
                lo, hi = max(previous, 0, expected - search), min(len(wave) - width, expected + search)
                candidates = np.arange(lo, hi + 1)
                if len(candidates):
                    grains = np.lib.stride_tricks.sliding_window_view(wave, overlap)[candidates]
                    scores = grains @ reference / np.maximum(1e-12, np.linalg.norm(grains, axis=1))
                    source = int(candidates[np.argmax(scores)])
        output[pos:pos + width] += wave[source:source + width] * window
        weights[pos:pos + width] += window
        previous = source
    return (output / weights).astype(np.float32)


def duration_view(wave, intervals, factors):
    """Copy every protected sample; replace only enrolled consonant cores.

    The caller supplies explicit speech islands for the separate global control.
    """
    wave = np.asarray(wave, dtype=np.float32)
    if len(intervals) != len(factors) or wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError("Need one finite factor per nonoverlapping core")
    pieces, mapping, cursor, dest = [], [], 0, 0
    for span, factor in zip(intervals, factors):
        start, end = span["start"], span["end"]
        if not cursor <= start < end <= len(wave) or not np.isfinite(factor) or factor <= 0:
            raise ValueError("Invalid duration interval or factor")
        pieces.append(wave[cursor:start].copy())
        dest += start - cursor
        size = int(round((end - start) * factor))
        altered = wsola(wave[start:end], size)
        pieces.append(altered)
        mapping.append(dict(span, outputStart=dest, outputEnd=dest + size, factor=float(factor)))
        dest += size
        cursor = end
    pieces.append(wave[cursor:].copy())
    return np.concatenate(pieces), mapping


def speech_islands(log_probs, samples):
    """Keep >= 200 ms confidently blank runs out of the global control.

    Only training audio is used. Uncertainty is retained as speech, avoiding
    removal of weak phones; these islands are not transcript boundaries.
    """
    blank = np.exp(np.asarray(log_probs)[:, 0]) >= .95
    cuts, start = [], None
    for t, is_blank in enumerate(np.r_[blank, False]):
        if is_blank and start is None:
            start = t
        if not is_blank and start is not None:
            if t - start >= 5:
                cuts.append((start * FRAME_SAMPLES, min(samples, t * FRAME_SAMPLES)))
            start = None
    cursor, islands = 0, []
    for start, end in cuts:
        if start - cursor >= 640:
            islands.append(dict(start=cursor, end=start))
        cursor = end
    if samples - cursor >= 640:
        islands.append(dict(start=cursor, end=samples))
    return islands


def global_view(wave, islands, intervals, factors):
    """Matched tempo control: same factors and net length within each island.

    Unlike the articulation arm this modifies vowel durations too. Consonant
    cores crossing a protected pause boundary are not changed in either arm.
    """
    if len(intervals) != len(factors) or any(not np.isfinite(f) or f <= 0 for f in factors):
        raise ValueError("Need one finite factor per consonant core")
    if any(sum(a["start"] <= s["start"] and s["end"] <= a["end"] for a in islands) != 1
           for s in intervals):
        raise ValueError("Each core must lie in exactly one speech island")
    spans, ratios = [], []
    for island in islands:
        delta = sum(int(round((s["end"] - s["start"]) * f)) - (s["end"] - s["start"])
                    for s, f in zip(intervals, factors)
                    if island["start"] <= s["start"] and s["end"] <= island["end"])
        spans.append(island)
        ratios.append((island["end"] - island["start"] + delta) / (island["end"] - island["start"]))
    return duration_view(wave, spans, ratios)

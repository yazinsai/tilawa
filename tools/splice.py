"""Phoneme splicing on 16 kHz mono audio: synthetic recitation errors cut from clean recordings.

Segments come from teacher CTC forced alignment (`e1_report.viterbi`), turned into contiguous per-phoneme
sample ranges by `boundaries`. Every cut snaps to a zero crossing and is joined with a short raised-cosine
crossfade, so an error splice and its control splice carry the same kind of seam.
"""

import numpy as np

SR = 16_000
FADE_S = 0.008
SNAP_S = 0.005
# Donor loudness is matched to the segment it replaces, within this factor.
MAX_GAIN = 2.0


def boundaries(spans: list[tuple[int, int]], n_frames: int, n_samples: int) -> list[tuple[int, int]]:
    """Contiguous sample range [start, end) per token from CTC frame spans; each blank gap splits at its midpoint."""
    per = n_samples / n_frames
    cuts = [spans[0][0]] + [(spans[k][1] + spans[k + 1][0]) / 2 for k in range(len(spans) - 1)] + [spans[-1][1]]
    s = [round(c * per) for c in cuts]
    return list(zip(s[:-1], s[1:]))


def snap(x: np.ndarray, i: int) -> int:
    """The zero crossing nearest i within SNAP_S, else i."""
    r = int(SNAP_S * SR)
    lo, hi = max(1, i - r), min(len(x) - 1, i + r)
    if hi <= lo:
        return i
    w = x[lo - 1 : hi + 1]
    z = np.nonzero(w[:-1] * w[1:] <= 0)[0] + lo
    return int(z[np.argmin(np.abs(z - i))]) if len(z) else i


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def join(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """x then y, overlapping the last n samples of x with the first n of y."""
    n = min(n, len(x), len(y))
    if n == 0:
        return np.concatenate([x, y])
    fade_in = (0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)).astype(np.float32)
    return np.concatenate([x[: len(x) - n], x[len(x) - n :] * (1 - fade_in) + y[:n] * fade_in, y[n:]])


def replace(host: np.ndarray, a: int, b: int, donor: np.ndarray, d0: int, d1: int, fade_s: float = FADE_S) -> np.ndarray:
    """host with samples [a, b) replaced by donor [d0, d1), loudness-matched and crossfaded at both seams."""
    a, b, d0, d1 = snap(host, a), snap(host, b), snap(donor, d0), snap(donor, d1)
    h = int(fade_s * SR) // 2
    h = min(h, a, len(host) - b, d0, len(donor) - d1)
    seg = donor[d0 - h : d1 + h]
    src, dst = rms(donor[d0:d1]), rms(host[a:b])
    if src > 0 and dst > 0:
        seg = seg * float(np.clip(dst / src, 1 / MAX_GAIN, MAX_GAIN))
    return join(join(host[: a + h], seg.astype(np.float32), 2 * h), host[b - h :], 2 * h)


def shorten(x: np.ndarray, a: int, b: int, keep: int, fade_s: float = FADE_S) -> np.ndarray:
    """x with the segment [a, b) cut down to about keep samples by removing its middle."""
    if b - a <= keep:
        return x
    mid = (a + b) // 2
    lo, hi = snap(x, mid - (b - a - keep) // 2), snap(x, mid + (b - a - keep + 1) // 2)
    h = min(int(fade_s * SR) // 2, lo, len(x) - hi)
    return join(x[: lo + h], x[hi - h :], 2 * h)

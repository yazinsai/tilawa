# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with numpy --with pytest pytest tools/test_splice.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splice import SR, boundaries, replace, shorten  # noqa: E402


def tone(hz: float, seconds: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return np.sin(2 * np.pi * hz * t + phase).astype(np.float32)


def peak_hz(x: np.ndarray) -> float:
    return float(np.argmax(np.abs(np.fft.rfft(x))) * SR / len(x))


def test_replace_swaps_the_segment_and_keeps_length():
    host = tone(200, 1.0)
    donor = tone(700, 1.0, phase=1.0)
    out = replace(host, 4000, 8000, donor, 2000, 6000)
    assert abs(len(out) - len(host)) < 0.004 * SR
    assert abs(peak_hz(out[5000:7000]) - 700) < 20
    assert np.allclose(out[:3000], host[:3000])
    assert np.allclose(out[-3000:], host[-3000:])


def test_replace_has_no_click_at_the_seams():
    host = tone(200, 1.0)
    donor = tone(700, 1.0, phase=1.0)
    out = replace(host, 4000, 8000, donor, 2000, 6000)
    # A hard cut between these tones jumps by up to ~1.0 in one sample; a crossfade stays smooth.
    assert np.max(np.abs(np.diff(out))) < 0.35


def test_shorten_cuts_the_middle_of_a_hold_and_keeps_its_onset_and_release():
    # 0.3 s onset at 300 Hz, a 0.4 s hold at 500 Hz, then 0.3 s release at 900 Hz.
    x = np.concatenate([tone(300, 0.3), tone(500, 0.4), tone(900, 0.3)])
    a, b = int(0.3 * SR), int(0.7 * SR)
    out = shorten(x, a, b, int(0.1 * SR))
    assert abs(len(out) - (len(x) - int(0.3 * SR))) < 0.012 * SR
    assert abs(peak_hz(out[: int(0.25 * SR)]) - 300) < 20
    assert abs(peak_hz(out[-int(0.25 * SR) :]) - 900) < 20
    assert np.max(np.abs(np.diff(out))) < 0.4


def test_boundaries_split_blank_gaps_at_their_midpoint():
    # 20 frames over 6400 samples: 320 samples per frame.
    spans = [(2, 3), (6, 8), (10, 11)]
    assert boundaries(spans, 20, 6400) == [(640, 1440), (1440, 2880), (2880, 3520)]


def test_shorten_never_lengthens():
    x = tone(300, 1.0)
    assert len(shorten(x, 4000, 6000, 3000)) == len(x)
    # A cut shorter than the zero-crossing snap radius must not snap into a reversed range.
    noise = np.random.default_rng(0).standard_normal(SR).astype(np.float32)
    for cut in range(1, 200, 7):
        assert len(shorten(noise, 4000, 6000, 2000 - cut)) <= len(noise)


def test_replace_of_a_tiny_segment_stays_ordered():
    noise = np.random.default_rng(1).standard_normal(SR).astype(np.float32)
    for w in range(1, 200, 7):
        out = replace(noise, 4000, 4000 + w, noise, 9000, 9000 + w)
        assert abs(len(out) - len(noise)) <= 2 * 80 + w

"""Pure helpers for the frozen-encoder slip-head probe.

Nothing here reads audio or writes per-clip rows. Callers print aggregates only.
"""
from __future__ import annotations

import numpy as np

# Trained as positives. tajweed is out of the recall denominator (the engine
# does not grade it) but it is still a real slip the head can see.
HEAD_KINDS = frozenset({"substitution", "vowel", "tajweed", "skip_word"})
IN_SCOPE = ("skip_word", "substitution", "vowel", "skip_ayah", "repeat")
ERR = frozenset({
    "possible_omission", "possible_substitution", "possible_vowel",
    "possible_skipped_ayah", "unclear_ayah",
})
# Operating points are additional false flags per clean minute on the
# out-of-fold calibration pool. Chosen before any test score.
OPS_PER_MIN = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25)
# Pre-declared heads. Test is scored once for each (encoder, head, synth).
HEADS = ("logistic", "mlp", "frame")
ENCODERS = ("shipped", "a0w")

# Harakat and the phoneme-vocab vowel marks (madd, small yeh, etc.).
_VOWEL = set("َُِّْٰٕٓٔ") | set(chr(c) for c in (
    list(range(0x064B, 0x0653)) + [0x0670, 0x0640] + list(range(0x06D6, 0x06EE))
    + [0x0653, 0x0654, 0x0655, 0x0656, 0x0657, 0x0658]
))
_VOWEL |= set("ٲڇںۜۥۦ۪۾ـ")


def skeleton(phonemes: str) -> str:
    """Consonant skeleton: drop vowels so a harakat change compares equal."""
    return "".join(ch for ch in phonemes if ch not in _VOWEL)


def units_from_track(track) -> list[dict]:
    """Tracker chars -> word spans in encoder frames.

    A word the tracker skipped sits in the gap between its neighbours. The gap
    is split across those words so they do not all share one score.
    """
    words: dict[tuple[int, int, int], list[int]] = {}
    for item in track or []:
        if item is None or len(item) < 4:
            continue
        if len(item) >= 5:
            _ch, fr, surah, ayah, w = item[:5]
        else:
            _ch, fr, ayah, w = item[:4]
            surah = -1
        key = (int(surah), int(ayah), int(w))
        words.setdefault(key, []).append(int(fr))
    order = sorted(words, key=lambda k: min(words[k]))
    out: list[dict] = []
    for i, key in enumerate(order):
        fs = words[key]
        out.append({
            "surah": key[0], "ayah": key[1], "word": key[2],
            "a": int(min(fs)), "b": int(max(fs)) + 1, "skipped": False,
        })
        if i + 1 >= len(order):
            continue
        nxt = order[i + 1]
        if nxt[0] != key[0] or nxt[1] != key[1] or nxt[2] <= key[2] + 1:
            continue
        nskip = nxt[2] - key[2] - 1
        gap_a = int(max(fs)) + 1
        gap_b = int(min(words[nxt]))
        span = max(nskip, gap_b - gap_a)
        for j, w in enumerate(range(key[2] + 1, nxt[2])):
            a = gap_a + (span * j) // nskip
            b = gap_a + (span * (j + 1)) // nskip
            if b <= a:
                b = a + 1
            out.append({
                "surah": key[0], "ayah": key[1], "word": w,
                "a": a, "b": b, "skipped": True,
            })
    return out


def pool_span(enc: np.ndarray, a: int, b: int, pad: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, max, and the frame slice. ±pad frames, matching what a word pool sees."""
    n = len(enc)
    if n == 0:
        z = np.zeros(512, np.float32)
        return z, z.copy(), np.zeros((0, 512), np.float32)
    a = max(0, int(a) - pad)
    b = min(n, max(a + 1, int(b) + pad))
    sl = np.asarray(enc[a:b], dtype=np.float32)
    return sl.mean(0), sl.max(0), sl


def hits_label(surah: int, ayah: int, word: int, lab: dict | None, tol: int = 1) -> bool:
    """Any-kind catch: same surah:ayah and word within ±tol. skip_ayah is ayah-level."""
    if not lab or not lab.get("mapped"):
        return False
    if lab.get("surah") is not None and int(surah) != int(lab["surah"]):
        return False
    if int(ayah) != int(lab["ayah"]):
        return False
    if lab.get("label_kind") == "skip_ayah":
        return True
    if lab.get("word_index") is None:
        return False
    return abs(int(word) - int(lab["word_index"])) <= tol


def k_for_rate(rate_per_min: float, minutes: float) -> int:
    """How many extra clean flags a per-minute budget allows (floor)."""
    if rate_per_min <= 0 or minutes <= 0:
        return 0
    return int(np.floor(rate_per_min * minutes + 1e-9))


def threshold_at_k(scores_desc: np.ndarray, k: int) -> float:
    """Score cutoff so at most the top-k calibration words flag, if scores are unique."""
    if len(scores_desc) == 0:
        return 1.0
    if k <= 0:
        return float(scores_desc[0]) + 1e-6
    if k >= len(scores_desc):
        return float(scores_desc[-1])
    return float(scores_desc[k]) + 1e-9


def fit_splice(host: np.ndarray, a0: int, a1: int, donor: np.ndarray, d0: int, d1: int,
               fade_s: float = 0.015, sr: int = 16000) -> np.ndarray | None:
    """Replace host[a0:a1] with donor audio time-fit to that slot, edges crossfaded.

    Length is unchanged, so the rest of the clip stays on the same clock.
    """
    tgt_n = int(a1) - int(a0)
    don = np.asarray(donor[int(d0):int(d1)], dtype=np.float32)
    if tgt_n < 8 or len(don) < 8 or a0 < 0 or a1 > len(host):
        return None
    xi = np.linspace(0, len(don) - 1, tgt_n)
    piece = np.interp(xi, np.arange(len(don)), don).astype(np.float32)
    nfade = min(int(fade_s * sr), tgt_n // 4)
    if nfade > 0:
        ramp = np.linspace(0.0, 1.0, nfade, dtype=np.float32)
        piece[:nfade] = host[a0:a0 + nfade] * (1.0 - ramp) + piece[:nfade] * ramp
        piece[-nfade:] = piece[-nfade:] * (1.0 - ramp) + host[a1 - nfade:a1] * ramp
    out = np.array(host, dtype=np.float32, copy=True)
    out[a0:a1] = piece
    return out


def word_flags(issues) -> tuple[set[tuple[int, int, int]], int]:
    """(surah, ayah, word) keys of word-level error flags, plus unkeyed issue count."""
    keys: set[tuple[int, int, int]] = set()
    extra = 0
    for issue in issues or []:
        if issue.get("kind") not in ERR:
            continue
        word = issue.get("word")
        if word is None:
            extra += 1
            continue
        keys.add((int(issue["surah"]), int(issue["ayah"]), int(issue["word"])))
    return keys, extra


def group_overlap(train_groups: set, test_groups: set) -> int:
    return len(train_groups & test_groups)

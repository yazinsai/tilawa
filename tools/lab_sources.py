"""Pure logic for the lab source fetch and word alignment (.scratch/lab, ticket 02); run by modal/lab_fetch_align.py.

Pool rule, mp3quran timing parsing, the teacher's stitched windows, word spans from CTC token spans, and the
ayah-window consistency check. No I/O beyond the reference index; numpy and rapidfuzz only.
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261028
SR = 16_000
FRAME = 640  # teacher frame: 40 ms of 16 kHz samples
# Stitched teacher pass: 30 s windows every 25 s; each frame is taken from the window whose centre is nearer.
WINDOW_S, HOP_S = 30, 25

# Shared-run surahs the strict slice needs (refrains), always pooled when timed.
REQUIRED = (55, 77, 54, 26)
# (min words, max words, count): hosts for 3-8 min sessions from ayah 1, then medium and long surahs.
STRATA = ((160, 450, 8), (450, 1500, 5), (1500, 10_000, 3))
POOL = 20
MIN_WORDS = 160
# Juz start (surah, ayah), juz 1..30.
JUZ = [(1, 1), (2, 142), (2, 253), (3, 93), (4, 24), (4, 148), (5, 82), (6, 111), (7, 88), (8, 41), (9, 93), (11, 6), (12, 53), (15, 1),
       (17, 1), (18, 75), (21, 1), (23, 1), (25, 21), (27, 56), (29, 46), (33, 31), (36, 28), (39, 32), (41, 47), (46, 1), (51, 31),
       (58, 1), (67, 1), (78, 1)]

PAD_S = 1.5  # alignment search margin around the neighbours' mp3quran windows
GARBAGE_LOGP = float(np.log(1e-3))
MISMATCH_S = 0.3
MAX_PER = 0.15
MIN_WORD_SCORE = 0.2
BASMALA_MAX_S = 20.0


def reference() -> dict:
    return json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text())


def surah_words(ref: dict) -> dict[int, int]:
    n: Counter = Counter()
    for a in ref["ayahs"]:
        n[a["s"]] += len(a["w"])
    return dict(n)


def juz_of(surah: int) -> int:
    return max(j for j, start in enumerate(JUZ, 1) if start <= (surah, 1))


def pool(read_id: int, timed: list[int], words: dict[int, int], seed: int = SEED) -> list[int]:
    """About POOL surahs: REQUIRED, then per stratum a seeded pick that always takes the least-covered juz."""
    rng = random.Random(f"{seed}:{read_id}")
    timed = sorted(set(timed))
    chosen = [s for s in REQUIRED if s in timed]

    def take(cands: list[int], n: int) -> None:
        cands = [s for s in cands if s not in chosen]
        rng.shuffle(cands)
        for _ in range(n):
            if not cands:
                return
            used = Counter(juz_of(s) for s in chosen)
            best = min(cands, key=lambda s: used[juz_of(s)])
            chosen.append(best)
            cands.remove(best)

    for lo, hi, n in STRATA:
        take([s for s in timed if lo <= words[s] < hi], n)
    take([s for s in timed if words[s] >= MIN_WORDS], POOL - len(chosen))
    return sorted(chosen)


def parse_timing(raw: list[dict], n_ayahs: int, duration_s: float | None = None) -> tuple[list[dict], dict | None, list[str]]:
    """mp3quran ayat_timing rows -> ([{ayah, startS, endS}] for 1..n_ayahs, pre-roll (ayah 0) or None, surah-level problems).

    Timings that stop well before the end of the audio (more than max(60 s, a quarter of it) left over) were made
    for another edit of the recording.
    """
    rows = sorted(raw, key=lambda r: int(r["ayah"]))
    pre = next(({"startS": r["start_time"] / 1000, "endS": r["end_time"] / 1000} for r in rows if int(r["ayah"]) == 0), None)
    ayat = [{"ayah": int(r["ayah"]), "startS": r["start_time"] / 1000, "endS": r["end_time"] / 1000} for r in rows if int(r["ayah"]) > 0]
    bad = []
    if [a["ayah"] for a in ayat] != list(range(1, n_ayahs + 1)):
        bad.append(f"ayahs {[a['ayah'] for a in ayat][:3]}..{len(ayat)} != 1..{n_ayahs}")
    for a, b in zip(ayat, ayat[1:]):
        if b["startS"] < a["endS"] - 1e-3:
            bad.append(f"ayah {b['ayah']} starts before {a['ayah']} ends")
    bad += [f"ayah {a['ayah']} empty window" for a in ayat if a["endS"] <= a["startS"]]
    if duration_s and ayat and duration_s - ayat[-1]["endS"] > max(60.0, duration_s / 4):
        bad.append(f"timing ends at {ayat[-1]['endS']:.0f} s of {duration_s:.0f} s audio")
    if duration_s and ayat and ayat[-1]["endS"] > duration_s + 1:
        bad.append(f"timing ends at {ayat[-1]['endS']:.0f} s, after the {duration_s:.0f} s audio")
    return ayat, pre, bad


def windows(n_samples: int) -> list[tuple[int, int, int]]:
    """(start sample, first kept local frame, end kept local frame or -1 for all) per teacher window.

    Windows start on the frame grid, so local frame f of a window starting at s is global frame s // FRAME + f.
    """
    w, hop = WINDOW_S * SR, HOP_S * SR
    starts = [0]
    while starts[-1] + w < n_samples:
        starts.append(starts[-1] + hop)
    edge = (w - hop) // 2 // FRAME  # frames into the overlap where the next window takes over
    out = []
    for j, s in enumerate(starts):
        lo = edge if j else 0
        hi = hop // FRAME + edge if j + 1 < len(starts) else -1
        out.append((s, lo, hi))
    return out


def greedy(ids: np.ndarray, conf: np.ndarray) -> list[tuple[int, int, int, float]]:
    """CTC collapse of top-1 ids: (token id, start frame, end frame inclusive, max confidence)."""
    out, prev = [], 0
    for f, (i, c) in enumerate(zip(ids.tolist(), conf.tolist())):
        if i and i == prev:
            out[-1] = (i, out[-1][1], f, max(out[-1][3], c))
        elif i:
            out.append((i, f, f, c))
        prev = i
    return out


def per(canon: str, hyp: str) -> float:
    """Length-insensitive phoneme error rate: runs squeezed on both sides."""
    canon, hyp = squeeze(canon), squeeze(hyp)
    return Levenshtein.distance(canon, hyp) / max(1, len(canon))


def garbage(lp: np.ndarray) -> np.ndarray:
    """Log-posteriors where the blank may absorb any speech (isti'adha, a repeated phrase) at GARBAGE_LOGP below the frame's best."""
    out = lp.copy()
    out[:, 0] = np.maximum(out[:, 0], out.max(axis=1) + GARBAGE_LOGP)
    return out


def collapse(p: str, offsets: list[int]) -> tuple[str, list[int]]:
    """Runs of one phoneme (madd, ghunna, shadda lengths) as one CTC token, keeping word starts as token starts.

    Reciters' madd and ghunna lengths differ from the reference profile's; one token per run lets the aligner take
    whatever count the teacher heard. Returns the collapsed string and each word's offset into it.
    """
    starts = set(offsets)
    keep = [i for i in range(len(p)) if i == 0 or p[i] != p[i - 1] or i in starts]
    pos = {i: k for k, i in enumerate(keep)}
    return "".join(p[i] for i in keep), [pos[o] if o in pos else len(keep) for o in offsets]


def squeeze(s: str) -> str:
    """s with runs of one character reduced to one: the length-insensitive form PER is taken on."""
    return "".join(c for i, c in enumerate(s) if i == 0 or c != s[i - 1])


def extend(spans: list[tuple[int, int]], seq: list[int], top1: np.ndarray, limit: int, max_gap: int = 5) -> list[tuple[int, int]]:
    """Each token's span stretched over later top-1 spikes of the same token (madd repeats), each within max_gap frames
    of the last, before the next token starts."""
    out = []
    for k, ((s, e), c) in enumerate(zip(spans, seq)):
        nxt = spans[k + 1][0] if k + 1 < len(spans) else limit
        for f in np.nonzero(top1[e:nxt] == c)[0] + e:
            if f - e > max_gap:
                break
            e = int(f) + 1
        out.append((s, e))
    return out


def word_spans(offsets: list[int], spans: list[tuple[int, int]], token_p: list[float]) -> list[tuple[int, int, float | None]]:
    """Per word (start frame, end frame exclusive, mean token posterior) from per-token CTC frame spans.

    A word whose phonemes are all merged into its neighbour (no tokens of its own) gets a zero-length span at the
    previous token's end and no score.
    """
    out = []
    ends = offsets[1:] + [len(spans)]
    for o, e in zip(offsets, ends):
        if e <= o:
            at = spans[o - 1][1] if o else spans[0][0]
            out.append((at, at, None))
        else:
            out.append((spans[o][0], spans[e - 1][1], float(np.mean(token_p[o:e]))))
    return out


def mismatches(ayat: list[dict], aligned: list[tuple[float, float] | None]) -> list[float]:
    """Seconds by which each mp3quran window disagrees with the aligned speech.

    A window must contain its own aligned span and none of its neighbours' aligned speech. An ayah that failed to
    align is infinitely off; its neighbours are checked without it.
    """
    out = []
    for k, (a, sp) in enumerate(zip(ayat, aligned)):
        if sp is None:
            out.append(float("inf"))
            continue
        prev = aligned[k - 1] if k else None
        nxt = aligned[k + 1] if k + 1 < len(aligned) else None
        d = [a["startS"] - sp[0], sp[1] - a["endS"]]
        if prev:
            d.append(prev[1] - a["startS"])
        if nxt:
            d.append(a["endS"] - nxt[0])
        out.append(max(0.0, *d))
    return out


def verdict(mismatch: float, ayah_per: float | None, min_word: float | None) -> list[str]:
    reasons = []
    if ayah_per is None:
        return ["align-failed"]
    if mismatch > MISMATCH_S:
        reasons.append(f"window-mismatch {mismatch:.2f}s")
    if ayah_per > MAX_PER:
        reasons.append(f"per {ayah_per:.2f}")
    if min_word is not None and min_word < MIN_WORD_SCORE:
        reasons.append(f"word-score {min_word:.2f}")
    return reasons


def rms_db(x: np.ndarray, hop: int = SR // 100) -> np.ndarray:
    """dB RMS per 10 ms frame of float audio."""
    n = len(x) // hop
    f = x[: n * hop].astype(np.float64).reshape(n, hop)
    return 10 * np.log10(np.mean(f * f, axis=1) + 1e-10)


def cut(db: np.ndarray, end_s: float, start_s: float, margin: float = 0.04) -> tuple[float, float]:
    """Quietest 10 ms frame between a word's end and the next word's start (time, dB)."""
    lo = max(0, int((min(end_s, start_s) - margin) * 100))
    hi = min(len(db), max(lo + 1, int(np.ceil((max(end_s, start_s) + margin) * 100))))
    i = lo + int(np.argmin(db[lo:hi]))
    return (i + 0.5) / 100, float(db[i])

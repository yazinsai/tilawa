"""Student distillation: targets, teacher/student frame mapping, losses and PER."""

# Phoneme level of obadx/muaalem-model-v3_2 in id order 1..42 (the `ref-v2` inventory); id 0 is the CTC blank.
TOKENS = "".join(chr(c) for c in (
    0x621, 0x628, 0x62A, 0x62B, 0x62C, 0x62D, 0x62E, 0x62F, 0x630, 0x631, 0x632, 0x633, 0x634, 0x635,
    0x636, 0x637, 0x638, 0x639, 0x63A, 0x641, 0x642, 0x643, 0x644, 0x645, 0x646, 0x647, 0x648, 0x64A,
    0x627, 0x6E6, 0x6E5, 0x64E, 0x64F, 0x650, 0x6EA, 0x640, 0x672, 0x687, 0x6BA, 0x6FE, 0x6DC, 0x619,
))
VOCAB = len(TOKENS) + 1
_ID = {t: i + 1 for i, t in enumerate(TOKENS)}

_AYAH_POS: dict[int, dict[tuple[int, int], int]] = {}


def phoneme_ids(phonemes: str) -> list[int]:
    return [_ID[t] for t in phonemes]


def ids_phonemes(ids) -> str:
    return "".join(TOKENS[i - 1] for i in ids if i > 0)


def ctc_collapse(frame_ids) -> list[int]:
    out, prev = [], -1
    for i in frame_ids:
        if i != prev and i != 0:
            out.append(i)
        prev = i
    return out


def edit_distance(a, b) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        diag, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            diag, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diag + (x != y))
    return row[-1]


def corpus_per(pairs) -> float:
    """Total edits over total reference length, for (hypothesis, reference) pairs."""
    edits = sum(edit_distance(h, r) for h, r in pairs)
    return edits / max(1, sum(len(r) for _, r in pairs))


def mix_noise(wave, noise, snr_db: float, rng):
    """Add `noise`, looped from a random offset to the clip length, at `snr_db` below the clip's power."""
    import numpy as np

    start = int(rng.integers(len(noise)))
    n = np.resize(np.roll(noise, -start), len(wave))
    p_wave = float(np.mean(np.square(wave, dtype=np.float64)))
    p_noise = float(np.mean(np.square(n, dtype=np.float64)))
    if p_wave == 0 or p_noise == 0:
        return wave
    return (wave + n * np.sqrt(p_wave / (p_noise * 10 ** (snr_db / 10)))).astype(np.float32)


def teacher_frames(n_student: int, n_teacher: int, delay: int = 0):
    """Teacher frame index for each student frame. Both run at 25 Hz; `delay` makes the student lag the teacher."""
    import numpy as np

    return np.clip(np.arange(n_student) - delay, 0, n_teacher - 1)


def soft_target_loss(log_probs, top_ids, top_p, valid):
    """Mean per-frame cross-entropy from the teacher's renormalised top-k posteriors to the student's.

    log_probs (B, T, V); top_ids, top_p (B, T, K); valid (B, T) bool.
    """
    p = top_p.float()
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-6)
    ce = -(p * log_probs.gather(-1, top_ids.long())).sum(-1)
    return (ce * valid).sum() / valid.sum().clamp_min(1)


def reference_phonemes(ref: dict, label: dict) -> str:
    """Canonical phonemes of a tlog-v2 label: a word span in the reference index, or a text-only label's text."""
    if label["kind"] == "text":
        return label["text"]
    ayahs = ref["ayahs"]
    pos = _AYAH_POS.setdefault(id(ref), {(a["s"], a["a"]): i for i, a in enumerate(ayahs)})
    start, end = label["span"]["start"], label["span"]["end"]
    i, j = pos[(start["surah"], start["ayah"])], pos[(end["surah"], end["ayah"])]
    out = []
    for k in range(i, j + 1):
        a = ayahs[k]
        lo = a["o"][start["word"] - 1] if k == i else 0
        hi = a["o"][end["word"]] if k == j and end["word"] < len(a["o"]) else len(a["p"])
        out.append(a["p"][lo:hi])
    return "".join(out)

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


def exclusions(text: str) -> set[tuple[str, int]]:
    """`shard:row` lines (docs/benchmarks/gold-per-v1-exclude.txt) as (shard, row) keys."""
    return {(s, int(r)) for s, r in (line.rsplit(":", 1) for line in text.split() if line)}


def held_out(excluded: set[tuple[str, int]], shard: str, row: int, split: str) -> bool:
    """Gold PER clips never enter training; eval splits keep them so logged PER stays comparable."""
    return split == "train" and (shard, row) in excluded


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


def ctc_token_spans(frame_ids) -> tuple[list[int], list[tuple[int, int]]]:
    """Collapsed CTC tokens and the inclusive-exclusive frame span of each emitted token."""
    tokens, spans = [], []
    prev = -1
    for frame, token in enumerate(frame_ids):
        token = int(token)
        if token and token != prev:
            tokens.append(token)
            spans.append((frame, frame + 1))
        elif token and token == prev:
            spans[-1] = (spans[-1][0], frame + 1)
        prev = token
    return tokens, spans


def unfaithful_splice(host_wave, host_frames, donor_wave, donor_frames, operation: str, rng) -> dict | None:
    """Make a training-only insertion, substitution or omission with an as-heard CTC target.

    The old teacher remains usable outside the edited interval; `kd_edit` describes how output samples map back to
    the unedited host. The caller masks the interval itself because the old host teacher is deliberately wrong there.
    """
    import numpy as np
    from splice import boundaries, replace, shorten

    host_tokens, host_spans = ctc_token_spans(host_frames)
    donor_tokens, donor_spans = ctc_token_spans(donor_frames)
    if len(host_tokens) < 8 or (operation in ("insert", "swap") and len(donor_tokens) < 2):
        return None

    def segment(tokens, spans, leave_edges: bool) -> tuple[int, int, int, int] | None:
        lo, hi = (3, len(tokens) - 3) if leave_edges else (0, len(tokens))
        if hi - lo < 2:
            return None
        width = int(rng.integers(2, min(9, hi - lo) + 1))
        start = int(rng.integers(lo, hi - width + 1))
        return start, start + width, spans[start][0], spans[start + width - 1][1]

    host_bounds = boundaries(host_spans, len(host_frames), len(host_wave))
    if operation == "insert":
        # The first emitted phone follows the clip's leading pause, matching the strongest continuation-prior site.
        h0 = h1 = 0
        a = b = host_bounds[0][0]
    else:
        picked = segment(host_tokens, host_spans, True)
        if not picked:
            return None
        h0, h1, _, _ = picked
        a, b = host_bounds[h0][0], host_bounds[h1 - 1][1]

    inserted = []
    if operation in ("insert", "swap"):
        picked = segment(donor_tokens, donor_spans, False)
        if not picked:
            return None
        d0, d1, _, _ = picked
        donor_bounds = boundaries(donor_spans, len(donor_frames), len(donor_wave))
        da, db = donor_bounds[d0][0], donor_bounds[d1 - 1][1]
        wave = replace(host_wave, a, b, donor_wave, da, db)
        inserted = donor_tokens[d0:d1]
    elif operation == "delete":
        wave = shorten(host_wave, a, b, keep=0)
    else:
        raise ValueError(f"unknown splice operation: {operation}")

    target = host_tokens[:h0] + inserted + host_tokens[h1:]
    if not target or len(wave) < 400:
        return None
    delta = len(wave) - len(host_wave)
    edited_samples = max(0, (b - a) + delta)
    return {
        "wave": np.asarray(wave, dtype=np.float32),
        "target_ids": target,
        "operation": operation,
        "kd_edit": {
            "start": a,
            "end": a + edited_samples,
            "delta": delta,
            "source_samples": len(host_wave),
            "output_samples": len(wave),
        },
    }


def splice_teacher_frames(n_output: int, n_teacher: int, edit: dict, sample_rate: int = 16_000,
                          pad_s: float = 0.08):
    """Map edited output frames to the old host teacher and mask the edit plus its coarticulation boundary."""
    import numpy as np

    samples = (np.arange(n_output, dtype=np.float64) + 0.5) * edit["output_samples"] / max(1, n_output)
    pad = pad_s * sample_rate
    valid = (samples < edit["start"] - pad) | (samples >= edit["end"] + pad)
    source_samples = np.where(samples < edit["start"], samples, samples - edit["delta"])
    valid &= (source_samples >= 0) & (source_samples < edit["source_samples"])
    frames = np.floor(source_samples * n_teacher / max(1, edit["source_samples"])).astype(np.int64)
    return np.clip(frames, 0, max(0, n_teacher - 1)), valid


def spliced_target(ref: str, teacher: str) -> str:
    """CTC target that lets the student outgrow the teacher: `per_breakdown.spliced` (teacher at the clip edges,
    reference between the first and last 3-phoneme exact match) with runs collapsed, leaving length to soft KD."""
    from per_breakdown import collapse, spliced

    return collapse(spliced(ref, teacher))


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

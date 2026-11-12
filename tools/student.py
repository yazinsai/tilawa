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


def censored_target_projection(log_probs, top_ids, top_p):
    """Minimum-relative-entropy completion of a conditional top-k observation.

    A completion has retained probabilities alpha*q and omitted probabilities
    <= alpha*min(q), where q is the recorded conditional distribution. Thus
    alpha >= 1/(1 + (V-K)*min(q)); neither alpha nor the tail identities are
    treated as known. Unconstrained tail classes keep their student ratios.
    This is set-valued supervision, not reconstruction of the teacher's tail.

    The KL projection is convex. Sort omitted classes, cap the largest ones,
    and solve the remaining pooled mass analytically for each active-set size.
    This helper is detached: the loss uses the envelope theorem, so gradients
    never flow through a chosen nuisance completion. Inputs are valid rows only.
    """
    import torch

    with torch.no_grad():
        if log_probs.ndim != 2 or top_ids.shape != top_p.shape or top_ids.ndim != 2:
            raise ValueError('Censored targets require (frames, classes) arrays')
        N, V = log_probs.shape
        K = top_ids.shape[1]
        if top_ids.shape[0] != N or not 0 < K <= V:
            raise ValueError('Invalid retained-set geometry')
        ids = top_ids.long()
        p = top_p.to(log_probs.dtype)
        if (not torch.isfinite(log_probs).all() or not torch.isfinite(p).all() or
                (p < 0).any() or (p.sum(-1) <= 0).any() or (ids < 0).any() or (ids >= V).any()):
            raise ValueError('Invalid censored observation')
        if K > 1 and ((ids.sort(-1).values[:, 1:] == ids.sort(-1).values[:, :-1]).any() or
                      (p[:, 1:] > p[:, :-1]).any()):
            raise ValueError('Retained classes must be unique and top-k ordered')
        q = p / p.sum(-1, keepdim=True)
        retained = log_probs.gather(-1, ids)
        # Zero recorded entries constrain every omitted class to zero. Avoid
        # log(0) in both the geometric mean and the active-set calculation.
        tiny = torch.finfo(log_probs.dtype).tiny
        log_q = q.clamp_min(tiny).log()
        log_g = (q * (retained - log_q)).sum(-1)
        target = torch.zeros_like(log_probs)
        if K == V:
            return target.scatter(-1, ids, q)
        omitted = torch.ones_like(log_probs, dtype=torch.bool).scatter(-1, ids, False)
        tail = log_probs[omitted].reshape(N, V-K)
        values, order = tail.sort(-1, descending=True)
        m = q[:, -1]
        log_m = m.clamp_min(tiny).log()
        rank = torch.arange(V-K+1, device=log_probs.device, dtype=log_probs.dtype)
        cumulative = torch.cat([torch.zeros_like(log_g[:, None]),
                                (values - log_m[:, None]).cumsum(-1)], -1)
        log_g_all = (log_g[:, None] + m[:, None]*cumulative) / (1 + m[:, None]*rank)
        next_value = torch.cat([values, torch.full_like(log_g[:, None], -torch.inf)], -1)
        # The first consistent set caps precisely the tail classes whose
        # unconstrained solution would outrank the smallest retained class.
        feasible = next_value <= log_g_all + log_m[:, None]
        active = feasible.to(torch.int64).argmax(-1)
        log_g = log_g_all.gather(-1, active[:, None]).squeeze(-1)
        capped = rank[:-1][None] < active[:, None]
        free_log_mass = values.masked_fill(capped, -torch.inf).logsumexp(-1)
        log_normalizer = torch.logaddexp(log_g + (1 + active*m).log(), free_log_mass)
        alpha = (log_g - log_normalizer).exp()
        tail_target = torch.where(capped, (alpha*m)[:, None], (values-log_normalizer[:, None]).exp())
        # A zero smallest entry implies no admissible omitted mass, even when
        # the student gives excluded classes positive probability.
        zero_tail = m == 0
        alpha = torch.where(zero_tail, torch.ones_like(alpha), alpha)
        tail_target = tail_target.masked_fill(zero_tail[:, None], 0)
        target.scatter_(-1, ids, alpha[:, None]*q)
        target[omitted] = torch.zeros_like(tail).scatter(-1, order, tail_target).reshape(-1)
        return target


def censored_soft_target_loss(log_probs, top_ids, top_p, valid):
    """Mean KL to the admissible censored target set, ignoring padding.

    Minimize over unknown retained mass and marginalize unrestricted tail
    classes into their pooled student mass. Unlike renormalized cross-entropy,
    a student consistent with the observation has zero loss and gradient even
    when it assigns positive probability to omitted phones. Ordering violations
    remain penalized. Full-support targets have the ordinary KD gradient.
    """
    import torch

    if valid.shape != log_probs.shape[:-1] or top_ids.shape[:-1] != valid.shape:
        raise ValueError('Censored target mask does not match frame geometry')
    lp = log_probs[valid]
    if not len(lp):
        return lp.sum()
    target = censored_target_projection(lp, top_ids[valid], top_p[valid])
    return (torch.xlogy(target, target) - target*lp).sum(-1).mean()


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

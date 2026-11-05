"""Slot-restricted CTC and complementary shipped-model KL for listener adaptation."""


def collapse_owned(phones, owners):
    """Duration runs collapse as in shipped CTC; a shared phone retains both word owners."""
    out, who = [], []
    for p, w in zip(phones, owners, strict=True):
        if out and out[-1] == p:
            who[-1].add(w)
        else:
            out.append(p)
            who.append({w})
    return out, who


def viterbi_states(log_probs, target):
    """Full-clip CTC alignment; returns state indices, or rejects an impossible target."""
    import numpy as np

    lp = np.asarray(log_probs)
    seq = np.zeros(2 * len(target) + 1, dtype=np.int64)
    seq[1::2] = target
    if not target or lp.ndim != 2 or not len(lp):
        raise ValueError("Empty alignment input")
    prev = np.full(len(seq), -np.inf)
    prev[:2] = lp[0, seq[:2]]
    back = np.zeros((len(lp), len(seq)), np.int8)
    jump = np.zeros(len(seq), bool)
    jump[2:] = (seq[2:] != 0) & (seq[2:] != seq[:-2])
    for t in range(1, len(lp)):
        one = np.r_[-np.inf, prev[:-1]]
        two = np.r_[[-np.inf, -np.inf], prev[:-2]]
        two[~jump] = -np.inf
        choices = np.stack([prev, one, two])
        back[t] = choices.argmax(0)
        prev = choices.max(0) + lp[t, seq]
    state = len(seq) - 1 if prev[-1] >= prev[-2] else len(seq) - 2
    if not np.isfinite(prev[state]):
        raise ValueError("Impossible CTC alignment")
    path = np.empty(len(lp), np.int64)
    for t in range(len(lp) - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    return path


def word_slots(path, target, owners, selected, margin=5):
    """Emit each selected word's CTC target and half-open frame slot, plus ±200 ms."""
    import numpy as np

    slots = []
    for word in sorted(selected):
        ids = [i for i, who in enumerate(owners) if word in who]
        if not ids:
            raise ValueError("Selected word has no phone target")
        if ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError("Repeated/noncontiguous selected word")
        hits = np.flatnonzero(np.isin(path, [2 * i + 1 for i in ids]))
        if not len(hits):
            raise ValueError("Selected word not aligned")
        a, b = max(0, int(hits[0]) - margin), min(len(path), int(hits[-1]) + 1 + margin)
        local = [target[i] for i in ids]
        need = len(local) + sum(x == y for x, y in zip(local, local[1:]))
        if b - a < need:
            raise ValueError("Word slot too short for CTC")
        slots.append({"word": word, "from": a, "to": b, "target": local})
    return slots


def masked_loss(log_probs, shipped, slots, kl_weight=10.0):
    """CTC has exactly zero logit gradient outside selected slots; KL covers their complement."""
    import torch
    import torch.nn.functional as F

    if log_probs.shape != shipped.shape or log_probs.ndim != 2:
        raise ValueError("Mismatched frame grids")
    outside = torch.ones(len(log_probs), dtype=torch.bool, device=log_probs.device)
    ctc = log_probs.sum() * 0
    for s in slots:
        a, b, y = s["from"], s["to"], s["target"]
        if not 0 <= a < b <= len(log_probs) or not y or min(y) <= 0:
            raise ValueError("Invalid local CTC slot")
        outside[a:b] = False
        ctc = ctc + F.ctc_loss(log_probs[a:b, None], torch.tensor(y, device=log_probs.device),
                             torch.tensor([b - a]), torch.tensor([len(y)]), reduction="sum", zero_infinity=False)
    # Normalize by total clip frames as in shipped CTC, so short words cannot dominate clips.
    ctc = ctc / max(1, len(log_probs))
    kd = F.kl_div(log_probs, shipped.detach(), log_target=True, reduction="none").sum(-1)
    kl = kd[outside].sum() / max(1, int(outside.sum()))
    return {"loss": ctc + kl_weight * kl, "ctc": ctc, "kl": kl, "outside": outside}

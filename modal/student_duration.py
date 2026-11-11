"""Occurrence-balanced identity distillation for a frozen encoder's phone head.

The two arms share transcript-conditioned teacher occupancies. Only the measure
over nonblank identity targets changes: frames, or distinct spoken occurrences.
Blank/nonblank supervision, CTC and shipped-output preservation are identical.
"""
import hashlib


def speaker_partition(speakers):
    if any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Every recording needs an identified training speaker")
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(unique) < 5:
        raise ValueError("Need at least five identified training speakers")
    return {s: "selection" if i % 5 == 0 else "fit" for i, s in enumerate(unique)}


def admitted_none(labels, excluded=()):
    excluded = set(excluded) | {r.get("speaker") for r in labels if r.get("split") != "train"}
    selected, seen = [], set()
    for row in labels:
        if row.get("split") != "train" or row.get("kind") != "none":
            continue
        if not row.get("speaker") or row["speaker"] in excluded:
            continue
        if (row.get("scored") is not True or any(row.get(k) != [] for k in
                ("events", "non_errors", "unsupported")) or
                row.get("excluded") or row.get("exclude")):
            continue
        if not row.get("id") or not row.get("wav") or not row.get("ayahs"):
            raise ValueError("None-labelled recording lacks identity, audio or transcript")
        if row["id"] in seen:
            raise ValueError("Duplicate none-labelled recording identity")
        seen.add(row["id"])
        selected.append(row)
    return selected


def check_transcript(evidence, target):
    import torch
    if (evidence.ndim != 2 or target.ndim != 1 or target.dtype != torch.int64 or
            evidence.shape[1] < 2 or not len(target) or (target <= 0).any() or
            (target >= evidence.shape[1]).any() or len(evidence) < len(target) +
            int((target[1:] == target[:-1]).sum())):
        raise ValueError("Impossible complete transcript alignment")


def ctc_occupancy(evidence, target):
    """Marginalize all CTC paths; return one column per transcript occurrence.

    State identity, rather than phone class, distinguishes repeated phones.
    Alpha includes the current emission; beta includes only future emissions.
    Alignment is frozen, accumulated in float64, and never follows a trained arm.
    """
    import torch
    check_transcript(evidence, target)
    with torch.no_grad():
        lp = evidence.detach().double().log_softmax(-1)
        if torch.isnan(lp).any() or torch.isposinf(lp).any():
            raise ValueError("Non-finite alignment evidence")
        labels = torch.zeros(2 * len(target) + 1, dtype=torch.int64, device=lp.device)
        labels[1::2] = target.to(lp.device)
        emissions = lp[:, labels]
        skip = torch.zeros_like(labels, dtype=torch.bool)
        skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
        alpha = torch.full_like(emissions, -torch.inf)
        alpha[0, :2] = emissions[0, :2]
        for t in range(1, len(lp)):
            prev = alpha[t - 1]
            one = torch.cat([prev.new_full((1,), -torch.inf), prev[:-1]])
            two = torch.cat([prev.new_full((2,), -torch.inf), prev[:-2]])
            alpha[t] = emissions[t] + torch.logsumexp(
                torch.stack([prev, one, two.masked_fill(~skip, -torch.inf)]), 0)
        log_z = torch.logsumexp(alpha[-1, -2:], 0)
        if not torch.isfinite(log_z):
            raise ValueError("Transcript has zero probability under alignment evidence")
        beta = torch.full_like(emissions, -torch.inf)
        beta[-1, -2:] = 0
        for t in range(len(lp) - 2, -1, -1):
            following = beta[t + 1] + emissions[t + 1]
            one = torch.cat([following[1:], following.new_full((1,), -torch.inf)])
            two = torch.cat([following[2:].masked_fill(~skip[2:], -torch.inf),
                             following.new_full((2,), -torch.inf)])
            beta[t] = torch.logsumexp(torch.stack([following, one, two]), 0)
        states = (alpha + beta - log_z).exp()
        if not torch.allclose(states.sum(1), torch.ones_like(lp[:, 0]), atol=1e-8, rtol=1e-8):
            raise ValueError("CTC state marginals do not conserve probability")
        occurrences, blank = states[:, 1::2], states[:, ::2].sum(1)
        if (occurrences.sum(0) < 1 - 1e-8).any():
            raise ValueError("Complete transcript occurrence has no alignment mass")
        return occurrences, blank, log_z


def identity_weights(occupancy, arm, normalizer=None):
    import math
    import torch
    if (arm not in ("frame", "occurrence") or occupancy.ndim != 2 or
            not occupancy.numel() or not torch.isfinite(occupancy).all() or
            (occupancy < 0).any() or (occupancy.sum(1) > 1 + 1e-6).any()):
        raise ValueError("Invalid frozen occurrence occupancy or matched arm")
    q = occupancy.detach()
    duration = q.sum(0)
    if (duration <= 0).any():
        raise ValueError("Every occurrence must carry identity supervision")
    if normalizer is None:
        normalizer = q.shape[1] if arm == "occurrence" else float(duration.sum())
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("Invalid identity-supervision budget")
    if arm == "occurrence":
        return q / duration[None] / normalizer
    return q / normalizer


def identity_budget(rows):
    """Fix both arms' total identity weight from the fit partition only.

    A shared corpus denominator gives every occurrence equal mass even across
    recordings of different lengths. Neither arm's alignments or updates can
    change the budget. Blank/CTC/preservation retain their recording averages.
    """
    if not rows or any(not r["views"] for r in rows):
        raise ValueError("Need fit recordings for the shared identity budget")
    counts, masses, speech = [], [], []
    for row in rows:
        counts.append(sum(len(v["target"]) for v in row["views"]) / len(row["views"]))
        masses.append(sum(float(v["occupancy"].detach().sum()) for v in row["views"]) / len(row["views"]))
        speech.append(sum(float(v["teacher"].detach().log_softmax(-1).exp()[:, 1:].sum(-1).mean())
                          for v in row["views"]) / len(row["views"]))
    return dict(occurrence=sum(counts) / len(rows), frame=sum(masses) / len(rows),
                scale=sum(speech) / len(rows))


def phone_head(width, vocab):
    import torch
    head = torch.nn.Linear(width, vocab)
    # Preserve the installed quantized projection and refit an affine correction
    # on its pre-quantized features. No encoder weights or operators are changed.
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    return head


def log_probs(head, features, shipped):
    return (shipped.detach() + head(features.detach())).log_softmax(-1)


def objectives(predicted, target, teacher, shipped, occupancy, arm, budget=None):
    import torch
    from torch.nn import functional as F
    check_transcript(predicted, target)
    check_transcript(teacher, target)
    if (teacher.shape[1] != predicted.shape[1] or not 0 < len(teacher) <= len(predicted) or
            shipped.shape != predicted.shape or occupancy.shape != (len(teacher), len(target))):
        raise ValueError("Complete transcript, teacher and student clocks must agree")
    if any(not torch.isfinite(x).all() for x in (predicted, teacher, shipped)):
        raise ValueError("Non-finite recording evidence")
    ctc = F.ctc_loss(predicted[:, None], target,
        torch.tensor([len(predicted)]), torch.tensor([len(target)]),
        blank=0, reduction="sum", zero_infinity=False) / len(target)
    paired = predicted[:len(teacher)]
    teacher_probabilities = teacher.detach().log_softmax(-1).exp()
    teacher_blank = teacher_probabilities[:, 0]
    teacher_speech = teacher_probabilities[:, 1:].sum(-1)
    # Decompose full-distribution KD into the blank/nonblank decision and phone
    # identity conditional on speech. Occurrence balancing cannot directly push
    # down blank: identity has zero derivative with respect to the blank logit.
    blank = -(teacher_blank * paired[:, 0] +
              teacher_speech * paired[:, 1:].logsumexp(-1)).mean()
    conditional = paired[:, 1:].log_softmax(-1)
    weights = identity_weights(occupancy, arm, budget[arm] if budget else None).to(predicted)
    scale = budget["scale"] if budget else teacher_speech.mean()
    identity = -(weights * conditional[:, target - 1]).sum() * scale
    q = shipped.detach().log_softmax(-1)
    preserve = (q.exp() * (q - predicted)).sum(-1).mean()
    return ctc, identity, blank, preserve


def recording_loss(head, views, arm, budget=None):
    import torch
    if not views:
        raise ValueError("Need nonempty matched recording views")
    losses = [objectives(log_probs(head, v["features"], v["shipped"]),
                        v["target"], v["teacher"], v["shipped"], v["occupancy"], arm, budget)
              for v in views]
    terms = torch.stack([torch.stack(row) for row in losses]).mean(0)
    return terms.sum(), dict(zip(("ctc", "identity", "blank", "preservation"),
                                terms.detach().tolist()))

"""Occurrence-local temporal ranking for a frozen streaming phone-head refit.

Only teacher frame ordering shared by two independent existing augmentation
views is supervised. CTC supplies probability scale and shipped output KL
retains ordinary speech and pause examples. No inference decoding rule changes.
"""
import hashlib

# Fixed before any practice diagnostic. Sample comparisons by frame identity,
# never by teacher probability magnitude or an arm's current predictions.
PAIR_RULE = dict(version=1, centres="CTC occurrence posterior medians",
                 neighbourhood="previous through following occurrence centre",
                 views="intersection of two independent augmentation intervals",
                 ordering="strict same-phone log-probability agreement, no ties",
                 maxPairsPerOccurrence=64, sampling="lowest SHA256 frame-pair hashes")


def transcript_target(phones):
    """Use the shipped CTC recipe's run-collapse of duration-expanded notation.

    The reference writes a held phone repeatedly. Those copies denote duration,
    not separate articulations that should be forced apart by CTC blanks.
    """
    return "".join(p for i, p in enumerate(phones) if not i or p != phones[i - 1])

def speaker_partition(speakers):
    if any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Every recording needs an identified training speaker")
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(unique) < 5 or any(not isinstance(s, str) or not s for s in unique):
        raise ValueError("Need at least five identified training speakers")
    # Selection speakers belong to the training corpus, never the release set.
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


def phone_head(width, vocab):
    import torch
    head = torch.nn.Linear(width, vocab)
    # An affine correction to the installed projection is a head refit. No
    # nonlinear bottleneck, extra phone labels or encoder parameters are added.
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    return head


def log_probs(head, features, shipped):
    return (shipped.detach() + head(features.detach())).log_softmax(-1)


def objectives(predicted, target, teacher, shipped):
    import torch
    from torch.nn import functional as F
    if (predicted.ndim != 2 or target.ndim != 1 or target.dtype != torch.int64 or
            teacher.ndim != 2 or teacher.shape[1] != predicted.shape[1] or
            not 0 < len(teacher) or shipped.shape != predicted.shape):
        raise ValueError("Complete transcript, teacher and student clocks must agree")
    if (not len(target) or (target <= 0).any() or (target >= predicted.shape[1]).any() or
            len(predicted) < len(target) + int((target[1:] == target[:-1]).sum())):
        raise ValueError("Impossible complete transcript alignment")
    if any(not torch.isfinite(x).all() for x in (predicted, teacher, shipped)):
        raise ValueError("Non-finite recording evidence")
    ctc = F.ctc_loss(predicted[:, None], target,
        torch.tensor([len(predicted)]), torch.tensor([len(target)]),
        blank=0, reduction="sum", zero_infinity=False) / len(target)
    # Full-distribution KD, not hard teacher phones. All real pause frames stay.
    common = min(len(teacher), len(predicted))
    kd = -(teacher[:common].detach().log_softmax(-1).exp() * predicted[:common]).sum(-1).mean()
    q = shipped.detach().log_softmax(-1)
    preserve = (q.exp() * (q - predicted)).sum(-1).mean()
    return ctc, kd, preserve


def occurrence_intervals(evidence, target):
    """Transcript-constrained occurrence neighbourhoods, including pause frames.

    Marginalized CTC distinguishes repeated occurrences without fixing one
    emission path. Each interval extends to the adjacent occurrence medians;
    rankings may move the emission anywhere inside it. Boundaries are frozen.
    """
    occupancy, _, _ = ctc_occupancy(evidence, target)
    cumulative = occupancy.cumsum(0) / occupancy.sum(0)[None]
    centres = (cumulative >= .5).to(dtype=occupancy.dtype).argmax(0).tolist()
    return [(centres[k - 1] if k else 0,
             centres[k + 1] + 1 if k + 1 < len(centres) else len(evidence))
            for k in range(len(target))]


def consensus_pairs(first, second, phone, lo, hi, occurrence):
    """Return (phone, higher frame, lower frame) with agreement in both views.

    Selection uses only strict ordering and deterministic frame-pair hashes,
    so a teacher's confidence scale supplies neither margins nor weights.
    """
    import torch
    if not 0 <= lo <= hi <= min(len(first), len(second)):
        raise ValueError("Pair interval is outside the shared native frame clock")
    if first.shape[1] != second.shape[1] or not 0 < phone < first.shape[1]:
        raise ValueError("Temporal pairs need a shared nonblank phone inventory")
    if any(not torch.isfinite(v[lo:hi, phone]).all() for v in (first, second)):
        raise ValueError("Non-finite same-phone ranking evidence")
    candidates = []
    a, b = first.detach().cpu(), second.detach().cpu()
    for i in range(lo, hi):
        for j in range(i + 1, hi):
            da, db = float(a[i, phone] - a[j, phone]), float(b[i, phone] - b[j, phone])
            if not ((da > 0 and db > 0) or (da < 0 and db < 0)):
                continue
            key = hashlib.sha256(f"{occurrence}:{phone}:{i}:{j}".encode()).digest()
            candidates.append((key, (phone, i, j) if da > 0 else (phone, j, i)))
    candidates.sort(key=lambda c: c[0])
    return torch.tensor([p for _, p in candidates[:PAIR_RULE["maxPairsPerOccurrence"]]],
                        dtype=torch.int64).reshape(-1, 3)


def stable_pairs(first, second, target, student_frames):
    """Align full teacher views, then pair only their shared student-clock rows."""
    import torch
    if student_frames < 1 or any(not torch.isfinite(v).all() for v in (first, second)):
        raise ValueError("Invalid two-view ranking evidence")
    first, second = first.detach().log_softmax(-1), second.detach().log_softmax(-1)
    intervals = [occurrence_intervals(v, target) for v in (first, second)]
    common = min(student_frames, len(first), len(second))
    out = []
    for k, phone in enumerate(target.tolist()):
        lo = max(intervals[0][k][0], intervals[1][k][0])
        hi = min(common, intervals[0][k][1], intervals[1][k][1])
        if hi <= lo:
            continue
        pairs = consensus_pairs(first, second, phone, lo, hi, k)
        if len(pairs):
            out.append(pairs)
    return out


def temporal_loss(predicted, pairs):
    """Equal-occurrence pairwise logistic ranking; no teacher value targets."""
    import torch
    from torch.nn import functional as F
    losses = []
    for p in pairs:
        if (p.ndim != 2 or p.shape[1] != 3 or p.dtype != torch.int64 or not len(p)
                or (p[:, 0] <= 0).any() or (p[:, 0] >= predicted.shape[1]).any()
                or (p[:, 1:] < 0).any() or (p[:, 1:] >= len(predicted)).any()
                or (p[:, 1] == p[:, 2]).any()):
            raise ValueError("Invalid frozen temporal comparison")
        p = p.to(predicted.device)
        difference = predicted[p[:, 1], p[:, 0]] - predicted[p[:, 2], p[:, 0]]
        losses.append(F.softplus(-difference).mean())
    return torch.stack(losses).mean() if losses else predicted.sum() * 0


def recording_loss(head, views, pairs, arm):
    """Matched data, graph work and updates in CTC/KD/ranking head refits.

    All arms compute every objective, including zero-weight auxiliary paths.
    The original view retains CTC and preservation; ranking is applied to the
    two augmented views whose teacher ordering was checked for agreement.
    """
    import torch
    if arm not in ("ctc", "ordinary", "rank") or len(views) != 3:
        raise ValueError("Need three matched views and a recognized refit arm")
    predicted = [log_probs(head, v["features"], v["shipped"]) for v in views]
    losses = [objectives(p, v["target"], v["teacher"], v["shipped"])
              for p, v in zip(predicted, views)]
    ctc, kd, preserve = [torch.stack([l[i] for l in losses]).mean() for i in range(3)]
    rank = torch.stack([temporal_loss(p, pairs) for p in predicted[1:]]).mean()
    return ctc + preserve + int(arm == "ordinary") * kd + int(arm == "rank") * rank

"""Timing-marginalized, transcript-occurrence consistency for a frozen CTC encoder.

The timing posterior is a latent alignment, not a framewise target. Each transcript
occurrence pools the *entire* phone distribution at its possible emission times.
Repeated phone identities keep distinct occurrence slots. Detaching the alignment
prevents the head from reducing consistency by moving the timing weights instead
of reconciling its acoustic readout. Transcript CTC still trains emission timing.
"""

import hashlib
import math


def output_frames(feature_frames):
    """The SDK's 40 ms output clock after the streaming convolutional frontend."""
    return max(0, ((feature_frames - 7) // 2 + 1) // 2)


def origins(manifest):
    """Two chunk grids over identical fbank samples, with an integral output offset.

    The second stream starts half a chunk before acoustic time zero. Only the
    synthetic leading padding is added; speech samples and fbank rows are intact.
    Its first ``phase / 4`` outputs precede the recording and are discarded.
    """
    shift = manifest["shift"]
    if manifest["frameMs"] != 40 or manifest["fbank"]["frameShiftMs"] != 10:
        raise ValueError("paired origins require the existing 10/40 ms clocks")
    if shift % 8 or manifest["window"] != shift + 13:
        raise ValueError("half-chunk origin must lie on the output grid")
    return (0, shift // 2)


def speaker_partition(speakers, seed=0):
    """Freeze a recording-independent 80/20 split of authorized training speakers."""
    speakers = sorted(set(speakers), key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).digest())
    if len(speakers) < 2:
        raise ValueError("need at least two training speakers for grouped selection")
    valid = set(speakers[:min(len(speakers) - 1, max(1, math.ceil(len(speakers) / 5)))])
    return {s: "validation" if s in valid else "train" for s in speakers}


def authorized_none(label):
    """A prompt with a planted error or repeat is never a performed transcript."""
    return (label.get("split") == "train" and label.get("kind") == "none"
            and not label.get("excluded") and not label.get("events")
            and not label.get("non_errors") and not label.get("unsupported")
            and bool(label.get("speaker")) and bool(label.get("ayahs")))


def occurrence_weights(log_probs, target):
    """P(frame occupies phone occurrence | transcript), normalized over time.

    Exact CTC forward/backward over blank-interleaved transcript states. The
    result is (occurrences, frames), including separate adjacent equal phones.
    Impossible transcripts are rejected rather than silently producing zero loss.
    """
    import torch

    if log_probs.ndim != 2 or log_probs.shape[0] == 0:
        raise ValueError("expected a nonempty (frames, vocabulary) distribution")
    target = torch.as_tensor(target, dtype=torch.long, device=log_probs.device)
    if target.ndim != 1 or bool(((target <= 0) | (target >= log_probs.shape[1])).any()):
        raise ValueError("transcript must contain nonblank inventory IDs")
    if not len(target):
        return log_probs.new_zeros((0, len(log_probs)))
    with torch.no_grad():
        lp = log_probs.detach().double()
        labels = torch.zeros(2 * len(target) + 1, dtype=torch.long, device=lp.device)
        labels[1::2] = target
        emit = lp[:, labels]
        skip = (labels != 0) & (labels != torch.roll(labels, 2))
        skip[:2] = False
        initial = lp.new_full((len(labels),), -torch.inf)
        initial[:2] = emit[0, :2]
        alpha = [initial]
        for t in range(1, len(lp)):
            previous = alpha[-1]
            one = torch.roll(previous, 1)
            one[0] = -torch.inf
            two = torch.roll(previous, 2).masked_fill(~skip, -torch.inf)
            alpha.append(torch.logsumexp(torch.stack((previous, one, two)), dim=0) + emit[t])
        log_z = torch.logsumexp(alpha[-1][-2:], dim=0)
        if not bool(torch.isfinite(log_z)):
            raise ValueError("transcript has no legal CTC path in this recording")
        last = lp.new_full((len(labels),), -torch.inf)
        last[-2:] = 0
        beta = [last]
        for t in range(len(lp) - 2, -1, -1):
            following = beta[-1] + emit[t + 1]
            one = torch.roll(following, -1)
            one[-1] = -torch.inf
            two = torch.roll(following.masked_fill(~skip, -torch.inf), -2)
            two[-2:] = -torch.inf
            beta.append(torch.logsumexp(torch.stack((following, one, two)), dim=0))
        occupancy = (torch.stack(alpha) + torch.stack(beta[::-1]) - log_z).exp()[:, 1::2].T
        return (occupancy / occupancy.sum(dim=1, keepdim=True)).to(log_probs.dtype)


def occurrence_distributions(log_probs, target):
    """Full phone identity distributions; these are not one-hot transcript labels."""
    return occurrence_weights(log_probs, target) @ log_probs.exp()


def consistency_loss(first, second, target):
    """Mean symmetric Jensen-Shannon divergence, with timing marginalized per view."""
    import torch

    p = occurrence_distributions(first, target)
    q = occurrence_distributions(second, target)
    if p.shape != q.shape:
        raise ValueError("views must share a transcript and phone inventory")
    if not len(p):
        return (first.sum() + second.sum()) * 0
    tiny = torch.finfo(p.dtype).tiny
    mean = ((p + q) / 2).clamp_min(tiny)
    return ((p * (p.clamp_min(tiny).log() - mean.log())).sum(-1)
            + (q * (q.clamp_min(tiny).log() - mean.log())).sum(-1)).mean() / 2


def paired_loss(head, views, target, consistency_weight=1.0):
    """Both arms use both recordings, CTC and frozen-output speech/pause preservation.

    Only consistency_weight differs between the experimental arm and its matched
    control. Recording losses are averaged before the optimizer update.
    """
    import torch
    import torch.nn.functional as F

    target = torch.as_tensor(target, dtype=torch.long)
    predictions = [head(v["features"].float()).log_softmax(-1) for v in views]
    ctc, preservation = [], []
    for lp, view in zip(predictions, views):
        ctc.append(F.ctc_loss(lp[:, None], target, torch.tensor([len(lp)]),
                              torch.tensor([len(target)]), reduction="sum") / len(lp))
        frozen = view["log_probs"].float()
        preservation.append(F.kl_div(lp, frozen, log_target=True, reduction="sum") / len(lp))
    ctc = torch.stack(ctc).mean()
    preservation = torch.stack(preservation).mean()
    agreement = consistency_loss(*predictions, target) if consistency_weight else ctc * 0
    return dict(loss=ctc + preservation + consistency_weight * agreement,
                ctc=ctc, preservation=preservation, consistency=agreement)

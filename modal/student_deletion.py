"""Occurrence-normalized phone-deletion contrast on a frozen CTC encoder.

No inference rule changes: the learned affine correction is folded into the
ordinary phone output. Each negative retains every phone except one actually
transcribed consonant occurrence, including its neighbouring vowels.
"""
import hashlib
import math

CONSONANTS = frozenset("ءبتثجحخدذرزسشصضطظعغفقكلمنهويڇں۾")


def speaker_partition(speakers):
    if any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Every recording needs an identified training speaker")
    speakers = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(speakers) < 5:
        raise ValueError("Need at least five training speakers for disjoint selection")
    return {s: "selection" if i % 5 == 0 else "fit" for i, s in enumerate(speakers)}


def admitted_none(labels, excluded=()):
    """Scripts for mistakes and repeats are never used as performed transcripts."""
    excluded = set(excluded) | {r.get("speaker") for r in labels if r.get("split") != "train"}
    rows, seen = [], set()
    for r in labels:
        if r.get("split") != "train" or r.get("kind") != "none":
            continue
        if not r.get("speaker") or r["speaker"] in excluded:
            continue
        if (r.get("scored") is not True or r.get("exclude") or r.get("excluded") or
                any(r.get(k) != [] for k in ("events", "non_errors", "unsupported"))):
            continue
        if not r.get("id") or not r.get("wav") or not r.get("ayahs"):
            raise ValueError("None take lacks identity, audio or reference transcript")
        if r["id"] in seen:
            raise ValueError("Duplicate none take")
        seen.add(r["id"])
        rows.append(r)
    return rows


def deletion_targets(target, tokens):
    """One alternative per occurrence; identical negatives keep their multiplicity."""
    import torch
    if (target.ndim != 1 or target.dtype != torch.int64 or not len(target) or
            (target <= 0).any() or (target > len(tokens)).any()):
        raise ValueError("Need a complete non-blank transcript in the installed alphabet")
    positions = [i for i, label in enumerate(target.tolist()) if tokens[label - 1] in CONSONANTS]
    return positions, [torch.cat((target[:i], target[i + 1:])) for i in positions]


def sequence_log_prob(log_probs, targets):
    """Complete forward sums with shared emissions, preserving repeated-phone blanks."""
    import torch
    from torch.nn import functional as F
    if log_probs.ndim != 2 or not len(log_probs) or not torch.isfinite(log_probs).all():
        raise ValueError("Need finite full-utterance CTC evidence")
    if not targets:
        raise ValueError("No transcript hypotheses")
    for t in targets:
        if (t.ndim != 1 or t.dtype != torch.int64 or (t <= 0).any() or
                (t >= log_probs.shape[1]).any() or
                len(log_probs) < len(t) + int((t[1:] == t[:-1]).sum())):
            raise ValueError("Impossible transcript alignment")
    emissions = log_probs[:, None, :].expand(-1, len(targets), -1).contiguous()
    return -F.ctc_loss(emissions, torch.cat(targets),
        torch.full((len(targets),), len(log_probs), dtype=torch.int64),
        torch.tensor([len(t) for t in targets], dtype=torch.int64),
        blank=0, reduction="none", zero_infinity=False)


def weak_contrast(margins):
    """log(1 + mean(exp(-margin))); gradients emphasize weakest occurrences.

    Normalizing by occurrence count makes replicating the same comparisons
    neutral. Strong comparisons cannot cancel a weak one with an opposite sign.
    Computation stays stable even for very large likelihood ratios.
    """
    import torch
    if margins.ndim != 1 or not torch.isfinite(margins).all():
        raise ValueError("Need finite per-occurrence likelihood margins")
    if not len(margins):
        return margins.sum() * 0
    return torch.nn.functional.softplus(torch.logsumexp(-margins, 0) - math.log(len(margins)))


def objectives(predicted, target, shipped, tokens, contrast=False):
    import torch
    if predicted.shape != shipped.shape or predicted.shape[1] != len(tokens) + 1:
        raise ValueError("Head alphabet or preservation clock differs")
    if not torch.isfinite(shipped).all():
        raise ValueError("Non-finite preservation evidence")
    _, deleted = deletion_targets(target, tokens)
    # The ordinary arm has exactly the same transcript and preservation terms.
    scores = sequence_log_prob(predicted, [target, *deleted] if contrast else [target])
    ctc = -scores[0] / len(target)
    q = shipped.detach().log_softmax(-1)
    preserve = (q.exp() * (q - predicted)).sum(-1).mean()
    margins = scores[0] - scores[1:]
    deletion = weak_contrast(margins) if contrast else predicted.sum() * 0
    return ctc, preserve, deletion


def phone_head(width, vocab):
    import torch
    head = torch.nn.Linear(width, vocab)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    return head


def log_probs(head, features, shipped):
    return (shipped.detach() + head(features.detach())).log_softmax(-1)

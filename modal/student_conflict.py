"""Per-recording conflict filtering for a frozen-encoder phone-head refit.

The projection is over the complete head parameter vector, before averaging
recordings. It changes only the distillation gradient, not CTC or preservation.
Plain SGD preserves the geometry of that projection in the applied update.
"""
import hashlib
import math


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


def project_distillation(ctc, distillation):
    """Return g_kd - min(0, <g_kd,g_ctc>)/||g_ctc||² * g_ctc.

    The dot product spans weights AND bias; projecting individual tensors would
    change agreeing directions. Accumulate in float64 without an epsilon that
    could leave a negative component. A zero CTC gradient keeps KD unchanged.
    """
    import torch
    if not ctc or len(ctc) != len(distillation):
        raise ValueError("Need two gradients of the same complete head")
    if any(a.shape != b.shape or a.device != b.device or a.dtype != b.dtype
           for a, b in zip(ctc, distillation)):
        raise ValueError("Gradient parameter layouts differ")
    if any(not x.is_floating_point() or not torch.isfinite(x).all()
           for x in (*ctc, *distillation)):
        raise ValueError("Non-finite or non-floating recording gradient")
    dot = sum((a.double() * b.double()).sum() for a, b in zip(ctc, distillation))
    norm = sum(a.double().square().sum() for a in ctc)
    conflict = bool(dot < 0)
    if conflict:
        coefficient = dot / norm
        filtered = tuple((b.double() - coefficient * a.double()).to(b.dtype)
                         for a, b in zip(ctc, distillation))
    else:
        filtered = tuple(b.clone() for b in distillation)
    return filtered, {"dot": float(dot), "ctcNormSquared": float(norm), "conflict": conflict}


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
            not 0 < len(teacher) <= len(predicted) or shipped.shape != predicted.shape):
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
    kd = -(teacher.detach().log_softmax(-1).exp() * predicted[:len(teacher)]).sum(-1).mean()
    q = shipped.detach().log_softmax(-1)
    preserve = (q.exp() * (q - predicted)).sum(-1).mean()
    return ctc, kd, preserve


def recording_gradient(head, views, arm, kd_weight=1., preservation_weight=1.):
    import torch
    if arm not in ("ordinary", "filtered") or not views:
        raise ValueError("Need a matched arm and nonempty recording views")
    if any(not math.isfinite(w) or w < 0 for w in (kd_weight, preservation_weight)):
        raise ValueError("Objective weights must be finite and nonnegative")
    losses = [objectives(log_probs(head, v["features"], v["shipped"]),
                        v["target"], v["teacher"], v["shipped"]) for v in views]
    ctc, kd, preserve = [torch.stack([l[i] for l in losses]).mean() for i in range(3)]
    params = tuple(head.parameters())
    g_ctc = torch.autograd.grad(ctc, params, retain_graph=True)
    g_kd = torch.autograd.grad(kd_weight * kd, params, retain_graph=True)
    g_preserve = torch.autograd.grad(preservation_weight * preserve, params)
    projected, diagnostic = project_distillation(g_ctc, g_kd)
    used = projected if arm == "filtered" else g_kd
    diagnostic.update(ctc=float(ctc.detach()), kd=float(kd.detach()),
                      preservation=float(preserve.detach()))
    return tuple(a + b + c for a, b, c in zip(g_ctc, used, g_preserve)), diagnostic

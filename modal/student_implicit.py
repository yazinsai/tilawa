"""Implicit updates of an existing frozen-encoder CTC phone head.

Every proximal solve evaluates the full recording objective at the candidate
parameters. It is a bounded local solve of a nonconvex objective, not a global
minimizer or a claim of safe recognition. Explicit controls share the objective,
views, recording weights and outer schedule; the multi-step control additionally
matches each solve's objective/gradient evaluation count.
"""
import copy
import hashlib
import math
import random

import torch
from torch.nn import functional as F


def speaker_partition(speakers):
    if any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Every recording needs an identified training speaker")
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(unique) < 5:
        raise ValueError("Need at least five training speakers")
    return {s: "selection" if i % 5 == 0 else "fit" for i, s in enumerate(unique)}


def admitted_none(labels, excluded=()):
    excluded = set(excluded) | {r.get("speaker") for r in labels if r.get("split") != "train"}
    selected, seen = [], set()
    for r in labels:
        if r.get("split") != "train" or r.get("kind") != "none" or not r.get("speaker") or r["speaker"] in excluded:
            continue
        if r.get("scored") is not True or any(r.get(k) != [] for k in ("events", "non_errors", "unsupported")) or r.get("excluded") or r.get("exclude"):
            continue
        if not r.get("id") or not r.get("wav") or not r.get("ayahs"):
            raise ValueError("None take lacks identity, audio or transcript")
        if r["id"] in seen:
            raise ValueError("Duplicate none take")
        seen.add(r["id"])
        selected.append(r)
    return selected


def phone_head(weight, bias):
    weight, bias = torch.as_tensor(weight), torch.as_tensor(bias)
    if weight.ndim != 2 or bias.shape != (len(weight),) or not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("Invalid native affine head")
    head = torch.nn.Linear(weight.shape[1], weight.shape[0], dtype=torch.float64)
    with torch.no_grad():
        head.weight.copy_(weight)
        head.bias.copy_(bias)
    return head


def log_probs(head, features):
    return head(features.detach().to(head.weight.dtype)).log_softmax(-1)


def recording_loss(head, views):
    """Equal views of one recording: complete CTC plus shipped-output KL.

    KL includes every valid frame, including ordinary speech and real pauses.
    No alignment is fixed and no impossible transcript is silently zeroed.
    """
    if not views:
        raise ValueError("Missing recording views")
    losses = []
    for v in views:
        lp, target = log_probs(head, v["features"]), v["target"]
        if target.ndim != 1 or target.dtype != torch.int64 or not len(target) or (target <= 0).any() or (target >= lp.shape[1]).any():
            raise ValueError("Invalid full transcript")
        if len(lp) < len(target) + int((target[1:] == target[:-1]).sum()):
            raise ValueError("Impossible complete transcript")
        q = v["shipped"].detach().to(lp.dtype)
        if q.shape != lp.shape or not torch.isfinite(q).all():
            raise ValueError("Shipped preservation clock differs")
        q = q.log_softmax(-1)
        ctc = F.ctc_loss(lp[:, None], target, torch.tensor([len(lp)]),
                         torch.tensor([len(target)]), reduction="sum", zero_infinity=False) / len(target)
        preserve = (q.exp() * (q - lp)).sum(-1).mean()
        losses.append(ctc + preserve)
    value = torch.stack(losses).mean()
    if not torch.isfinite(value):
        raise ValueError("Non-finite complete recording objective")
    return value


def distance(head, preceding):
    return sum((p - old).square().sum() for p, old in zip(head.parameters(), preceding))


def implicit_update(head, objective, tau, max_evaluations=32, tolerance=1e-6):
    """Approximately solve L(theta) + ||theta-theta_old||²/(2*tau).

    L-BFGS uses backtracking at proposed parameters. The final objective and
    stationarity residual are measured explicitly. An unconverged or increasing
    solve rolls back; it never falls back to a mislabeled explicit update.
    All value-and-gradient calls (including diagnostics) count as compute.
    """
    if not math.isfinite(tau) or tau <= 0 or max_evaluations < 3 or tolerance <= 0:
        raise ValueError("Invalid proximal solve schedule")
    previous = [p.detach().clone() for p in head.parameters()]
    count, initial, initial_grad = 0, None, None
    def closure():
        nonlocal count, initial, initial_grad
        head.zero_grad(set_to_none=True)
        loss = objective(head)
        total = loss + distance(head, previous) / (2 * tau)
        total.backward()
        count += 1
        grad = torch.cat([p.grad.flatten() for p in head.parameters()])
        if not torch.isfinite(total) or not torch.isfinite(grad).all():
            raise ValueError("Non-finite proximal objective or gradient")
        if initial is None:
            initial, initial_grad = float(total.detach()), float(grad.abs().max())
        return total
    # Strong-Wolfe may exceed a requested evaluation budget by a line search;
    # count actual calls, then match that work in the explicit control.
    optimizer = torch.optim.LBFGS(head.parameters(), lr=1., max_iter=max_evaluations - 2,
        max_eval=max_evaluations - 1, tolerance_grad=tolerance,
        tolerance_change=1e-14, history_size=10, line_search_fn="strong_wolfe")
    try:
        optimizer.step(closure)
        final = float(closure().detach())
        residual = max(float(p.grad.abs().max()) for p in head.parameters())
        accepted = final <= initial + 1e-12 and residual <= tolerance * max(1., initial_grad)
    except (ValueError, RuntimeError):
        with torch.no_grad():
            for p, old in zip(head.parameters(), previous):
                p.copy_(old)
        raise
    if not accepted:
        with torch.no_grad():
            for p, old in zip(head.parameters(), previous):
                p.copy_(old)
    return dict(evaluations=count, accepted=accepted, initial=initial, proposed=final,
                residual=residual, distance=float(distance(head, previous).detach()))


def explicit_update(head, objective, tau, evaluations=1):
    if evaluations < 1 or not math.isfinite(tau) or tau <= 0:
        raise ValueError("Invalid explicit update schedule")
    # Each multi-step solve traverses the same outer time interval as one
    # ordinary step. Extra compute is spent resolving the explicit trajectory.
    for _ in range(evaluations):
        head.zero_grad(set_to_none=True)
        loss = objective(head)
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in head.parameters()):
            raise ValueError("Non-finite explicit gradient")
        with torch.no_grad():
            for p in head.parameters():
                p.sub_(p.grad * (tau / evaluations))


def matched_refit(initial, rows, groups, validate, schedules, epochs=8, seed=1729, emit=None):
    """Select a proximal schedule and shared checkpoint on training speakers.

    Schedule candidates are fixed before practice. Each candidate is a triple
    of implicit/ordinary/multi-step heads. Selection minimizes their mean
    equal-speaker objective at one shared epoch, including epoch zero.
    """
    if not schedules or len(set(schedules)) != len(schedules) or epochs < 1:
        raise ValueError("Invalid candidate schedule")
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    if not fit or not selection or {r["speaker"] for r in fit} & {r["speaker"] for r in selection}:
        raise ValueError("Need disjoint speaker-grouped partitions")
    heads = {tau: {a: copy.deepcopy(initial) for a in ("implicit", "ordinary", "multi-step")} for tau in schedules}
    initial_metrics = validate(initial, selection)
    best = (initial_metrics["objective"], schedules[0], 0)
    saved = copy.deepcopy(heads[schedules[0]])
    history, ledger = [], []
    for epoch in range(1, epochs + 1):
        order = list(range(len(fit)))
        random.Random(seed + epoch).shuffle(order)
        for tau, arms in heads.items():
            stats = []
            for i in order:
                objective = lambda h: recording_loss(h, fit[i]["views"])
                diag = implicit_update(arms["implicit"], objective, tau)
                explicit_update(arms["ordinary"], objective, tau)
                explicit_update(arms["multi-step"], objective, tau, diag["evaluations"])
                stats.append(diag)
                ledger.append(dict(epoch=epoch, tau=tau, recording=i,
                    implicit=diag["evaluations"], ordinary=1, multiStep=diag["evaluations"], accepted=diag["accepted"]))
            metrics = {a: validate(h, selection) for a, h in arms.items()}
            score = sum(m["objective"] for m in metrics.values()) / len(metrics)
            item = dict(epoch=epoch, tau=tau, metrics=metrics, selectionObjective=score,
                accepted=sum(d["accepted"] for d in stats), solves=len(stats),
                maxResidual=max(d["residual"] for d in stats))
            history.append(item)
            if emit:
                emit(item)
            if score < best[0]:
                best, saved = (score, tau, epoch), copy.deepcopy(arms)
    for h in saved.values():
        h.eval()
    return saved, dict(selectedEpoch=best[2], selectedTau=best[1], selectionObjective=best[0],
        epochZero=initial_metrics, history=history, ledger=ledger,
        selectionRule="minimum-triple-mean-complete-objective-at-shared-epoch-on-training-selection-speakers")

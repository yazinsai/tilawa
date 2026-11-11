"""Frozen-feature linear-head refitting with a damped Kronecker Fisher.

The factors approximate framewise categorical curvature, not the exact CTC
Hessian: temporal correlations and feature/output dependence are omitted.
Recordings and their fixed augmented views receive equal weight in both the
objective and factor estimate. Only fit speakers contribute to the factors.
"""
import copy
import hashlib
import math
import random

import numpy as np


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
                ("events", "non_errors", "unsupported")) or row.get("excluded") or row.get("exclude")):
            continue
        if not row.get("id") or not row.get("wav") or not row.get("ayahs"):
            raise ValueError("None take lacks identity, audio or transcript")
        if row["id"] in seen:
            raise ValueError("Duplicate none take")
        seen.add(row["id"])
        selected.append(row)
    return selected


def view_factors(features, probabilities):
    x, p = np.asarray(features, dtype=np.float64), np.asarray(probabilities, dtype=np.float64)
    if (x.ndim != 2 or p.ndim != 2 or not len(x) or len(x) != len(p) or
            not np.isfinite(x).all() or not np.isfinite(p).all() or (p < 0).any() or
            not np.allclose(p.sum(1), 1., atol=1e-6, rtol=1e-6)):
        raise ValueError("Need finite same-clock features and categorical probabilities")
    z = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    return z.T @ z / len(z), np.diag(p.mean(0)) - p.T @ p / len(p)


def estimate_factors(head, rows, groups):
    import torch
    selected = [r for r in rows if groups[r["speaker"]] == "fit"]
    if not selected:
        raise ValueError("Curvature estimation needs fit speakers")
    a, g = [], []
    with torch.no_grad():
        for r in selected:
            factors = [view_factors(v["features"].numpy(),
                       log_probs(head, v["features"]).exp().numpy()) for v in r["views"]]
            a.append(np.mean([f[0] for f in factors], axis=0))
            g.append(np.mean([f[1] for f in factors], axis=0))
    return np.mean(a, axis=0), np.mean(g, axis=0)


class KroneckerPreconditioner:
    """Solve (G ⊗ A + damping I) on the complete weight AND bias gradient.

    Normalize curvature by its mean eigenvalue so LR retains ordinary SGD
    units. Damping is a ratio of that scale, selected on training speakers.
    Frozen factors avoid arm-dependent examples or adaptive curvature sampling.
    """
    def __init__(self, a, g, damping_ratio):
        if not math.isfinite(damping_ratio) or damping_ratio <= 0:
            raise ValueError("Damping ratio must be positive and finite")
        spectra = []
        for factor in (a, g):
            factor = np.asarray(factor, dtype=np.float64)
            if (factor.ndim != 2 or factor.shape[0] != factor.shape[1] or
                    not len(factor) or not np.isfinite(factor).all() or
                    not np.allclose(factor, factor.T, atol=1e-10, rtol=1e-10)):
                raise ValueError("Curvature factors must be finite symmetric matrices")
            values, vectors = np.linalg.eigh((factor + factor.T) / 2)
            if values.min() < -1e-10 * max(1., abs(values).max()):
                raise ValueError("Curvature factor is not positive semidefinite")
            spectra.append((np.maximum(values, 0.), vectors))
        (self.ea, self.ua), (self.eg, self.ug) = spectra
        self.scale = float(self.ea.mean() * self.eg.mean())
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Curvature has no positive scale")
        self.ratio = damping_ratio
        self.denominator = self.eg[:, None] * self.ea[None, :] + damping_ratio * self.scale

    def apply(self, gradient):
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.shape != self.denominator.shape or not np.isfinite(gradient).all():
            raise ValueError("Need finite complete weight-and-bias gradient")
        rotated = self.ug.T @ gradient @ self.ua
        result = self.scale * (self.ug @ (rotated / self.denominator) @ self.ua.T)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite preconditioned gradient")
        return result


def phone_head(weight, bias):
    import torch
    weight, bias = np.asarray(weight), np.asarray(bias)
    if weight.ndim != 2 or bias.shape != (len(weight),) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Invalid native linear head")
    head = torch.nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        head.weight.copy_(torch.tensor(weight.copy()))
        head.bias.copy_(torch.tensor(bias.copy()))
    return head


def log_probs(head, features):
    return head(features.detach()).log_softmax(-1)


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
    kd = -(teacher.detach().log_softmax(-1).exp() * predicted[:len(teacher)]).sum(-1).mean()
    q = shipped.detach().log_softmax(-1)
    preserve = (q.exp() * (q - predicted)).sum(-1).mean()
    return ctc, kd, preserve


def recording_loss(head, views):
    import torch
    if not views:
        raise ValueError("Need nonempty fixed recording views")
    losses = [objectives(log_probs(head, v["features"]), v["target"],
                        v["teacher"], v["shipped"]) for v in views]
    components = [torch.stack([loss[i] for loss in losses]).mean() for i in range(3)]
    return sum(components), dict(zip(("ctc", "kd", "preservation"),
                                    (float(x.detach()) for x in components)))


def matched_refit(initial, rows, groups, validate, ratios, epochs, lr, seed, emit=None):
    """Select damping and ONE shared epoch using equal-speaker validation.

    Every candidate and SGD control sees exactly the same ordered recordings,
    fixed views and objective, with one update per recording. No practice input
    enters here. Validation callback reads the selection partition only.
    """
    import torch
    if epochs < 1 or lr <= 0 or not math.isfinite(lr) or not ratios or len(set(ratios)) != len(ratios):
        raise ValueError("Invalid matched training schedule")
    a, g = estimate_factors(initial, rows, groups)
    solvers = {str(r): KroneckerPreconditioner(a, g, r) for r in ratios}
    heads = {arm: copy.deepcopy(initial) for arm in ("ordinary", *solvers)}
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    if not selection:
        raise ValueError("Need separate training-speaker selection recordings")
    history = {arm: [dict(epoch=0, **validate(head, selection))] for arm, head in heads.items()}
    best_states = {arm: copy.deepcopy(head.state_dict()) for arm, head in heads.items()}
    chosen_epoch, chosen_ratio = 0, str(ratios[0])
    best_score = history["ordinary"][0]["objective"]
    for epoch in range(1, epochs + 1):
        order = list(range(len(fit)))
        random.Random(seed + epoch).shuffle(order)
        for arm, head in heads.items():
            diagnostics = []
            for i in order:
                head.zero_grad(set_to_none=True)
                loss, diag = recording_loss(head, fit[i]["views"])
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                    raise ValueError("Non-finite applied head gradient")
                gradient = torch.cat([head.weight.grad, head.bias.grad[:, None]], dim=1)
                if arm != "ordinary":
                    gradient = torch.tensor(solvers[arm].apply(gradient.numpy()), dtype=gradient.dtype)
                with torch.no_grad():
                    head.weight.sub_(lr * gradient[:, :-1])
                    head.bias.sub_(lr * gradient[:, -1])
                diagnostics.append(diag)
            metrics = dict(epoch=epoch, **validate(head, selection), recordings=len(order),
                           losses={k: float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]})
            history[arm].append(metrics)
            if emit:
                emit(dict(arm=arm, **metrics))
        for ratio in solvers:
            score = (history["ordinary"][-1]["objective"] + history[ratio][-1]["objective"]) / 2
            if score < best_score:
                best_score, chosen_epoch, chosen_ratio = score, epoch, ratio
                best_states = {arm: copy.deepcopy(head.state_dict()) for arm, head in heads.items()}
    selected = {}
    for arm, key in (("ordinary", "ordinary"), ("curvature", chosen_ratio)):
        selected[arm] = copy.deepcopy(initial)
        selected[arm].load_state_dict(best_states[key])
        selected[arm].eval()
    return selected, dict(history=history, selectedEpoch=chosen_epoch,
        dampingRatio=float(chosen_ratio), selectionObjective=float(best_score),
        selectedUpdates=chosen_epoch * len(fit), trainingUpdates=epochs * len(fit),
        curvatureScale=solvers[chosen_ratio].scale,
        factorEigenvalues={"features": solvers[chosen_ratio].ea.tolist(), "outputs": solvers[chosen_ratio].eg.tolist()},
        selectionRule="minimum-pair-mean-objective-at-shared-epoch-on-selection-speakers"), (a, g)

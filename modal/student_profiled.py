"""Variable projection of the phone classifier over a nonlinear acoustic adapter.

All observations are frozen training caches. Exact marginalized CTC, teacher
CE, shipped-output KL, and parameter preservation are shared by both arms.
At a stationary head, the envelope gradient differentiates through the fitted
classifier with respect to the adapter; it does not backpropagate optimizer
iterations. An approximate stationary point is not a global CTC minimum.
"""
import copy
import hashlib
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


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
        if r.get("split") != "train" or r.get("kind") != "none":
            continue
        if not r.get("speaker") or r["speaker"] in excluded:
            continue
        if (r.get("scored") is not True or any(r.get(k) != [] for k in
                ("events", "non_errors", "unsupported")) or r.get("excluded") or r.get("exclude")):
            continue
        if not r.get("id") or not r.get("wav") or not r.get("ayahs"):
            raise ValueError("None take lacks identity, audio or transcript")
        if r["id"] in seen:
            raise ValueError("Duplicate none take")
        seen.add(r["id"])
        selected.append(r)
    return selected


def ctc_statistics(lp, target):
    """Exact log likelihood and frame-class occupancies, including blank."""
    lp, y = np.asarray(lp, dtype=np.float64), np.asarray(target)
    if (lp.ndim != 2 or not len(lp) or lp.shape[1] < 2 or not np.isfinite(lp).all() or
            y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or
            (y <= 0).any() or (y >= lp.shape[1]).any() or
            len(lp) < len(y) + int((y[1:] == y[:-1]).sum()) or
            not np.allclose(logsumexp(lp, axis=-1), 0., atol=1e-6)):
        raise ValueError("Invalid or impossible complete CTC transcript")
    labels = np.zeros(2 * len(y) + 1, dtype=np.int64)
    labels[1::2] = y
    skip = np.zeros(len(labels), dtype=bool)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    emission = lp[:, labels]
    alpha = np.full(emission.shape, -np.inf)
    alpha[0, :min(2, len(labels))] = emission[0, :min(2, len(labels))]
    for t in range(1, len(lp)):
        previous = alpha[t - 1]
        jump = np.full(len(labels), -np.inf)
        jump[2:] = np.where(skip[2:], previous[:-2], -np.inf)
        alpha[t] = np.logaddexp(np.logaddexp(previous, np.r_[-np.inf, previous[:-1]]), jump) + emission[t]
    terminal = slice(max(0, len(labels) - 2), len(labels))
    z = logsumexp(alpha[-1, terminal])
    beta = np.full(emission.shape, -np.inf)
    beta[-1, terminal] = 0.
    for t in range(len(lp) - 2, -1, -1):
        following = beta[t + 1] + emission[t + 1]
        jump = np.full(len(labels), -np.inf)
        jump[:-2] = np.where(skip[2:], following[2:], -np.inf)
        beta[t] = np.logaddexp(np.logaddexp(following, np.r_[following[1:], -np.inf]), jump)
    gamma = np.zeros_like(lp)
    for state, label in enumerate(labels):
        gamma[:, label] += np.exp(alpha[:, state] + beta[:, state] - z)
    if not np.isfinite(z) or not np.allclose(gamma.sum(1), 1., atol=2e-8):
        raise ArithmeticError("CTC occupancy failed normalization")
    return float(z), gamma


@dataclass
class AdapterHead:
    down: np.ndarray                 # H x D
    offset: np.ndarray               # H
    up: np.ndarray                   # D x H
    weight: np.ndarray               # V x D
    bias: np.ndarray                 # V

    def vector(self, block="all"):
        names = ("down", "offset", "up") if block == "adapter" else ("weight", "bias") if block == "head" else self.__dataclass_fields__
        return np.concatenate([getattr(self, k).ravel() for k in names])

    def with_vector(self, vector, block="all"):
        result = copy.deepcopy(self)
        names = ("down", "offset", "up") if block == "adapter" else ("weight", "bias") if block == "head" else self.__dataclass_fields__
        start = 0
        for k in names:
            p = getattr(self, k)
            setattr(result, k, vector[start:start + p.size].reshape(p.shape).copy())
            start += p.size
        if start != len(vector) or not np.isfinite(vector).all():
            raise ValueError("Invalid adapter/head parameter vector")
        return result


def initialize(weight, bias, hidden=32, seed=1729):
    w, b = np.asarray(weight, dtype=np.float64), np.asarray(bias, dtype=np.float64)
    if w.ndim != 2 or b.shape != (len(w),) or hidden < 1 or not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError("Invalid shipped phone head")
    rng = np.random.default_rng(seed)
    return AdapterHead(rng.normal(0., 1. / math.sqrt(w.shape[1]), (hidden, w.shape[1])),
                       np.zeros(hidden), np.zeros((w.shape[1], hidden)), w.copy(), b.copy())


def log_probs(model, features):
    hidden = np.tanh(features @ model.down.T + model.offset)
    adapted = features + hidden @ model.up.T
    scores = adapted @ model.weight.T + model.bias
    return scores - logsumexp(scores, axis=-1, keepdims=True)


def corpus_value_gradient(model, initial, rows, regularization, deadline=math.inf):
    """Equal recordings/views; all real frames, including pauses, preserved.

    CTC is normalized by transcript length. Teacher CE and shipped KL average
    real frames on their native common 40 ms clock; no duration clipping or
    impossible-path zeroing changes a recording's contribution.
    """
    if not rows or regularization <= 0 or not math.isfinite(regularization):
        raise ValueError("Need fit recordings and positive preservation regularization")
    gradients = AdapterHead(*(np.zeros_like(getattr(model, k)) for k in model.__dataclass_fields__))
    value = 0.
    for row in rows:
        if not row["views"]:
            raise ValueError("Empty recording views")
        scale = 1. / (len(rows) * len(row["views"]))
        for v in row["views"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("One-day experiment budget exhausted")
            x, y = v["features"], v["target"]
            hidden = np.tanh(x @ model.down.T + model.offset)
            adapted = x + hidden @ model.up.T
            scores = adapted @ model.weight.T + model.bias
            lp = scores - logsumexp(scores, axis=-1, keepdims=True)
            z, gamma = ctc_statistics(lp, y)
            teacher, shipped = v["teacher"], v["shipped"]
            if (not len(y) or shipped.shape != lp.shape or teacher.ndim != 2 or
                    teacher.shape[1] != lp.shape[1] or not 0 < len(teacher) <= len(lp)):
                raise ValueError("Transcript, posterior clock or inventory differs")
            tq = np.exp(teacher - logsumexp(teacher, axis=-1, keepdims=True))
            sq = shipped - logsumexp(shipped, axis=-1, keepdims=True)
            p = np.exp(lp)
            dz = (p - gamma) / len(y) + (p - np.exp(sq)) / len(lp)
            dz[:len(tq)] += (p[:len(tq)] - tq) / len(tq)
            loss = -z / len(y) - (tq * lp[:len(tq)]).sum() / len(tq) + (np.exp(sq) * (sq - lp)).sum() / len(lp)
            value += scale * loss
            gradients.weight += scale * dz.T @ adapted
            gradients.bias += scale * dz.sum(0)
            da = dz @ model.weight
            gradients.up += scale * da.T @ hidden
            dh = (da @ model.up) * (1. - hidden ** 2)
            gradients.down += scale * dh.T @ x
            gradients.offset += scale * dh.sum(0)
    delta = model.vector() - initial.vector()
    gradient = gradients.vector() + regularization * delta
    value += .5 * regularization * float(delta @ delta)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ArithmeticError("Non-finite acoustic objective")
    return float(value), gradients.with_vector(gradient)


def minimize(initial, evaluate, allowance, deadline, tolerance=1e-5):
    """Bound objective evaluations; retain the best complete finite iterate."""
    from scipy.optimize import minimize as lbfgs
    best, count = [None], [0]
    class BudgetReached(Exception):
        pass
    def closure(vector):
        if time.monotonic() >= deadline:
            raise TimeoutError("One-day experiment budget exhausted")
        if count[0] >= allowance:
            raise BudgetReached()
        value, gradient = evaluate(vector)
        count[0] += 1
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise ArithmeticError("Non-finite optimizer closure")
        if best[0] is None or value < best[0][0]:
            best[0] = (float(value), vector.copy(), gradient.copy())
        return value, gradient
    try:
        result = lbfgs(closure, initial.copy(), jac=True, method="L-BFGS-B", options=dict(
            gtol=tolerance, ftol=1e-15, maxfun=allowance, maxiter=allowance, maxls=40, maxcor=10))
        # A last stationary iterate may tie an earlier value within roundoff.
        # Keep its more accurate derivative rather than misreporting the old
        # iterate as unconverged after a successful gradient termination.
        if (best[0] is not None and result.fun <= best[0][0] + 1e-12 and
                np.abs(result.jac).max() < np.abs(best[0][2]).max()):
            best[0] = (float(result.fun), result.x.copy(), result.jac.copy())
        reason = str(result.message)
    except BudgetReached:
        reason = "objective evaluation budget"
    if best[0] is None:
        raise ValueError("No complete objective evaluation")
    value, vector, gradient = best[0]
    norm = float(np.abs(gradient).max())
    return vector, dict(evaluations=count[0], objective=value, gradientInfinityNorm=norm,
                        converged=norm <= tolerance, termination=reason)


def profile(model, initial, rows, regularization, allowance, deadline, tolerance=1e-5):
    """Optimize only the head, then return the envelope adapter gradient."""
    def evaluate(vector):
        value, gradient = corpus_value_gradient(model.with_vector(vector, "head"), initial, rows, regularization, deadline)
        return value, gradient.vector("head")
    vector, report = minimize(model.vector("head"), evaluate, allowance, deadline, tolerance)
    if not report["converged"]:
        raise RuntimeError("Head profiling did not reach stationarity: " + report["termination"] +
                           f'; gradient infinity norm {report["gradientInfinityNorm"]:.3g}')
    fitted = model.with_vector(vector, "head")
    value, gradient = corpus_value_gradient(fitted, initial, rows, regularization, deadline)
    report["evaluations"] += 1  # Include the envelope-gradient pass in control allowance.
    return fitted, value, gradient.vector("adapter"), report


def matched_fit(initial, rows, groups, validate, regularizations=(.001, .01, .1),
                rounds=4, outer_evaluations=12, head_evaluations=512, seconds=86400., emit=None):
    """Paired L-BFGS fits; shared phase/regularization selected on training speakers.

    The implementation computes complete gradients in all closures, including
    head-only fits. Each complete corpus evaluation therefore has identical
    adapter/projection/lattice work. The joint arm receives precisely the same
    evaluation allowance. This is a compute proxy, not measured hardware FLOPs.
    """
    if rounds < 1 or outer_evaluations < 1 or head_evaluations < 1 or not 0 < seconds <= 86400:
        raise ValueError("Invalid one-day fitting schedule")
    expected = speaker_partition([r["speaker"] for r in rows])
    if groups != expected:
        raise ValueError("Training speaker partitions changed")
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    deadline = time.monotonic() + seconds
    history, best_score, best_models, chosen = [], math.inf, None, None
    for reg in regularizations:
        models = {arm: copy.deepcopy(initial) for arm in ("profiled", "joint")}
        total = {arm: 0 for arm in models}
        for phase in range(rounds + 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("One-day experiment budget exhausted")
            diagnostics = {}
            if phase:
                current, fitted_iterates, inner_reports = models["profiled"], {}, []
                def evaluate_adapter(vector):
                    nonlocal current
                    proposed = current.with_vector(vector, "adapter")
                    fitted, value, gradient, report = profile(proposed, initial, fit, reg, head_evaluations, deadline)
                    total["profiled"] += report["evaluations"]
                    inner_reports.append(report)
                    fitted_iterates[vector.tobytes()] = fitted
                    current = fitted  # Warm start head; all head parameters are still reoptimized.
                    return value, gradient
                vector, outer = minimize(models["profiled"].vector("adapter"), evaluate_adapter,
                                         outer_evaluations, deadline)
                models["profiled"] = fitted_iterates[vector.tobytes()]
                current = models["joint"]
                def evaluate_joint(vector):
                    value, gradient = corpus_value_gradient(current.with_vector(vector), initial, fit, reg, deadline)
                    return value, gradient.vector()
                allowance = total["profiled"] - total["joint"]
                vector, joint = minimize(current.vector(), evaluate_joint, allowance, deadline)
                models["joint"] = current.with_vector(vector)
                total["joint"] += joint["evaluations"]
                diagnostics = dict(profiled=dict(outer=outer, headFits=inner_reports), joint=joint)
            metrics = {arm: validate(model, selection, initial, reg) for arm, model in models.items()}
            if time.monotonic() >= deadline:
                raise TimeoutError("One-day experiment budget exhausted during selection")
            pair_score = float(np.mean([m["objective"] for m in metrics.values()]))
            entry = dict(phase=phase, regularization=reg, metrics=metrics,
                         diagnostics=diagnostics, corpusEvaluations=total.copy())
            history.append(entry)
            if emit:
                emit(entry)
            if pair_score < best_score:
                best_score, best_models = pair_score, copy.deepcopy(models)
                chosen = dict(phase=phase, regularization=reg, selectionObjective=pair_score)
    if best_models is None:
        raise ValueError("No selection checkpoint")
    return best_models, dict(history=history, selected=chosen,
        selectionRule="minimum-pair-mean-objective-at-shared-phase-and-regularization-on-selection-speakers",
        computeRule="identical-complete-gradient-corpus-evaluation-allowance; joint-may-stop-stationary")

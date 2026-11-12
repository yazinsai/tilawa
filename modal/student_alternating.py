"""Whole-corpus CTC expectation / converged affine-head maximization.

No audio, references, or evaluation incidents enter the optimizer. All views
are cached once, and a phase's path assignments are immutable. The ordinary
control refreshes its assignments on every complete objective evaluation.
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


@dataclass
class PhoneHead:
    weight: np.ndarray
    bias: np.ndarray

    def vector(self):
        return np.concatenate([self.weight.ravel(), self.bias])

    def with_vector(self, vector):
        n = self.weight.size
        return PhoneHead(vector[:n].reshape(self.weight.shape).copy(), vector[n:].copy())


def phone_head(weight, bias):
    w, b = np.asarray(weight, dtype=np.float64), np.asarray(bias, dtype=np.float64)
    if w.ndim != 2 or b.shape != (len(w),) or not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError("Invalid native affine head")
    return PhoneHead(w.copy(), b.copy())


def log_probs(head, features):
    logits = np.asarray(features) @ head.weight.T + head.bias
    return logits - logsumexp(logits, axis=-1, keepdims=True)


def forward_backward(lp, target):
    """Return log P(transcript), frame-class occupancies, including blank.

    Alpha includes this frame's emission; beta excludes it. Repeated labels
    cannot skip the intervening blank. Empty transcripts have one blank path.
    Reject impossible alignments instead of silently dropping their loss.
    """
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
        stay = alpha[t - 1]
        step = np.r_[-np.inf, stay[:-1]]
        jump = np.full(len(labels), -np.inf)
        if len(labels) > 2:
            jump[2:] = np.where(skip[2:], stay[:-2], -np.inf)
        alpha[t] = np.logaddexp(np.logaddexp(stay, step), jump) + emission[t]
    terminal = slice(max(0, len(labels) - 2), len(labels))
    log_z = logsumexp(alpha[-1, terminal])
    if not np.isfinite(log_z):
        raise ValueError("No finite CTC path")
    beta = np.full(emission.shape, -np.inf)
    beta[-1, terminal] = 0.
    for t in range(len(lp) - 2, -1, -1):
        following = beta[t + 1] + emission[t + 1]
        step = np.r_[following[1:], -np.inf]
        jump = np.full(len(labels), -np.inf)
        if len(labels) > 2:
            jump[:-2] = np.where(skip[2:], following[2:], -np.inf)
        beta[t] = np.logaddexp(np.logaddexp(following, step), jump)
    posterior = np.exp(alpha + beta - log_z)
    gamma = np.zeros_like(lp)
    for s, label in enumerate(labels):
        gamma[:, label] += posterior[:, s]
    if not np.allclose(gamma.sum(1), 1., atol=2e-8, rtol=2e-8):
        raise ArithmeticError("Forward-backward occupancy failed normalization")
    return float(log_z), gamma


def assignments(head, rows, deadline=math.inf):
    """One complete E step; return only after every fit view is assigned."""
    result = []
    for row in rows:
        views = []
        for view in row["views"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("One-day fitting budget exhausted")
            _, gamma = forward_backward(log_probs(head, view["features"]), view["target"])
            gamma.setflags(write=False)
            views.append(gamma)
        result.append(tuple(views))
    return tuple(result)


def corpus_value_gradient(head, initial, rows, regularization, fixed=None):
    """Equal recordings, equal fixed views; same preservation in both arms.

    Fixed occupancy CE is the CTC EM surrogate. Its T/U scaling matches the
    exact transcript loss -log P(y)/U, rather than changing duration weights.
    Teacher CE and shipped KL average all real frames, including real pauses.
    """
    if not rows or not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Need fit recordings and positive head regularization")
    if fixed is not None and (len(fixed) != len(rows) or
            any(len(q) != len(r["views"]) for q, r in zip(fixed, rows))):
        raise ValueError("Assignments do not cover the whole fitting corpus")
    dw, db, value = np.zeros_like(head.weight), np.zeros_like(head.bias), 0.
    for i, row in enumerate(rows):
        if not row["views"]:
            raise ValueError("Empty recording views")
        scale = 1. / (len(rows) * len(row["views"]))
        for j, v in enumerate(row["views"]):
            x, y = v["features"], v["target"]
            lp = log_probs(head, x)
            if not len(y):
                raise ValueError("Supervised recordings need a performed transcript")
            if fixed is None:
                z, gamma = forward_backward(lp, y)
                ctc = -z / len(y)
            else:
                gamma = fixed[i][j]
                if gamma.shape != lp.shape or gamma.flags.writeable:
                    raise ValueError("Need immutable same-clock assignments")
                ctc = -float((gamma * lp).sum()) / len(y)
            teacher, shipped = v["teacher"], v["shipped"]
            if (shipped.shape != lp.shape or teacher.ndim != 2 or
                    teacher.shape[1] != lp.shape[1] or not 0 < len(teacher) <= len(lp)):
                raise ValueError("Preservation posterior clocks or inventory differ")
            tq = np.exp(teacher - logsumexp(teacher, axis=-1, keepdims=True))
            sq = shipped - logsumexp(shipped, axis=-1, keepdims=True)
            p = np.exp(lp)
            gradient = (p - gamma) / len(y) + (p - np.exp(sq)) / len(lp)
            gradient[:len(tq)] += (p[:len(tq)] - tq) / len(tq)
            loss = ctc - (tq * lp[:len(tq)]).sum() / len(tq) + (np.exp(sq) * (sq - lp)).sum() / len(lp)
            value += scale * loss
            dw += scale * (gradient.T @ x)
            db += scale * gradient.sum(0)
    delta = head.vector() - initial.vector()
    value += .5 * regularization * float(delta @ delta)
    gradient = np.r_[dw.ravel(), db] + regularization * delta
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ArithmeticError("Non-finite acoustic objective")
    return float(value), gradient


def minimize_head(initial_vector, evaluate, max_evaluations, deadline, tolerance=1e-5):
    """Deterministic full-data L-BFGS with bounded line-search evaluations.

    The M step must report stationarity; a budget or line-search termination
    is never described as convergence. The control has the identical solver.
    """
    from scipy.optimize import minimize
    best = [None]
    evaluations = [0]
    history = []

    class BudgetReached(Exception):
        pass

    def closure(vector):
        if time.monotonic() >= deadline:
            raise TimeoutError("One-day fitting budget exhausted")
        if evaluations[0] >= max_evaluations:
            raise BudgetReached()
        value, grad = evaluate(vector)
        evaluations[0] += 1
        if best[0] is None or value < best[0][0]:
            best[0] = (value, vector.copy(), grad.copy())
        history.append(float(value))
        return value, grad

    try:
        result = minimize(closure, initial_vector.copy(), method="L-BFGS-B", jac=True,
            options=dict(gtol=tolerance, ftol=1e-15, maxiter=max_evaluations,
                         maxfun=max_evaluations, maxls=40, maxcor=20))
        if result.fun <= best[0][0] + 1e-12:
            best[0] = (float(result.fun), result.x.copy(), result.jac.copy())
        reason = str(result.message)
    except BudgetReached:
        reason = "matched-compute evaluation budget"
    if best[0] is None:
        raise ValueError("No complete objective evaluation was possible")
    value, vector, grad = best[0]
    norm = float(np.abs(grad).max())
    return vector, dict(evaluations=evaluations[0], objective=value,
        gradientInfinityNorm=norm, converged=norm <= tolerance,
        termination=reason, evaluationObjectives=history)


def work_units(rows, width, classes):
    """Deterministic projection/lattice work proxy, not measured wall time.

    Same data and dimensions make units comparable. Include the E pass and
    account for the control's more frequent forward-backward. The unmatched
    allowance is strictly less than one full control evaluation.
    """
    frames = sum(len(v["features"]) for r in rows for v in r["views"])
    lattice = sum(len(v["features"]) * (2 * len(v["target"]) + 1)
                  for r in rows for v in r["views"])
    forward = frames * width * classes
    backward = forward + 8 * frames * classes
    paths = 8 * lattice
    return dict(expectation=forward + paths, fixed=forward + backward,
                ordinary=forward + backward + paths)


def matched_refit(initial, rows, groups, validate, regularizations=(.001, .01, .1),
                  rounds=6, max_m_evaluations=2048, seconds=86400., emit=None):
    """Training-speaker-only shared-round, shared-regularization selection."""
    if rounds < 1 or max_m_evaluations < 2 or seconds <= 0 or not regularizations:
        raise ValueError("Invalid bounded fitting schedule")
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    if not fit or not selection or {r["speaker"] for r in fit} & {r["speaker"] for r in selection}:
        raise ValueError("Need disjoint fit and selection speakers")
    costs = work_units(fit, initial.weight.shape[1], len(initial.bias))
    deadline = time.monotonic() + seconds
    best_score, best_heads = math.inf, None
    history = []
    for reg in regularizations:
        if not math.isfinite(reg) or reg <= 0:
            raise ValueError("Invalid regularization candidate")
        heads = {arm: copy.deepcopy(initial) for arm in ("alternating", "ordinary")}
        total_allowance = total_control_evaluations = total_em = 0
        for phase in range(rounds + 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("One-day fitting budget exhausted")
            diagnostics = {}
            if phase:
                frozen = assignments(heads["alternating"], fit, deadline)
                current = heads["alternating"]
                fn = lambda z: corpus_value_gradient(current.with_vector(z), initial, fit, reg, frozen)
                vector, diagnostics["alternating"] = minimize_head(current.vector(), fn,
                    max_m_evaluations, deadline)
                if not diagnostics["alternating"]["converged"]:
                    raise RuntimeError("Fixed-assignment head fit did not converge: " +
                                       diagnostics["alternating"]["termination"])
                heads["alternating"] = current.with_vector(vector)
                total_em += costs["expectation"] + diagnostics["alternating"]["evaluations"] * costs["fixed"]
                # Carry the remainder between phases; never overspend one evaluation.
                allowance = total_em // costs["ordinary"] - total_allowance
                total_allowance += allowance
                current = heads["ordinary"]
                fn = lambda z: corpus_value_gradient(current.with_vector(z), initial, fit, reg)
                vector, diagnostics["ordinary"] = minimize_head(current.vector(), fn,
                    int(allowance), deadline)
                heads["ordinary"] = current.with_vector(vector)
                total_control_evaluations += diagnostics["ordinary"]["evaluations"]
            metrics = {arm: validate(head, selection, initial, reg) for arm, head in heads.items()}
            pair = np.mean([m["objective"] for m in metrics.values()])
            entry = dict(regularization=reg, phase=phase, metrics=metrics, diagnostics=diagnostics,
                alternatingWork=total_em, ordinaryAllowance=total_allowance,
                ordinaryEvaluations=total_control_evaluations,
                unusedRoundingWork=total_em - total_allowance * costs["ordinary"])
            history.append(entry)
            if emit:
                emit(entry)
            if pair < best_score:
                best_score, best_heads = float(pair), copy.deepcopy(heads)
                chosen = dict(regularization=reg, phase=phase, selectionObjective=best_score)
    return best_heads, dict(history=history, selected=chosen, workUnits=costs,
        selectionRule="minimum-pair-mean-objective-at-shared-phase-and-regularization-on-selection-speakers",
        computeRule="matched-projection-and-lattice-work-allowance; stop-control-if-stationary")

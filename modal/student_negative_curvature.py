"""Exact marginalized CTC curvature and matched frozen-phone-head refits.

Hessian-vector products differentiate path occupancies, not a fixed-path or
framewise Fisher surrogate. All objective calls cover every fit recording.
"""
import hashlib
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

GRADIENT_TOLERANCE = 1e-5


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
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.weight.size + self.bias.size,) or not np.isfinite(vector).all():
            raise ValueError("Invalid native head vector")
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


def _path_sum(values, tangents):
    """Log-sum and its tangent; unreachable states have zero tangent.

    Never differentiate -inf minus -inf. A finite sentinel would admit paths
    absent from the complete CTC graph, particularly on very peaky inputs.
    """
    z = logsumexp(values, axis=0)
    weights = np.zeros_like(values)
    reachable = np.isfinite(z)
    weights[:, reachable] = np.exp(values[:, reachable] - z[reachable])
    return z, (weights * tangents).sum(0)


def ctc_statistics(lp, target, tangent=None):
    """Complete transcript log mass, occupancies and directional derivatives.

    Alpha includes the current emission; beta excludes it. Their sensitivities
    retain the dependence of every path weight on every frame. Empty targets
    have exactly the all-blank path, and adjacent equal phones require blank.
    """
    lp, y = np.asarray(lp, dtype=np.float64), np.asarray(target)
    if (lp.ndim != 2 or not len(lp) or lp.shape[1] < 2 or not np.isfinite(lp).all() or
            y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or
            (y <= 0).any() or (y >= lp.shape[1]).any() or
            len(lp) < len(y) + int((y[1:] == y[:-1]).sum()) or
            not np.allclose(logsumexp(lp, axis=1), 0., atol=1e-6)):
        raise ValueError("Invalid or impossible complete CTC transcript")
    tangent = np.zeros_like(lp) if tangent is None else np.asarray(tangent, dtype=np.float64)
    if tangent.shape != lp.shape or not np.isfinite(tangent).all():
        raise ValueError("Invalid log-probability tangent")
    labels = np.zeros(2 * len(y) + 1, dtype=np.int64)
    labels[1::2] = y
    skip = np.zeros(len(labels), dtype=bool)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    emission, de = lp[:, labels], tangent[:, labels]
    alpha, beta = np.full_like(emission, -np.inf), np.full_like(emission, -np.inf)
    da, db = np.zeros_like(emission), np.zeros_like(emission)
    first = slice(0, min(2, len(labels)))
    alpha[0, first], da[0, first] = emission[0, first], de[0, first]
    for t in range(1, len(lp)):
        stay, ds = alpha[t - 1], da[t - 1]
        step, dstep = np.r_[-np.inf, stay[:-1]], np.r_[0., ds[:-1]]
        jump, djump = np.full(len(labels), -np.inf), np.zeros(len(labels))
        if len(labels) > 2:
            jump[2:] = np.where(skip[2:], stay[:-2], -np.inf)
            djump[2:] = np.where(skip[2:], ds[:-2], 0.)
        z, dz = _path_sum(np.stack([stay, step, jump]), np.stack([ds, dstep, djump]))
        alpha[t], da[t] = z + emission[t], dz + de[t]
    terminal = slice(max(0, len(labels) - 2), len(labels))
    z, dz = _path_sum(alpha[-1, terminal, None], da[-1, terminal, None])
    z, dz = float(z[0]), float(dz[0])
    if not math.isfinite(z):
        raise ArithmeticError("Complete transcript has no finite path mass")
    beta[-1, terminal] = 0.
    for t in range(len(lp) - 2, -1, -1):
        following, df = beta[t + 1] + emission[t + 1], db[t + 1] + de[t + 1]
        step, dstep = np.r_[following[1:], -np.inf], np.r_[df[1:], 0.]
        jump, djump = np.full(len(labels), -np.inf), np.zeros(len(labels))
        if len(labels) > 2:
            jump[:-2] = np.where(skip[2:], following[2:], -np.inf)
            djump[:-2] = np.where(skip[2:], df[2:], 0.)
        beta[t], db[t] = _path_sum(np.stack([following, step, jump]), np.stack([df, dstep, djump]))
    posterior = np.exp(alpha + beta - z)
    derivative = posterior * (da + db - dz)
    gamma, dgamma = np.zeros_like(lp), np.zeros_like(lp)
    for s, label in enumerate(labels):
        gamma[:, label] += posterior[:, s]
        dgamma[:, label] += derivative[:, s]
    if (not np.allclose(gamma.sum(1), 1., atol=2e-8, rtol=2e-8) or
            not np.allclose(dgamma.sum(1), 0., atol=2e-7, rtol=0)):
        raise ArithmeticError("CTC path sensitivity failed normalization")
    return z, gamma, dz, dgamma


def corpus_evaluate(head, initial, rows, regularization, direction=None, deadline=math.inf):
    """Exact loss, gradient and optional HVP, equally weighted by recording/view.

    CTC / transcript length + full teacher CE / real teacher frames + shipped
    KL / real student frames + L2 to the initial affine head. Pauses are not
    masked or mined into positives. Encoder features and teacher are constants.
    """
    if not rows or not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Need recordings and positive head regularization")
    vector = head.vector()
    if direction is not None:
        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape != vector.shape or not np.isfinite(direction).all():
            raise ValueError("Invalid head-space direction")
        vh = head.with_vector(direction)
    dw, db, hw, hb = (np.zeros_like(head.weight), np.zeros_like(head.bias),
                       np.zeros_like(head.weight), np.zeros_like(head.bias))
    value = 0.
    for row in rows:
        if not row["views"]:
            raise ValueError("Empty recording views")
        scale = 1. / (len(rows) * len(row["views"]))
        for v in row["views"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("One-day fitting budget exhausted")
            x, y = np.asarray(v["features"], dtype=np.float64), np.asarray(v["target"])
            if x.ndim != 2 or x.shape[1] != head.weight.shape[1] or not np.isfinite(x).all() or not len(y):
                raise ValueError("Need finite frozen features and a performed transcript")
            lp = log_probs(head, x)
            p = np.exp(lp)
            if direction is not None:
                dl = x @ vh.weight.T + vh.bias
                dl -= (p * dl).sum(1, keepdims=True)
            else:
                dl = None
            z, gamma, _, dgamma = ctc_statistics(lp, y, dl)
            teacher, shipped = np.asarray(v["teacher"]), np.asarray(v["shipped"])
            if (shipped.shape != lp.shape or teacher.ndim != 2 or
                    teacher.shape[1] != lp.shape[1] or not 0 < len(teacher) <= len(lp) or
                    not np.isfinite(teacher).all() or not np.isfinite(shipped).all()):
                raise ValueError("Preservation posterior clocks or inventory differ")
            tq = np.exp(teacher - logsumexp(teacher, axis=1, keepdims=True))
            sq = shipped - logsumexp(shipped, axis=1, keepdims=True)
            gradient = (p - gamma) / len(y) + (p - np.exp(sq)) / len(lp)
            gradient[:len(tq)] += (p[:len(tq)] - tq) / len(tq)
            loss = (-z / len(y) - (tq * lp[:len(tq)]).sum() / len(tq) +
                    (np.exp(sq) * (sq - lp)).sum() / len(lp))
            value += scale * loss
            dw += scale * (gradient.T @ x)
            db += scale * gradient.sum(0)
            if direction is not None:
                dp = p * dl
                hg = (dp - dgamma) / len(y) + dp / len(lp)
                hg[:len(tq)] += dp[:len(tq)] / len(tq)
                hw += scale * (hg.T @ x)
                hb += scale * hg.sum(0)
    delta = vector - initial.vector()
    value += .5 * regularization * float(delta @ delta)
    gradient = np.r_[dw.ravel(), db] + regularization * delta
    hv = None if direction is None else np.r_[hw.ravel(), hb] + regularization * direction
    if not math.isfinite(value) or not np.isfinite(gradient).all() or (hv is not None and not np.isfinite(hv).all()):
        raise ArithmeticError("Non-finite complete acoustic objective")
    return float(value), gradient, hv


class CorpusOracle:
    def __init__(self, template, initial, rows, regularization, deadline):
        self.template, self.initial, self.rows = template, initial, rows
        self.regularization, self.deadline = regularization, deadline
        self.evaluations, self.hvps = 0, 0

    def evaluate(self, vector):
        result = corpus_evaluate(self.template.with_vector(vector), self.initial,
                                 self.rows, self.regularization, deadline=self.deadline)
        self.evaluations += 1
        return result[:2]

    def hvp(self, vector, direction):
        result = corpus_evaluate(self.template.with_vector(vector), self.initial, self.rows,
                                 self.regularization, direction, self.deadline)
        self.evaluations += 1
        self.hvps += 1
        return result[2]


def smallest_curvature(hvp, dimension, seed, iterations=24):
    """Fully reorthogonalized Lanczos/Ritz search, with fixed HVP compute.

    The measured Rayleigh quotient certifies negative curvature even when the
    reported eigenvector residual has not converged. Controls run this same
    search for diagnostics but do not use its direction. Pad breakdowns with
    zero probes so each arm still performs iterations + 1 complete HVP calls.
    """
    if dimension < 1 or iterations < 1:
        raise ValueError("Invalid curvature-search dimensions")
    rng = np.random.default_rng(seed)
    q = rng.normal(size=dimension)
    q /= np.linalg.norm(q)
    basis, products = [], []
    active = True
    for _ in range(iterations):
        product = hvp(q if active else np.zeros(dimension))
        if not active:
            continue
        if product.shape != q.shape or not np.isfinite(product).all():
            raise ArithmeticError("Invalid Hessian-vector product")
        basis.append(q.copy())
        products.append(product.copy())
        matrix = np.stack(basis, axis=1)
        residual = product.copy()
        for _ in range(2):
            residual -= matrix @ (matrix.T @ residual)
        norm = np.linalg.norm(residual)
        active = len(basis) < dimension and norm > 1e-12 * max(1., np.linalg.norm(product))
        if active:
            q = residual / norm
    matrix, products = np.stack(basis, axis=1), np.stack(products, axis=1)
    projected = matrix.T @ products
    eigenvalues, eigenvectors = np.linalg.eigh((projected + projected.T) / 2)
    direction = matrix @ eigenvectors[:, 0]
    direction /= np.linalg.norm(direction)
    measured = hvp(direction)
    rayleigh = float(direction @ measured)
    return direction, dict(rayleigh=rayleigh, residualNorm=float(np.linalg.norm(measured - rayleigh * direction)),
        ritzMinimum=float(eigenvalues[0]), subspaceDimension=len(basis), hvps=iterations + 1,
        negative=rayleigh < -1e-8 * max(1., np.linalg.norm(measured)))


def trust_candidates(vector, direction, radius, evaluate):
    """Same four norms and both signs in every arm; accept actual loss decrease.

    Evaluate all eight candidates even when an early one improves. No predicted
    quadratic decrease, surrogate loss, or held-out metric authorizes a step.
    """
    norm = np.linalg.norm(direction)
    if radius <= 0 or not math.isfinite(radius) or not np.isfinite(norm):
        raise ValueError("Invalid trust region")
    direction = direction / norm if norm else np.zeros_like(vector)
    before, _ = evaluate(vector)
    if not math.isfinite(before):
        raise ArithmeticError("Non-finite current trust objective")
    candidates = []
    for fraction in (1., .5, .25, .125):
        for sign in (-1., 1.):
            delta = sign * radius * fraction * direction
            value, _ = evaluate(vector + delta)
            if not math.isfinite(value):
                raise ArithmeticError("Non-finite trust candidate")
            candidates.append((value, delta))
    value, delta = min(candidates, key=lambda item: item[0])
    accepted = value < before - 1e-12 * max(1., abs(before))
    return (vector + delta if accepted else vector.copy()), dict(before=before,
        after=value if accepted else before, accepted=bool(accepted),
        stepNorm=float(np.linalg.norm(delta)) if accepted else 0., radius=radius,
        candidateNorms=[radius * f for f in (1., .5, .25, .125)], evaluations=9)


def first_order_start(initial, oracle, evaluations=128, tolerance=GRADIENT_TOLERANCE):
    """Shared full-corpus L-BFGS warm start; never call budget exit convergence."""
    from scipy.optimize import minimize
    if evaluations < 1:
        raise ValueError("Invalid first-order budget")
    best, count = [None], [0]

    class BudgetReached(Exception):
        pass

    def closure(vector):
        if count[0] >= evaluations:
            raise BudgetReached()
        value, gradient = oracle.evaluate(vector)
        count[0] += 1
        if best[0] is None or value < best[0][0]:
            best[0] = (value, vector.copy(), gradient.copy())
        return value, gradient

    try:
        result = minimize(closure, initial.vector(), method="L-BFGS-B", jac=True,
            options=dict(gtol=tolerance, ftol=1e-15, maxiter=evaluations,
                         maxfun=evaluations, maxls=40, maxcor=20))
        reason = str(result.message)
    except BudgetReached:
        reason = "shared objective-evaluation budget"
    value, vector, gradient = best[0]
    norm = float(np.abs(gradient).max())
    return vector, dict(objective=value, evaluations=count[0], gradientInfinityNorm=norm,
        converged=norm <= tolerance, termination=reason)


def matched_refit(initial, rows, groups, validate, rounds=6, lanczos_iterations=24,
                  warm_evaluations=128, regularization=.01, seconds=86400., seed=1729, emit=None):
    """Shared training-speaker selection; fixed equal per-arm call budgets.

    The first-order arm uses its gradient; random uses a seeded unit direction;
    curvature uses a negative Ritz direction or its ordinary gradient when no
    negative direction is found. A common fixed radius schedule matches every
    attempted norm. All three retain the same complete preservation objective.
    """
    if (rounds < 1 or lanczos_iterations < 1 or not math.isfinite(seconds) or seconds <= 0 or seconds > 86400 or
            set(groups.values()) != {"fit", "selection"} or
            any(r["speaker"] not in groups for r in rows)):
        raise ValueError("Invalid training-only matched schedule")
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    if not fit or not selection:
        raise ValueError("Need disjoint fit and selection speakers")
    deadline = time.monotonic() + seconds
    common = CorpusOracle(initial, initial, fit, regularization, deadline)
    start, warm = first_order_start(initial, common, warm_evaluations)
    arms = ("curvature", "first-order", "random")
    vectors = {arm: start.copy() for arm in arms}
    oracles = {arm: CorpusOracle(initial, initial, fit, regularization, deadline) for arm in arms}
    base_radius = .01 * max(1., np.linalg.norm(initial.vector()))
    initial_metrics = {a: validate(initial, selection, initial, regularization) for a in arms}
    best_score = float(np.mean([m["objective"] for m in initial_metrics.values()]))
    if not math.isfinite(best_score):
        raise ArithmeticError("Invalid initial selection objective")
    chosen_heads = {a: initial.with_vector(initial.vector()) for a in arms}
    chosen = dict(phase=-1, selectionObjective=best_score)
    history = [dict(phase=-1, metrics=initial_metrics, diagnostics={})]
    for phase in range(rounds + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError("One-day fitting budget exhausted")
        diagnostics = {}
        if phase:
            radius = base_radius * .5 ** ((phase - 1) // 2)
            for arm in arms:
                oracle, vector = oracles[arm], vectors[arm]
                _, gradient = oracle.evaluate(vector)
                direction, curvature = smallest_curvature(lambda v: oracle.hvp(vector, v),
                    len(vector), seed + phase, lanczos_iterations)
                if arm == "random":
                    direction = np.random.default_rng(seed + 1000 + phase).normal(size=len(vector))
                    kind = "random"
                elif arm == "first-order" or not curvature["negative"]:
                    # Use the same stationarity test as the first-order solver;
                    # do not amplify floating-point gradient noise into a
                    # full-radius random escape at an already stationary head.
                    direction = (-gradient if np.abs(gradient).max() > GRADIENT_TOLERANCE
                                 else np.zeros_like(gradient))
                    kind = "first-order"
                else:
                    kind = "negative-curvature"
                vectors[arm], step = trust_candidates(vector, direction, radius, oracle.evaluate)
                diagnostics[arm] = dict(curvature=curvature, step=step, direction=kind,
                    gradientInfinityNorm=float(np.abs(gradient).max()),
                    evaluations=oracle.evaluations, hvps=oracle.hvps)
            if len({(o.evaluations, o.hvps) for o in oracles.values()}) != 1:
                raise ArithmeticError("Control and treatment objective compute differ")
        metrics = {a: validate(initial.with_vector(z), selection, initial, regularization)
                   for a, z in vectors.items()}
        if time.monotonic() >= deadline:
            raise TimeoutError("One-day fitting budget exhausted during selection")
        score = float(np.mean([m["objective"] for m in metrics.values()]))
        if not math.isfinite(score):
            raise ArithmeticError("Invalid selection objective")
        entry = dict(phase=phase, metrics=metrics, diagnostics=diagnostics)
        history.append(entry)
        if emit:
            emit(entry)
        if score < best_score:
            best_score = score
            chosen_heads = {a: initial.with_vector(z) for a, z in vectors.items()}
            chosen = dict(phase=phase, selectionObjective=score)
    return chosen_heads, dict(history=history, selected=chosen, warmStart=warm,
        selectionRule="minimum-three-arm-mean-objective-at-shared-phase-on-selection-speakers",
        computeRule="identical-full-corpus-value-gradient-and-HVP-call-counts-in-all-arms",
        radiusRule="one-percent-initial-head-norm-halved-every-two-rounds",
        regularization=regularization, seconds=seconds)

"""Frozen phone-head refit with joint or independent paired-view CTC paths.

Only transcribed training features enter this module. The joint lattice sums
products of separate view emissions; it never normalizes their product or
turns either view's predictions into the other view's targets.
"""
import copy
import hashlib
import time
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass
class PhoneHead:
    weight: np.ndarray
    bias: np.ndarray

    def vector(self):
        return np.r_[self.weight.ravel(), self.bias]

    def with_vector(self, vector):
        n = self.weight.size
        return PhoneHead(vector[:n].reshape(self.weight.shape).copy(), vector[n:].copy())


def phone_head(weight, bias):
    w, b = np.asarray(weight, dtype=np.float64), np.asarray(bias, dtype=np.float64)
    if w.ndim != 2 or b.shape != (len(w),) or not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError("Invalid affine phone head")
    return PhoneHead(w.copy(), b.copy())


def log_probs(head, features):
    logits = np.asarray(features) @ head.weight.T + head.bias
    return logits - logsumexp(logits, axis=-1, keepdims=True)


def forward_backward(emissions, target):
    """Log-domain CTC sum and class occupancies for unnormalized emissions.

    Joint emissions are sums of view log-probabilities, NOT log-softmaxes.
    Blank is zero; a repeated target cannot skip its intervening blank.
    Impossible transcripts are errors, rather than zero-loss examples.
    """
    e, y = np.asarray(emissions, dtype=np.float64), np.asarray(target)
    if (e.ndim != 2 or not len(e) or e.shape[1] < 2 or not np.isfinite(e).all() or
            y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or
            np.any(y <= 0) or np.any(y >= e.shape[1]) or
            len(e) < len(y) + int((y[1:] == y[:-1]).sum())):
        raise ValueError("Invalid or impossible CTC transcript")
    labels = np.zeros(2 * len(y) + 1, dtype=np.int64)
    labels[1::2] = y
    skip = np.zeros(len(labels), dtype=bool)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    emission = e[:, labels]
    alpha = np.full(emission.shape, -np.inf)
    alpha[0, :min(2, len(labels))] = emission[0, :min(2, len(labels))]
    for t in range(1, len(e)):
        previous = alpha[t - 1]
        step = np.r_[-np.inf, previous[:-1]]
        jump = np.full(len(labels), -np.inf)
        jump[2:] = np.where(skip[2:], previous[:-2], -np.inf)
        alpha[t] = np.logaddexp(np.logaddexp(previous, step), jump) + emission[t]
    terminal = slice(max(0, len(labels) - 2), len(labels))
    z = logsumexp(alpha[-1, terminal])
    if not np.isfinite(z):
        raise ValueError("No finite complete CTC path")
    beta = np.full(emission.shape, -np.inf)
    beta[-1, terminal] = 0.
    for t in range(len(e) - 2, -1, -1):
        following = beta[t + 1] + emission[t + 1]
        step = np.r_[following[1:], -np.inf]
        jump = np.full(len(labels), -np.inf)
        jump[:-2] = np.where(skip[2:], following[2:], -np.inf)
        beta[t] = np.logaddexp(np.logaddexp(following, step), jump)
    posterior = np.exp(alpha + beta - z)
    gamma = np.zeros_like(e)
    for state, label in enumerate(labels):
        gamma[:, label] += posterior[:, state]
    if not np.allclose(gamma.sum(1), 1., atol=2e-8, rtol=2e-8):
        raise ArithmeticError("CTC occupancies failed normalization")
    return float(z), gamma


def correspondence(times):
    """Union of exact original-sample timestamps; no frame rounding/interpolation.

    Each view contributes each observation exactly once. At coincident times
    both emit from the same state. At other times the unobserved view contributes
    log(1), not a fabricated blank or a repeated observation. The shared state
    advances on this union grid; independent paths advance on their own grids.
    """
    if len(times) != 2:
        raise ValueError("Need original and degraded observations")
    for t in times:
        if (np.asarray(t).ndim != 1 or not len(t) or
                not np.issubdtype(np.asarray(t).dtype, np.integer) or np.any(np.diff(t) <= 0)):
            raise ValueError("Need strictly increasing integer sample timestamps")
    if max(t[0] for t in times) > min(t[-1] for t in times):
        raise ValueError("The paired recordings have no corresponding interval")
    grid = np.unique(np.concatenate(times))
    return grid, [np.searchsorted(grid, t) for t in times]


def paired_paths(lps, target, times):
    """Compute BOTH exact sums in both arms for identical lattice work.

    This explicit extra computation matches the control's compute without
    approximating assignments, padding fake data, or refreshing one arm less.
    The unused sum contributes no loss or gradient.
    """
    grid, maps = correspondence(times)
    if (len(lps) != 2 or any(p.ndim != 2 or len(p) != len(t) for p, t in zip(lps, times)) or
            lps[0].shape[1] != lps[1].shape[1] or
            any(not np.allclose(logsumexp(p, axis=-1), 0., atol=1e-6) for p in lps)):
        raise ValueError("View emission shapes or probability normalization differ")
    joint = np.zeros((len(grid), lps[0].shape[1]), dtype=np.float64)
    for p, indices in zip(lps, maps):
        joint[indices] += p
    shared_z, shared_gamma = forward_backward(joint, target)
    independent = [forward_backward(p, target) for p in lps]
    return {
        "shared": (shared_z, [shared_gamma[i] for i in maps]),
        "independent": (sum(z for z, _ in independent), [g for _, g in independent]),
    }


def corpus_value_gradient(head, initial, rows, regularization, arm, deadline=np.inf):
    """Equal recordings, two equal views, transcript CTC and shipped-output KL.

    KL retains every native real frame, including breaths and pauses. Both arms
    have the same head, initialization, recording weights and preservation.
    """
    if arm not in ("shared", "independent") or not rows or not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Invalid refit arm, corpus or regularization")
    dw, db, value = np.zeros_like(head.weight), np.zeros_like(head.bias), 0.
    for row in rows:
        if time.monotonic() >= deadline:
            raise TimeoutError("One-day training allowance exhausted")
        views, y = row["views"], row["target"]
        if len(views) != 2 or not len(y):
            raise ValueError("Need two full views and a performed transcript")
        lps = [log_probs(head, v["features"]) for v in views]
        z, gammas = paired_paths(lps, y, [v["times"] for v in views])[arm]
        value -= z / (2 * len(y) * len(rows))
        for v, lp, gamma in zip(views, lps, gammas):
            shipped = v["shipped"]
            if shipped.shape != lp.shape or not np.isfinite(shipped).all():
                raise ValueError("Preservation frame clock or alphabet differs")
            sq = shipped - logsumexp(shipped, axis=-1, keepdims=True)
            p, q = np.exp(lp), np.exp(sq)
            value += float((q * (sq - lp)).sum()) / (2 * len(lp) * len(rows))
            gradient = ((p - gamma) / len(y) + (p - q) / len(lp)) / (2 * len(rows))
            dw += gradient.T @ v["features"]
            db += gradient.sum(0)
    delta = head.vector() - initial.vector()
    value += .5 * regularization * float(delta @ delta)
    gradient = np.r_[dw.ravel(), db] + regularization * delta
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ArithmeticError("Non-finite refit loss or gradient")
    return float(value), gradient


def speaker_partition(speakers):
    if any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Every recording needs an identified training speaker")
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(unique) < 5:
        raise ValueError("Need five training speakers for disjoint fit and selection")
    return {s: "selection" if i % 5 == 0 else "fit" for i, s in enumerate(unique)}


def admitted_none(labels, excluded=()):
    excluded = set(excluded) | {r.get("speaker") for r in labels if r.get("split") != "train"}
    selected, seen = [], set()
    for r in labels:
        if r.get("split") != "train" or r.get("kind") != "none" or not r.get("speaker") or r["speaker"] in excluded:
            continue
        if (r.get("scored") is not True or r.get("exclude") or r.get("excluded") or
                any(r.get(k) != [] for k in ("events", "non_errors", "unsupported"))):
            continue
        if not r.get("id") or not r.get("wav") or not r.get("ayahs") or r["id"] in seen:
            raise ValueError("Missing or duplicate none-take identity/transcript")
        seen.add(r["id"])
        selected.append(r)
    return selected


def matched_refit(initial, rows, groups, validate, regularizations=(.001, .01, .1),
                  updates=60, learning_rate=.01, seconds=86400., emit=None):
    """Identical full-corpus Adam updates and lattice work; joint round selection.

    Both sums are evaluated in both arms on every update. Checkpoint selection
    uses pair-mean PER on disjoint training speakers, never practice incidents.
    No early stopping of just one arm, line-search or unequal batch exposure.
    """
    if (updates < 1 or learning_rate <= 0 or not np.isfinite(learning_rate) or
            seconds <= 0 or not np.isfinite(seconds) or not regularizations):
        raise ValueError("Invalid bounded fitting schedule")
    if set(groups.values()) - {"fit", "selection"} or any(r["speaker"] not in groups for r in rows):
        raise ValueError("Invalid speaker partition")
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    if not fit or not selection:
        raise ValueError("Need disjoint fit and selection speakers")
    deadline, best = time.monotonic() + seconds, None
    history = []
    for reg in regularizations:
        if not np.isfinite(reg) or reg <= 0:
            raise ValueError("Invalid regularization candidate")
        heads = {a: copy.deepcopy(initial) for a in ("shared", "independent")}
        moments = {a: [np.zeros_like(initial.vector()), np.zeros_like(initial.vector())] for a in heads}
        for step in range(updates + 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("One-day training allowance exhausted")
            if step:
                for arm, head in heads.items():
                    _, g = corpus_value_gradient(head, initial, fit, reg, arm, deadline)
                    m, v = moments[arm]
                    m *= .9
                    m += .1 * g
                    v *= .999
                    v += .001 * g ** 2
                    delta = learning_rate * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
                    heads[arm] = head.with_vector(head.vector() - delta)
            if step % 10 == 0 or step == updates:
                metrics = {a: validate(h, selection) for a, h in heads.items()}
                score = float(np.mean([m["per"] for m in metrics.values()]))
                if not np.isfinite(score):
                    raise ArithmeticError("Non-finite training-speaker selection metric")
                if time.monotonic() >= deadline:
                    raise TimeoutError("One-day training allowance exhausted")
                entry = dict(regularization=reg, step=step, metrics=metrics)
                history.append(entry)
                if emit:
                    emit(entry)
                if best is None or score < best[0]:
                    best = (score, copy.deepcopy(heads), dict(regularization=reg, step=step))
    return best[1], dict(selected=best[2], selectionPer=best[0], history=history,
        updatesPerArm=updates * len(regularizations),
        computeRule="both lattice sums and identical projections per complete-corpus update in both arms",
        selectionRule="minimum pair-mean PER on training selection speakers at a shared round and regularization")

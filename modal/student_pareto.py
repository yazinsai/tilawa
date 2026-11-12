"""Transcript-conditioned CTC contributions and nearest common-descent SGD.

For group k, gamma_k is its forward/backward frame occupancy. Its complete
logit contribution is mass(gamma_k) * softmax(logits) - gamma_k: competing
classes remain in every contribution. Summing the three gives p - gamma,
exactly the ordinary CTC gradient. Occupancies are recomputed at every update;
they are responsibilities, not an independently differentiated target loss.
"""
import hashlib
import itertools
import random

import numpy as np

# The core's vowel/madd/mark inventory; tajweed consonant variants stay identities.
VOWEL_MARKS = frozenset("َُِ۪ۦۥاـؙۜ")


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


def log_softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Need finite nonempty frame logits")
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def ctc_occupancies(log_probs, target):
    """Exact CTC forward sum and class occupancies, including repeated labels.

    Callers pass only valid frames: encoder drain padding is never supervised.
    Empty transcripts are supported and constrain genuine silence to blank.
    Impossible alignments fail rather than silently contributing zero gradient.
    """
    lp, y = np.asarray(log_probs, dtype=np.float64), np.asarray(target)
    if (lp.ndim != 2 or not len(lp) or lp.shape[1] < 2 or not np.isfinite(lp).all() or
            not np.allclose(np.exp(lp).sum(1), 1., atol=1e-8, rtol=1e-8) or
            y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or
            np.any(y <= 0) or np.any(y >= lp.shape[1])):
        raise ValueError("Invalid normalized probabilities or transcript")
    if len(lp) < len(y) + int(np.sum(y[1:] == y[:-1])):
        raise ValueError("Impossible complete transcript alignment")
    states = np.zeros(2 * len(y) + 1, dtype=np.int64)
    states[1::2] = y
    skip = np.zeros(len(states), dtype=bool)
    skip[2:] = (states[2:] != 0) & (states[2:] != states[:-2])
    emissions = lp[:, states]
    alpha = np.full(emissions.shape, -np.inf)
    alpha[0, :min(2, len(states))] = emissions[0, :min(2, len(states))]
    for t in range(1, len(lp)):
        prev = alpha[t - 1]
        one = np.concatenate(([-np.inf], prev[:-1]))
        two = np.full(len(states), -np.inf)
        two[2:] = prev[:-2]
        alpha[t] = emissions[t] + np.logaddexp(np.logaddexp(prev, one), np.where(skip, two, -np.inf))
    z = np.logaddexp(alpha[-1, -1], alpha[-1, -2]) if len(y) else alpha[-1, 0]
    if not np.isfinite(z):
        raise ValueError("No finite CTC alignment")
    beta = np.full(emissions.shape, -np.inf)
    beta[-1, -min(2, len(states)):] = 0.
    for t in range(len(lp) - 2, -1, -1):
        nxt = emissions[t + 1] + beta[t + 1]
        one = np.concatenate((nxt[1:], [-np.inf]))
        two = np.full(len(states), -np.inf)
        two[:-2] = np.where(skip[2:], nxt[2:], -np.inf)
        beta[t] = np.logaddexp(np.logaddexp(nxt, one), two)
    occupancy = np.zeros_like(lp)
    posterior = np.exp(alpha + beta - z)
    for s, label in enumerate(states):
        occupancy[:, label] += posterior[:, s]
    if not np.allclose(occupancy.sum(1), 1., atol=1e-8, rtol=1e-8):
        raise ValueError("Forward/backward occupancies lost probability mass")
    return float(-z), occupancy


def group_ids(tokens):
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("Need unique ordered nonblank tokens")
    return np.asarray([2] + [1 if t in VOWEL_MARKS else 0 for t in tokens])


def ctc_contributions(features, weight, bias, target, groups):
    """Complete weight AND bias gradients, normalized by valid output frames."""
    x, w, b = (np.asarray(v, dtype=np.float64) for v in (features, weight, bias))
    groups = np.asarray(groups)
    if (x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1] or
            b.shape != (len(w),) or groups.shape != b.shape or
            np.any((groups < 0) | (groups > 2)) or groups[0] != 2 or
            np.any(groups[1:] == 2) or any(not np.isfinite(v).all() for v in (x, w, b))):
        raise ValueError("Invalid existing affine head or disjoint phone groups")
    lp = log_softmax(x @ w.T + b)
    loss, gamma = ctc_occupancies(lp, target)
    p = np.exp(lp)
    augmented = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    contributions = []
    for k in range(3):
        responsibility = gamma * (groups == k)[None, :]
        delta = responsibility.sum(1, keepdims=True) * p - responsibility
        contributions.append(delta.T @ augmented / len(x))
    ordinary = (p - gamma).T @ augmented / len(x)
    if not np.allclose(np.sum(contributions, axis=0), ordinary, atol=1e-10, rtol=1e-8):
        raise ValueError("Contributions do not reconstruct the original CTC gradient")
    return loss / len(x), np.stack(contributions)


def common_descent(contributions):
    """Euclidean projection of their sum onto {d: <d,g_k> >= 0}.

    Enumerate all active sets (only three constraints), solve the tiny KKT
    system, and retain the closest feasible stationary candidate. Normalize
    constraint normals for conditioning; this leaves every half-space intact.
    No learning-rate rescaling, example weighting or magnitude clipping occurs.
    """
    gradients = np.asarray(contributions, dtype=np.float64)
    if gradients.ndim < 2 or gradients.shape[0] != 3 or not np.isfinite(gradients).all():
        raise ValueError("Need three finite same-shaped contributions")
    shape = gradients.shape[1:]
    flat = gradients.reshape(3, -1)
    total = flat.sum(0)
    norms = np.linalg.norm(flat, axis=1)
    normals = flat[norms > 0] / norms[norms > 0, None]
    scale = max(np.linalg.norm(total), np.max(norms), np.finfo(float).tiny)
    tol = 1e-10 * scale
    if np.all(normals @ total >= 0):
        return total.reshape(shape)
    best, distance = None, np.inf
    for n in range(1, len(normals) + 1):
        for indices in itertools.combinations(range(len(normals)), n):
            active = normals[list(indices)]
            gram = active @ active.T
            rhs = -active @ total
            multipliers = np.linalg.lstsq(gram, rhs, rcond=1e-12)[0]
            if np.max(np.abs(gram @ multipliers - rhs)) > tol or np.any(multipliers < -tol):
                continue
            candidate = total + multipliers @ active
            if np.any(normals @ candidate < -tol):
                continue
            change = float(np.sum((candidate - total) ** 2))
            if change < distance:
                best, distance = candidate, change
    if best is None:
        raise ValueError("Could not solve common-descent projection")
    return best.reshape(shape)


def matched_refit(initial, rows, groups, tokens, validate, epochs, lr, seed, emit=None):
    """Same initial head, recordings, order, objective and shared selected epoch."""
    if epochs < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid bounded training schedule")
    fit = [r for r in rows if groups[r['speaker']] == 'fit']
    selection = [r for r in rows if groups[r['speaker']] == 'selection']
    if not fit or not selection:
        raise ValueError("Need disjoint training-speaker fit and selection partitions")
    heads = {arm: tuple(np.asarray(v, dtype=np.float64).copy() for v in initial)
             for arm in ('ordinary', 'pareto')}
    ids = group_ids(tokens)
    history = {arm: [dict(epoch=0, **validate(head, selection))] for arm, head in heads.items()}
    best = {arm: tuple(v.copy() for v in head) for arm, head in heads.items()}
    score = np.mean([v[0]['objective'] for v in history.values()])
    selected_epoch, conflicts, updates = 0, 0, 0
    group_conflicts, max_violation, squared_projection_change = np.zeros(3, dtype=int), 0., 0.
    for epoch in range(1, epochs + 1):
        order = list(range(len(fit)))
        random.Random(seed + epoch).shuffle(order)
        for arm, (weight, bias) in heads.items():
            for i in order:
                pieces = [ctc_contributions(v['features'], weight, bias, v['target'], ids)[1]
                          for v in fit[i]['views']]
                contributions = np.mean(pieces, axis=0)
                gradient = contributions.sum(0)
                if arm == 'pareto':
                    flat = contributions.reshape(3, -1)
                    against = flat @ gradient.ravel() < 0
                    conflicts += int(np.any(against))
                    group_conflicts += against
                    projected = common_descent(contributions)
                    squared_projection_change += float(np.sum((projected - gradient) ** 2))
                    max_violation = max(max_violation, float(max(0., -(flat @ projected.ravel()).min())))
                    gradient = projected
                    updates += 1
                weight -= lr * gradient[:, :-1]
                bias -= lr * gradient[:, -1]
                if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                    raise ValueError("Nonfinite head update")
            result = dict(epoch=epoch, **validate((weight, bias), selection))
            history[arm].append(result)
            if emit:
                emit(dict(arm=arm, **result))
        pair_score = np.mean([v[-1]['objective'] for v in history.values()])
        if pair_score < score:
            score, selected_epoch = float(pair_score), epoch
            best = {arm: tuple(v.copy() for v in head) for arm, head in heads.items()}
    return best, dict(history=history, selectedEpoch=selected_epoch,
        selectedUpdates=selected_epoch * len(fit), trainingUpdates=epochs * len(fit),
        paretoConflictedUpdates=conflicts, paretoUpdates=updates,
        paretoGroupConflicts=dict(zip(('consonant', 'vowel', 'blank'), group_conflicts.tolist())),
        maxCommonDescentViolation=max_violation,
        squaredProjectionChange=squared_projection_change,
        selectionObjective=float(score),
        selectionRule='minimum-pair-mean-CTC-at-shared-epoch-on-selection-speakers')

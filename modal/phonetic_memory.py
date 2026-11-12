"""Transcript-conditioned occurrence pooling and class-balanced feature regularization.

Alignment is deliberately detached: the acoustic features, rather than the CTC
alignment probabilities, must acquire variance and lose redundant directions.
Each target position has its own CTC state, including consecutive equal labels.
Explicit occurrence IDs can merge duration-expanded positions; phone identity
alone never establishes that two articulations are the same occurrence.
"""

import numpy as np


def occurrence_occupancy(log_probs, target, blank=0):
    """Return P(occurrence state at frame | entire transcript), or None if impossible.

    Runs detached forward/backward in float64 on CPU. Blank states are omitted
    from the returned (frames, transcript positions) matrix. Callers must slice
    to the recording's real output length before calling this function.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    y = np.asarray(target, dtype=np.int64)
    if lp.ndim != 2 or y.ndim != 1 or np.any(y == blank):
        raise ValueError("Expected frame log-probabilities and nonblank transcript")
    if np.any(y < 0) or np.any(y >= lp.shape[1]):
        raise ValueError("Transcript label outside inventory")
    T, L = len(lp), len(y)
    if not L:
        return np.zeros((T, 0), dtype=np.float32)
    if T < L + np.count_nonzero(y[1:] == y[:-1]):
        return None
    states = np.full(2 * L + 1, blank, dtype=np.int64)
    states[1::2] = y
    emission = lp[:, states]
    skip = np.zeros(len(states), dtype=bool)
    skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    alpha = np.full_like(emission, -np.inf)
    alpha[0, :2] = emission[0, :2]
    for t in range(1, T):
        prev = alpha[t - 1]
        one = np.r_[-np.inf, prev[:-1]]
        two = np.where(skip, np.r_[-np.inf, -np.inf, prev[:-2]], -np.inf)
        alpha[t] = np.logaddexp(np.logaddexp(prev, one), two) + emission[t]
    log_z = np.logaddexp(alpha[-1, -1], alpha[-1, -2])
    if not np.isfinite(log_z):
        return None
    # beta excludes the current frame's emission (alpha already includes it).
    beta = np.full_like(emission, -np.inf)
    beta[-1, -2:] = 0
    for t in range(T - 2, -1, -1):
        following = beta[t + 1] + emission[t + 1]
        one = np.r_[following[1:], -np.inf]
        two = np.r_[np.where(skip[2:], following[2:], -np.inf), -np.inf, -np.inf]
        beta[t] = np.logaddexp(np.logaddexp(following, one), two)
    return np.exp(alpha[:, 1::2] + beta[:, 1::2] - log_z).astype(np.float32)


def pool_occurrences(features, occupancy, target, occurrence_ids=None):
    """Pool one embedding per genuine occurrence, merging only explicit duration IDs."""
    import torch

    y = list(map(int, target))
    groups = list(range(len(y))) if occurrence_ids is None else list(occurrence_ids)
    if len(groups) != len(y) or tuple(occupancy.shape) != (len(features), len(y)):
        raise ValueError("Occurrence metadata does not match transcript or real frames")
    if any(p <= 0 for p in y):
        raise ValueError("Blank is not a phone occurrence")
    labels, group_indices = [], []
    unique = list(dict.fromkeys(groups))
    for group in unique:
        indices = [i for i, g in enumerate(groups) if g == group]
        if len({y[i] for i in indices}) != 1:
            raise ValueError("A duration group must represent one phone identity")
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("A duration group must be contiguous")
        labels.append(y[indices[0]])
    if not labels:
        return features.new_empty((0, features.shape[-1])), torch.empty(0, dtype=torch.long, device=features.device)
    lookup = {group: i for i, group in enumerate(unique)}
    group_indices = [lookup[group] for group in groups]
    raw = torch.as_tensor(occupancy, device=features.device, dtype=torch.float32).detach()
    weights = raw.new_zeros((len(features), len(unique)))
    weights.index_add_(1, torch.tensor(group_indices, device=features.device), raw)
    mass = weights.sum(0)
    valid = mass > 1e-6
    pooled = weights[:, valid].T @ features.float() / mass[valid, None]
    return pooled, torch.tensor(labels, dtype=torch.long, device=features.device)[valid]


def balanced_indices(labels, limit, seed):
    """Uniform occurrence sampling within each phone, shared by both training arms."""
    y = np.asarray(labels, dtype=np.int64)
    if np.any(y <= 0) or limit < 1:
        raise ValueError("Sampling requires nonblank labels and a positive class limit")
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(np.flatnonzero(y == c))[:limit] for c in np.unique(y)]) if len(y) else np.array([], dtype=np.int64)


def covariance_losses(embeddings, labels, scale, floor=1.0):
    """Equal total weight per represented class, independent of duration/frequency.

    Scale is fixed from a separate training-speaker calibration sample. Both
    arms compute these terms; only the covariance multiplier differs.
    """
    import torch

    if len(embeddings) != len(labels) or embeddings.ndim != 2 or floor <= 0:
        raise ValueError("Invalid occurrence embeddings or variance floor")
    if torch.any(labels <= 0) or torch.any(scale <= 0):
        raise ValueError("Blank classes and nonpositive calibration scales are invalid")
    if len(embeddings) < 2:
        zero = embeddings.sum() * 0
        return zero, zero
    _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    w = counts[inverse].float().reciprocal() / len(counts)
    x = embeddings.float() / scale.detach()
    centered = x - (w[:, None] * x).sum(0)
    covariance = centered.T @ (centered * w[:, None])
    std = torch.sqrt(covariance.diagonal() + 1e-4)
    variance = torch.relu(floor - std).mean()
    off = covariance - torch.diag_embed(covariance.diagonal())
    decorrelation = off.square().sum() / embeddings.shape[1]
    return variance, decorrelation

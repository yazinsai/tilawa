"""A transcript-conditioned observation channel for frozen-head distillation.

Rows are latent (student) phones, columns are observed teacher phones. Blank is
an ordinary channel row, but known silent frames are constrained to latent blank.
No teacher argmax alignment is used: EM sums over all legal transcript CTC paths.
This module has no Modal dependencies, and never loads scoring data.
"""

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class Observation:
    speaker: str
    target: tuple[int, ...]
    teacher: np.ndarray  # (time, vocabulary), normalized probabilities
    pause: np.ndarray  # (time,), independently labelled silence, not teacher blank


def speaker_partition(speaker: str, folds: int = 5) -> int:
    """Stable groups; all recordings of a speaker must stay in the same fold."""
    if not speaker or folds < 2:
        raise ValueError("a speaker and at least two partitions are required")
    return int.from_bytes(hashlib.sha256(speaker.encode()).digest()[:8], "big") % folds


def _probabilities(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or not len(p) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("expected finite, nonnegative probability rows")
    mass = p.sum(-1, keepdims=True)
    if (mass <= 0).any():
        raise ValueError("every probability row needs positive mass")
    return p / mass


def ctc_occupancy(emissions, target, pause=None):
    """Return log forward sum and phone occupancy over every legal CTC path.

    ``emissions[t, p]`` is the log likelihood of frame t given latent phone p.
    Repeated transcript phones require an intervening blank. Known pauses may
    occupy any blank state, but cannot consume transcript phones.
    """
    e = np.asarray(emissions, dtype=np.float64)
    if e.ndim != 2 or not len(e) or np.isnan(e).any():
        raise ValueError("invalid log emissions")
    target = tuple(int(p) for p in target)
    if any(p <= 0 or p >= e.shape[1] for p in target):
        raise ValueError("transcript must contain in-vocabulary nonblank phones")
    labels = np.zeros(2 * len(target) + 1, dtype=np.int64)
    labels[1::2] = target
    x = e[:, labels].copy()
    if pause is not None:
        pause = np.asarray(pause, dtype=bool)
        if pause.shape != (len(e),):
            raise ValueError("pause mask must match valid frames")
        x[np.ix_(pause, labels != 0)] = -np.inf
    T, S = x.shape
    skip = (labels != 0) & (labels != np.roll(labels, 2))
    skip[:2] = False
    alpha = np.full((T, S), -np.inf)
    alpha[0, :min(2, S)] = x[0, :min(2, S)]
    for t in range(1, T):
        prev = alpha[t - 1]
        one = np.r_[-np.inf, prev[:-1]]
        two = np.r_[[-np.inf] * min(2, S), prev[:max(0, S - 2)]]
        alpha[t] = x[t] + np.logaddexp(np.logaddexp(prev, one), np.where(skip, two, -np.inf))
    log_z = float(np.logaddexp.reduce(alpha[-1, max(0, S - 2):]))
    if not np.isfinite(log_z):
        raise ValueError("transcript has no legal path through the valid, nonpause frames")
    beta = np.full((T, S), -np.inf)
    beta[-1, max(0, S - 2):] = 0
    for t in range(T - 2, -1, -1):
        nxt = x[t + 1] + beta[t + 1]
        one = np.r_[nxt[1:], -np.inf]
        two = np.r_[nxt[min(2, S):], [-np.inf] * min(2, S)]
        can_skip = np.r_[skip[min(2, S):], [False] * min(2, S)]
        beta[t] = np.logaddexp(np.logaddexp(nxt, one), np.where(can_skip, two, -np.inf))
    state_p = np.exp(alpha + beta - log_z)
    occupancy = np.zeros_like(e)
    for s, phone in enumerate(labels):
        occupancy[:, phone] += state_p[:, s]
    return log_z, occupancy


def channel_posterior(channel, item, normalize=True):
    """Fractional teacher observations, with transcript placement marginalized."""
    channel = _probabilities(channel)
    q = _probabilities(item.teacher)
    if channel.shape != (q.shape[1], q.shape[1]):
        raise ValueError("channel must be square in the unchanged phone inventory")
    # A teacher distribution is a fractional categorical observation. This is
    # its expected log likelihood under each latent phone's observation row.
    emissions = q @ np.log(channel.clip(min=1e-12)).T
    log_z, posterior = ctc_occupancy(emissions, item.target, item.pause)
    # Normalize out the number of legal alignments when comparing recordings.
    path_count = ctc_occupancy(np.zeros_like(emissions), item.target, item.pause)[0] if normalize else 0
    return log_z - path_count, posterior


def fit_channel(items, diagonal_frames: float, iterations: int = 12):
    """EM with an identity Dirichlet prior measured in observation frames.

    Regularization is selected on disjoint training speakers, never incidents.
    Unobserved rows remain close to identity instead of inventing substitutions.
    """
    items = tuple(items)
    if not items or diagonal_frames <= 0 or iterations < 1:
        raise ValueError("need observations and positive regularization/iterations")
    V = items[0].teacher.shape[1]
    prior = diagonal_frames * np.eye(V) + 1e-6
    channel = _probabilities(np.eye(V) + np.full((V, V), 0.01 / V))
    for _ in range(iterations):
        counts = prior.copy()
        for item in items:
            _, latent = channel_posterior(channel, item, normalize=False)
            counts += latent.T @ _probabilities(item.teacher)
        channel = _probabilities(counts)
    channel.flags.writeable = False
    return channel


def observation_nll(channel, items):
    """Per valid frame likelihood on unseen training speakers for selection."""
    items = tuple(items)
    if not items:
        raise ValueError("validation observations are required")
    speakers = {}
    for x in items:
        loss, frames = speakers.get(x.speaker, (0., 0))
        speakers[x.speaker] = (loss - channel_posterior(channel, x)[0], frames + len(x.teacher))
    return float(np.mean([loss / frames for loss, frames in speakers.values()]))


def corrected_distillation(log_probs, teacher, valid, channel):
    """Forward correction: compare teacher with student probabilities @ channel.

    The channel is fixed, including when supplied as a tensor with gradients.
    Identity gives exactly ordinary full-distribution distillation.
    """
    import torch

    c = torch.as_tensor(channel, device=log_probs.device, dtype=log_probs.dtype).detach()
    if c.shape != (log_probs.shape[-1], log_probs.shape[-1]):
        raise ValueError("channel inventory differs from student")
    if not torch.isfinite(c).all() or (c < 0).any() or not torch.allclose(c.sum(-1), torch.ones_like(c[:, 0])):
        raise ValueError("channel rows must be normalized")
    q = teacher.detach().to(log_probs)
    if q.shape != log_probs.shape or valid.shape != log_probs.shape[:-1]:
        raise ValueError("teacher and valid mask must match student frames")
    if not torch.isfinite(q).all() or (q < 0).any() or (q.sum(-1)[valid] <= 0).any():
        raise ValueError("invalid teacher probability rows")
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
    # Log space also preserves gradients when student mass is tiny.
    observed = torch.logsumexp(log_probs.unsqueeze(-1) + c.log(), dim=-2)
    ce = -(q * observed).sum(-1)
    return ce[valid].sum() / valid.sum().clamp_min(1)


def head_loss(log_probs, teacher, targets, lengths, target_lengths, pause, channel,
              ctc_weight=1.0, kd_weight=1.0, pause_weight=1.0, preservation=None):
    """Shared objective for both matched arms; only their channel differs.

    Valid frames exclude padding; real-pause CE supervises latent blank directly,
    without sending silence through a possibly lossy teacher channel.
    """
    import torch
    import torch.nn.functional as F

    valid = torch.arange(log_probs.shape[1], device=log_probs.device)[None] < lengths[:, None]
    frames = valid.sum().clamp_min(1)
    ctc = F.ctc_loss(log_probs.transpose(0, 1), targets, lengths, target_lengths,
                     blank=0, reduction="sum", zero_infinity=False) / frames
    kd = corrected_distillation(log_probs, teacher, valid, channel)
    silent = pause & valid
    pause_ce = -log_probs[..., 0][silent].sum() / silent.sum().clamp_min(1)
    loss = ctc_weight * ctc + kd_weight * kd + pause_weight * pause_ce
    if preservation is not None:
        loss = loss + corrected_distillation(log_probs, preservation, valid,
                                             torch.eye(log_probs.shape[-1], device=log_probs.device))
    return loss, {"ctc": ctc, "kd": kd, "pause": pause_ce}

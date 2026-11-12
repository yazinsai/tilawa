"""Transcript-occurrence emission erasure for a frozen CTC phone head.

Only emission scores change in the auxiliary likelihood. The waveform, encoder
features, transcript and time axis never change. Selection is nondifferentiable;
gradients flow through the surviving acoustic emissions.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

CONSONANTS = frozenset("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي")
ARMS = ("ordinary", "random", "strongest")


def viterbi_states(log_probs, target):
    """Maximum legal CTC path, with distinct states for repeated occurrences.

    Return None iff the transcript has zero likelihood, including support zeros.
    Frame labels alone cannot distinguish two occurrences of the same phone.
    """
    lp = np.asarray(log_probs)
    y = np.asarray(target, dtype=np.int64)
    if lp.ndim != 2 or len(lp) == 0 or lp.shape[1] < 2 or y.ndim != 1:
        raise ValueError("Invalid CTC geometry")
    if np.isnan(lp).any() or np.isposinf(lp).any() or np.any(y <= 0) or np.any(y >= lp.shape[1]):
        raise ValueError("Invalid CTC emissions or transcript")
    labels = np.zeros(2 * len(y) + 1, dtype=np.int64)
    labels[1::2] = y
    skip = np.zeros(len(labels), dtype=bool)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    prev = np.full(len(labels), -np.inf, dtype=np.float64)
    prev[:min(2, len(labels))] = lp[0, labels[:min(2, len(labels))]]
    trace = np.zeros((len(lp), len(labels)), dtype=np.uint8)
    for t in range(1, len(lp)):
        one = np.concatenate(([-np.inf], prev[:-1]))
        two = np.full_like(prev, -np.inf)
        two[2:] = prev[:-2]
        two[~skip] = -np.inf
        choices = np.stack((prev, one, two))
        trace[t] = choices.argmax(0)
        prev = choices.max(0) + lp[t, labels]
    ends = [len(labels)-1] if len(y) == 0 else [len(labels)-2, len(labels)-1]
    state = max(ends, key=lambda s: prev[s])
    if not np.isfinite(prev[state]):
        return None
    path = np.empty(len(lp), dtype=np.int64)
    for t in range(len(lp)-1, -1, -1):
        path[t] = state
        state -= int(trace[t, state])
    return path


def neutralize(log_probs, frame):
    """One uniform log-probability row; no blank preference or renormalization.

    torch.where disconnects just this row from the acoustic model. A uniform
    row still occupies its original timestep, so this is emission erasure rather
    than deleting a frame or masking acoustic inputs.
    """
    if log_probs.ndim != 2 or not 0 <= frame < len(log_probs):
        raise ValueError("Erasure frame is outside the recording")
    mask = torch.arange(len(log_probs), device=log_probs.device) == frame
    return torch.where(mask[:, None], torch.full_like(log_probs, -math.log(log_probs.shape[1])), log_probs)


def select_exclusion(log_probs, target, tokens, rng, arm):
    """Sample a consonant occurrence, erase its peak or a random frame.

    The peak is the most probable emission among frames assigned to that
    occurrence by the full transcript's Viterbi alignment. All arms run the same
    alignment and feasibility work; the ordinary arm uses the original rows in
    its second CTC evaluation. No greedy hypothesis supplies training labels.
    """
    if arm not in ARMS or len(tokens) + 1 != log_probs.shape[1]:
        raise ValueError("Unknown arm or phone inventory")
    lp = log_probs.detach().cpu().numpy()
    y = target.detach().cpu().numpy()
    path = viterbi_states(lp, y)
    if path is None:
        raise ValueError("Authorized transcript is mathematically infeasible")
    occurrences = [j for j, phone in enumerate(y) if tokens[phone-1] in CONSONANTS]
    if not occurrences:
        # Empty/no-consonant targets still receive ordinary speech/pause loss.
        return log_probs, dict(occurrence=None, frame=None, skipped="no consonant")
    occurrence = occurrences[int(rng.integers(len(occurrences)))]
    assigned = np.flatnonzero(path == 2 * occurrence + 1)
    frame = int(assigned[np.argmax(lp[assigned, y[occurrence]])])
    random_frame = int(rng.integers(len(lp)))
    if arm == "random":
        frame = random_frame
    candidate = neutralize(log_probs, frame)
    feasible = viterbi_states(candidate.detach().cpu().numpy(), y) is not None
    report = dict(occurrence=occurrence, frame=frame, skipped=None if feasible else "zero likelihood")
    return (candidate if feasible and arm != "ordinary" else log_probs), report


def sequence_nll(log_probs, target):
    """Complete CTC forward sum, normalized by real emitted frames."""
    return F.ctc_loss(log_probs[:, None], target,
                      torch.tensor([len(log_probs)]), torch.tensor([len(target)]),
                      blank=0, reduction="sum", zero_infinity=False) / len(log_probs)


def head_objective(log_probs, target, baseline, tokens, rng, arm, anchor=2.):
    auxiliary, report = select_exclusion(log_probs, target, tokens, rng, arm)
    ordinary = sequence_nll(log_probs, target)
    robust = sequence_nll(auxiliary, target)
    preservation = F.kl_div(log_probs, baseline.exp(), reduction="sum") / len(log_probs)
    # Two complete sequence evaluations in every arm, including ordinary.
    # Full-recording preservation retains real pauses without fabricated labels.
    return ordinary + robust + anchor * preservation, report

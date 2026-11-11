"""Training-only observation censoring for deliberately applied time masks.

The mask is provenance, never inferred from zero features, silence, padding,
teacher disagreement or an expected word. At inference every frame is observed.
"""
import math


def spec_augment(feats, rng, n_freq=2, max_f=27, n_time=10, max_t_frac=0.05, *, return_mask=False):
    """The shipped draws, with optional union of time-mask intervals.

    Frequency masks do not mark whole observations hidden. Keep the original
    draw order even for zero-width masks so ordinary training is reproducible.
    """
    import torch

    feats = feats.clone()
    hidden = torch.zeros(feats.shape[0], dtype=torch.bool, device=feats.device)
    T = feats.shape[0]
    for _ in range(n_freq):
        f = int(rng.integers(0, max_f + 1))
        f0 = int(rng.integers(0, 80 - f + 1))
        feats[:, f0:f0 + f] = 0
    for _ in range(int(rng.integers(0, n_time + 1))):
        t = int(rng.integers(0, max(1, int(T * max_t_frac)) + 1))
        t0 = int(rng.integers(0, max(1, T - t)))
        feats[t0:t0 + t] = 0
        hidden[t0:t0 + t] = True
    return (feats, hidden) if return_mask else feats


def output_masks(time_mask, n_outputs):
    """Map provenance to the student's native 40 ms observation cells.

    Output j is centred near input fbank frame 4j+4. Censor only a complete
    central four-frame cell, [4j+2, 4j+6), covered by an explicit time mask.
    Partially covered cells retain CTC. Exclude imitation in *any* intersected
    cell. Neither mapping includes end padding or extends a mask into real
    silence. Encoder context remains frozen and can still carry information.
    """
    import torch

    if time_mask.ndim != 1 or time_mask.dtype != torch.bool or n_outputs < 0:
        raise ValueError('Expected one-dimensional boolean time-mask provenance')
    hidden = torch.zeros(n_outputs, dtype=torch.bool, device=time_mask.device)
    touched = hidden.clone()
    for j in range(n_outputs):
        lo, hi = 4*j + 2, 4*j + 6
        if hi <= len(time_mask):
            hidden[j] = time_mask[lo:hi].all()
            touched[j] = time_mask[lo:hi].any()
    return hidden, touched


def censored_ctc_loss(log_probs, target, hidden):
    """Sum all ordinary CTC progress paths through hidden observations.

    A hidden emission has a normalized uniform identity prior, independent of
    the fitted head. CTC can stay, advance or skip a separating blank under its
    usual repeated-label rules; it is not forced to emit blank or reconstruct
    the removed phones. The constant prior gives zero gradient there and avoids
    multiplying mass by the vocabulary size at every censored frame. Unmasked
    rows retain their full probabilities and complete-transcript requirement.
    """
    import torch
    import torch.nn.functional as F

    if (log_probs.ndim != 2 or hidden.shape != log_probs.shape[:1] or
            hidden.dtype != torch.bool or target.ndim != 1 or len(log_probs) == 0):
        raise ValueError('Invalid complete-sequence censoring geometry')
    if torch.any(target <= 0) or torch.any(target >= log_probs.shape[1]):
        raise ValueError('Transcript contains blank or an unknown phone')
    observed = torch.where(hidden[:, None], torch.full_like(log_probs, -math.log(log_probs.shape[1])), log_probs)
    return F.ctc_loss(observed[:, None], target,
                      torch.tensor([len(observed)]), torch.tensor([len(target)]),
                      blank=0, reduction='sum', zero_infinity=False)


def imitation_loss(log_probs, teacher, excluded):
    """Imitate on observed frames only, including ordinary pauses."""
    import torch
    import torch.nn.functional as F

    if teacher.shape != log_probs.shape or excluded.shape != log_probs.shape[:1] or excluded.dtype != torch.bool:
        raise ValueError('Invalid imitation mask or teacher geometry')
    valid = ~excluded
    if not valid.any():
        return log_probs.sum()*0
    return F.kl_div(log_probs[valid], teacher[valid].exp(), reduction='sum') / valid.sum()

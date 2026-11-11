"""Transcript CTC with float64 log-space accumulation and logit gradients.

This is an opt-in training loss, not an inference probability conversion. Taking
logits lets backward combine probability and occupancy before rounding, instead
of sending rounded occupancy through a separate log-softmax backward operation.
The native loss remains available as the matched arithmetic control.
"""

import torch
from torch.autograd.function import once_differentiable


def _signed_difference(log_a, log_b):
    """exp(log_a) - exp(log_b), retaining close differences with expm1."""
    hi = torch.maximum(log_a, log_b)
    lo = torch.minimum(log_a, log_b)
    magnitude = hi.exp() * -torch.expm1(lo - hi)
    # Two absent masses have difference zero, rather than exp(-inf) * NaN.
    magnitude = torch.where(torch.isneginf(hi), 0.0, magnitude)
    return torch.where(log_a >= log_b, magnitude, -magnitude)


def _recording(log_probs, target, blank):
    """Return NLL and its logit gradient for one valid recording interval."""
    time, vocab = log_probs.shape
    labels = torch.full((2 * target.numel() + 1,), blank, dtype=torch.long, device=log_probs.device)
    labels[1::2] = target
    states = labels.numel()
    if time == 0:
        loss = log_probs.new_tensor(0.0 if target.numel() == 0 else float("inf"))
        return loss, log_probs.clone()
    alpha = log_probs.new_full((time, states), -float("inf"))
    beta = torch.full_like(alpha, -float("inf"))
    skip = torch.zeros(states, dtype=torch.bool, device=labels.device)
    skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    alpha[0, 0] = log_probs[0, blank]
    if states > 1:
        alpha[0, 1] = log_probs[0, labels[1]]
    for t in range(1, time):
        previous = alpha[t - 1]
        incoming = previous.clone()
        incoming[1:] = torch.logaddexp(incoming[1:], previous[:-1])
        incoming[2:] = torch.logaddexp(incoming[2:], torch.where(skip[2:], previous[:-2], -float("inf")))
        alpha[t] = incoming + log_probs[t, labels]
    log_z = alpha[-1, -1] if states == 1 else torch.logaddexp(alpha[-1, -1], alpha[-1, -2])
    if torch.isneginf(log_z):
        return -log_z, torch.full_like(log_probs, float("nan"))
    # beta excludes the current emission; alpha includes it exactly once.
    beta[-1, -1] = 0.0
    if states > 1:
        beta[-1, -2] = 0.0
    for t in range(time - 2, -1, -1):
        following = beta[t + 1] + log_probs[t + 1, labels]
        outgoing = following.clone()
        outgoing[:-1] = torch.logaddexp(outgoing[:-1], following[1:])
        outgoing[:-2] = torch.logaddexp(outgoing[:-2], torch.where(skip[2:], following[2:], -float("inf")))
        beta[t] = outgoing
    log_occupancy = log_probs.new_full((time, vocab), -float("inf"))
    for label in labels.unique().tolist():
        log_occupancy[:, label] = torch.logsumexp(alpha[:, labels == label] + beta[:, labels == label], dim=-1) - log_z
    # Each frame occupies exactly one state. Normalize the posterior in log space
    # to remove accumulated roundoff before the cancellation-sensitive subtraction.
    log_occupancy -= torch.logsumexp(log_occupancy, dim=-1, keepdim=True)
    return -log_z, _signed_difference(log_probs, log_occupancy)


class _StableCTC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, input_lengths, target_lengths, blank, zero_infinity):
        with torch.autocast(device_type=logits.device.type, enabled=False):
            log_probs = logits.double().log_softmax(-1)
            gradient = torch.zeros_like(log_probs)
            losses = []
            offset = 0
            for b, (length, count) in enumerate(zip(input_lengths.tolist(), target_lengths.tolist())):
                target = targets[offset:offset + count].to(device=logits.device, dtype=torch.long)
                offset += count
                loss, grad = _recording(log_probs[:length, b], target, blank)
                if zero_infinity and torch.isinf(loss):
                    loss = torch.zeros_like(loss)
                    grad = torch.zeros_like(grad)
                losses.append(loss)
                gradient[:length, b] = grad
            ctx.save_for_backward(gradient)
            ctx.input_dtype = logits.dtype
            # Retain float64 gradients until after upstream recording weights are
            # applied. Reduced-precision callers still receive a float32 loss.
            dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
            return torch.stack(losses).to(dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (gradient,) = ctx.saved_tensors
        with torch.autocast(device_type=gradient.device.type, enabled=False):
            result = gradient * grad_output.double()[None, :, None]
        return result.to(ctx.input_dtype), None, None, None, None, None


def stable_ctc_loss(logits, targets, input_lengths, target_lengths, *, blank=0, reduction="sum", zero_infinity=True):
    """CTC on (time, batch, vocabulary) logits, preserving recording weights.

    Targets are concatenated, as in the student collator. ``none`` permits explicit
    recording weights; ``sum`` matches the shipped loss before its frame divisor.
    Only first derivatives are supported. Artificial batch-extension frames have
    zero gradient and never enter normalization or forward/backward accumulation.
"""
    if logits.ndim != 3 or not logits.is_floating_point() or logits.shape[1] == 0:
        raise ValueError("logits must be floating (time, nonempty batch, vocabulary)")
    if reduction not in ("sum", "none"):
        raise ValueError("reduction must be sum or none")
    if not 0 <= blank < logits.shape[2]:
        raise ValueError("blank is outside the vocabulary")
    if input_lengths.ndim != 1 or target_lengths.ndim != 1 or len(input_lengths) != logits.shape[1] or len(target_lengths) != logits.shape[1]:
        raise ValueError("one input and target length is required per recording")
    for length in (input_lengths, target_lengths, targets):
        if length.dtype not in (torch.int32, torch.int64):
            raise ValueError("targets and lengths must be integer tensors")
    if (input_lengths < 0).any() or (input_lengths > logits.shape[0]).any() or (target_lengths < 0).any():
        raise ValueError("length is outside the supplied interval")
    if targets.ndim != 1 or target_lengths.sum().item() != targets.numel():
        raise ValueError("targets must contain exactly the concatenated transcripts")
    if ((targets < 0) | (targets >= logits.shape[2]) | (targets == blank)).any():
        raise ValueError("transcripts must contain nonblank vocabulary labels")
    losses = _StableCTC.apply(logits, targets, input_lengths, target_lengths, blank, zero_infinity)
    return losses.sum() if reduction == "sum" else losses

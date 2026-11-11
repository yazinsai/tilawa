"""Transcript-conditioned CTC alignment entropy, without changing the phone head.

The entropy semiring stores (log path mass, conditional mean log path weight).
Forward/backward products add weights; unions average the second member under
normalized incoming mass. The backward pass gives the covariance derivative of
entropy, avoiding an autograd graph with one node per frame and CTC state.
"""

import torch


def _union(log_mass, mean_score):
    mass = torch.logsumexp(log_mass, dim=-1)
    reachable = torch.isfinite(mass)
    origin = torch.where(reachable, mass, torch.zeros_like(mass))
    weights = torch.exp(log_mass - origin.unsqueeze(-1))
    mean = (weights * mean_score).sum(-1)
    return mass, mean


def _predecessors(x, skip, fill):
    one = torch.cat((torch.full_like(x[:, :1], fill), x[:, :-1]), 1)
    two = torch.cat((torch.full_like(x[:, :2], fill), x[:, :-2]), 1)[:, :x.shape[1]]
    two = torch.where(skip, two, torch.full_like(two, fill))
    return torch.stack((x, one, two), -1)


def _successors(x, skip, fill):
    one = torch.cat((x[:, 1:], torch.full_like(x[:, :1], fill)), 1)
    two = torch.cat((x[:, 2:], torch.full_like(x[:, :2], fill)), 1)[:, :x.shape[1]]
    allowed = torch.cat((skip[:, 2:], torch.zeros_like(skip[:, :2])), 1)[:, :x.shape[1]]
    two = torch.where(allowed, two, torch.full_like(two, fill))
    return torch.stack((x, one, two), -1)


class _AlignmentEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, labels, lengths, label_lengths, blank):
        batch, time, vocab = log_probs.shape
        states = 2 * labels.shape[1] + 1
        state_ids = torch.full((batch, states), blank, dtype=torch.long, device=log_probs.device)
        state_ids[:, 1::2] = labels
        index = torch.arange(states, device=log_probs.device)[None]
        valid_state = index <= 2 * label_lengths[:, None]
        skip = (index % 2 == 1) & (index >= 2) & valid_state
        skip[:, 2:] &= state_ids[:, 2:] != state_ids[:, :-2]
        emissions = log_probs.gather(2, state_ids[:, None].expand(batch, time, states))
        emissions = emissions.masked_fill(~valid_state[:, None], -torch.inf)
        mass = torch.full((batch, states), -torch.inf, dtype=log_probs.dtype, device=log_probs.device)
        mass[:, 0] = 0
        mean = torch.zeros_like(mass)
        neutral = mass.clone()
        alphas, scores = [], []
        for t in range(time):
            next_mass, next_mean = _union(_predecessors(mass, skip, -torch.inf),
                                         _predecessors(mean, skip, 0))
            next_mass = next_mass + emissions[:, t]
            next_mean = next_mean + torch.where(torch.isfinite(emissions[:, t]), emissions[:, t], 0)
            next_neutral = torch.logsumexp(_predecessors(neutral, skip, -torch.inf), -1)
            next_neutral = next_neutral.masked_fill(~valid_state, -torch.inf)
            active = (t < lengths)[:, None]
            mass = torch.where(active, next_mass, mass)
            mean = torch.where(active, next_mean, mean)
            neutral = torch.where(active, next_neutral, neutral)
            alphas.append(mass)
            scores.append(mean)
        ends = torch.stack((2 * label_lengths, (2 * label_lengths - 1).clamp_min(0)), -1)
        end_mass = mass.gather(1, ends)
        end_mass[:, 1] = end_mass[:, 1].masked_fill(label_lengths == 0, -torch.inf)
        log_z, expected_score = _union(end_mass, mean.gather(1, ends))
        end_neutral = neutral.gather(1, ends)
        end_neutral[:, 1] = end_neutral[:, 1].masked_fill(label_lengths == 0, -torch.inf)
        baseline = torch.logsumexp(end_neutral, -1)
        if not torch.isfinite(log_z).all():
            raise ValueError("Transcript has no positive-mass CTC alignment")
        # Store forward statistics; no gradient flows through the neutral graph.
        ctx.save_for_backward(emissions, torch.stack(alphas, 1), torch.stack(scores, 1),
                              state_ids, skip, lengths, ends, label_lengths, log_z, expected_score)
        ctx.vocab = vocab
        ctx.mark_non_differentiable(baseline)
        return log_z - expected_score, log_z, baseline

    @staticmethod
    def backward(ctx, entropy_grad, partition_grad, baseline_grad):
        emissions, alpha, score, state_ids, skip, lengths, ends, label_lengths, log_z, expected_score = ctx.saved_tensors
        batch, time, states = emissions.shape
        beta = torch.full_like(emissions[:, 0], -torch.inf)
        beta.scatter_(1, ends[:, :1], 0)
        other = torch.full_like(beta, -torch.inf).scatter(1, ends[:, 1:], 0)
        beta = torch.where((label_lengths > 0)[:, None], torch.maximum(beta, other), beta)
        suffix_score = torch.zeros_like(beta)
        grad = torch.zeros((batch, time, ctx.vocab), dtype=emissions.dtype, device=emissions.device)
        for t in range(time - 1, -1, -1):
            active = (t < lengths)[:, None]
            occupancy = torch.exp(alpha[:, t] + beta - log_z[:, None])
            occupancy = torch.where(active, occupancy, 0)
            conditional_score = score[:, t] + suffix_score
            derivative = occupancy * (expected_score[:, None] - conditional_score) * entropy_grad[:, None]
            derivative = derivative + occupancy * partition_grad[:, None]
            grad[:, t].scatter_add_(1, state_ids, derivative)
            edge_score = suffix_score + torch.where(torch.isfinite(emissions[:, t]), emissions[:, t], 0)
            next_beta, next_score = _union(_successors(beta + emissions[:, t], skip, -torch.inf),
                                           _successors(edge_score, skip, 0))
            beta = torch.where(active, next_beta, beta)
            suffix_score = torch.where(active, next_score, suffix_score)
        return grad, None, None, None, None


def alignment_entropy(log_probs, targets, input_lengths, target_lengths, blank=0):
    """Return per-recording (H[path | audio, transcript], log Z, log path count).

    Inputs follow torch CTC except log_probs is (batch, time, vocab). Targets
    may be packed or padded. Blank is forbidden in transcripts. The neutral
    baseline is the entropy of this exact topology when every emission has
    identical weight; it retains repeated-label constraints and valid lengths.
    """
    if log_probs.ndim != 3 or log_probs.shape[1] == 0:
        raise ValueError("Expected nonempty (batch, time, vocab) log probabilities")
    lengths = torch.as_tensor(input_lengths, device=log_probs.device, dtype=torch.long)
    sizes = torch.as_tensor(target_lengths, device=log_probs.device, dtype=torch.long)
    if lengths.shape != (log_probs.shape[0],) or sizes.shape != lengths.shape:
        raise ValueError("One input and target length required per recording")
    if (lengths < 1).any() or (lengths > log_probs.shape[1]).any() or (sizes < 0).any():
        raise ValueError("Invalid CTC lengths")
    targets = torch.as_tensor(targets, device=log_probs.device, dtype=torch.long)
    sizes_list = sizes.tolist()
    if targets.ndim == 1:
        if targets.numel() != sum(sizes_list):
            raise ValueError("Packed transcript length differs from target lengths")
        labels = torch.zeros((len(sizes_list), max(sizes_list)), dtype=torch.long, device=log_probs.device)
        start = 0
        for i, size in enumerate(sizes_list):
            labels[i, :size] = targets[start:start + size]
            start += size
    elif targets.ndim == 2 and targets.shape[0] == len(sizes_list) and targets.shape[1] >= max(sizes_list):
        labels = targets[:, :max(sizes_list)].clone()
    else:
        raise ValueError("Targets must be packed or padded transcripts")
    mask = torch.arange(labels.shape[1], device=labels.device)[None] < sizes[:, None]
    if ((labels == blank) & mask).any() or ((labels < 0) & mask).any() or ((labels >= log_probs.shape[2]) & mask).any():
        raise ValueError("Transcript contains blank or an unknown phone")
    labels = labels.masked_fill(~mask, blank)
    repeats = ((labels[:, 1:] == labels[:, :-1]) & mask[:, 1:]).sum(1)
    if (sizes + repeats > lengths).any():
        raise ValueError("Insufficient frames for transcript and repeated phones")
    if not 0 <= blank < log_probs.shape[2]:
        raise ValueError("Invalid blank index")
    dtype = torch.float64 if log_probs.dtype == torch.float64 else torch.float32
    return _AlignmentEntropy.apply(log_probs.to(dtype), labels, lengths, sizes, blank)


def entropy_deficit(log_probs, targets, input_lengths, target_lengths, blank=0):
    """Per-frame identity-neutral entropy deficit, minimized to reward alignment diversity."""
    entropy, _, baseline = alignment_entropy(log_probs, targets, input_lengths, target_lengths, blank)
    return (baseline - entropy).sum() / torch.as_tensor(input_lengths, device=log_probs.device).sum().clamp_min(1)

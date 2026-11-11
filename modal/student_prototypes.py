"""Phone classifiers with latent acoustic prototypes and transcript occupancies.

Prototype identity is local to a phone. It is marginalized before the ordinary
phone softmax, so neither CTC nor the streaming SDK sees a larger alphabet.
"""

import torch
from torch import nn
from torch.nn import functional as F


class PrototypeHead(nn.Module):
    def __init__(self, dim: int, vocab: int, prototypes: int = 2):
        super().__init__()
        if prototypes < 1:
            raise ValueError("at least one prototype is required")
        self.dim, self.vocab, self.count = dim, vocab, prototypes
        self.weight = nn.Parameter(torch.randn(vocab, prototypes, dim))
        self.mixture = nn.Parameter(torch.zeros(vocab, prototypes))
        self.bias = nn.Parameter(torch.zeros(vocab))
        self.log_scale = nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        x = F.normalize(x, dim=-1, eps=1e-6)
        w = F.normalize(self.weight, dim=-1, eps=1e-6)
        similarities = F.linear(x, w.flatten(0, 1)).unflatten(-1, (self.vocab, self.count))
        # Normalized mixture prevents extra prototypes from increasing class
        # mass merely because the class has more latent alternatives.
        scores = similarities * self.log_scale.clamp(-4, 5).exp()
        return torch.logsumexp(scores + self.mixture.log_softmax(-1), dim=-1) + self.bias

    @torch.no_grad()
    def initialize(self, features, occupancies, seed: int = 0, iterations: int = 15):
        """Weighted spherical clustering using soft transcript-constrained labels.

        Both blank and phones use the same procedure. No greedy class filtering
        discards the legitimate renditions currently misclassified by the head.
        """
        x = F.normalize(features, dim=-1)
        if occupancies.shape != (len(x), self.vocab):
            raise ValueError("one occupancy per frame and phone is required")
        if not torch.isfinite(occupancies).all() or (occupancies < 0).any():
            raise ValueError("occupancies must be finite and nonnegative")
        gen = torch.Generator(device=x.device).manual_seed(seed)
        support = occupancies.sum(0)
        if (support <= 0).any():
            raise ValueError("initialization bank must cover every phone, including blank")
        for phone in range(self.vocab):
            weights = occupancies[:, phone]
            chosen = torch.multinomial(weights, 1, generator=gen).item()
            centers = [x[chosen]]
            for _ in range(1, self.count):
                distance = (1 - x @ torch.stack(centers).T).clamp_min(0).min(-1).values
                sampling = weights * distance
                if sampling.sum() <= 0:
                    sampling = weights
                centers.append(x[torch.multinomial(sampling, 1, generator=gen).item()])
            centers = torch.stack(centers)
            counts = torch.zeros(self.count, device=x.device)
            for _ in range(iterations):
                assignment = (x @ centers.T).argmax(-1)
                for k in range(self.count):
                    wk = weights * (assignment == k)
                    counts[k] = wk.sum()
                    if counts[k] > 0:
                        centers[k] = F.normalize((wk[:, None] * x).sum(0), dim=0)
            self.weight[phone].copy_(centers)
            self.mixture[phone].copy_(counts.clamp_min(1e-6).log())
        return support


class ConventionalHead(nn.Module):
    """Conventional linear classifier plus a parameter-matched residual MLP."""
    def __init__(self, linear: nn.Linear, prototype_count: int):
        super().__init__()
        dim, vocab = linear.in_features, linear.out_features
        budget = vocab * prototype_count * dim + vocab * prototype_count + vocab + 1
        base = vocab * (dim + 1)
        hidden = max(1, round((budget - base) / (dim + 1 + vocab)))
        self.linear = linear
        self.residual = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, vocab, bias=False))
        nn.init.zeros_(self.residual[-1].weight)

    def forward(self, x):
        return self.linear(x) + self.residual(x)


def ctc_occupancies(log_probs, labels, blank: int = 0):
    """Forward-backward frame occupancies conditioned on a complete transcript.

    Repeated articulations retain CTC's blank-separation constraint. Work in
    float64; return class occupancies, including blank, on the native frame clock.
    """
    if log_probs.ndim != 2 or len(log_probs) == 0:
        raise ValueError("expected nonempty frame-by-phone log probabilities")
    lp = log_probs.detach().double()
    labels = torch.as_tensor(labels, dtype=torch.long, device=lp.device)
    if ((labels <= blank) | (labels >= lp.shape[1])).any():
        raise ValueError("transcript contains blank or an invalid phone")
    states = torch.full((len(labels) * 2 + 1,), blank, device=lp.device, dtype=torch.long)
    states[1::2] = labels
    emissions = lp[:, states]
    skip = torch.zeros(len(states), dtype=torch.bool, device=lp.device)
    skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    alpha = torch.full_like(emissions, -torch.inf)
    alpha[0, 0] = emissions[0, 0]
    if len(states) > 1:
        alpha[0, 1] = emissions[0, 1]
    for t in range(1, len(lp)):
        prev = alpha[t - 1]
        step = F.pad(prev[:-1], (1, 0), value=-torch.inf)
        jump = F.pad(prev[:-2], (2, 0), value=-torch.inf) if len(states) > 1 else prev.new_full(prev.shape, -torch.inf)
        alpha[t] = torch.logsumexp(torch.stack((prev, step, jump.masked_fill(~skip, -torch.inf))), 0) + emissions[t]
    z = torch.logsumexp(alpha[-1, -2:] if len(states) > 1 else alpha[-1], 0)
    if not torch.isfinite(z):
        raise ValueError("transcript has no complete CTC alignment")
    beta = torch.full_like(emissions, -torch.inf)
    beta[-1, -2:] = 0
    for t in range(len(lp) - 2, -1, -1):
        nxt = beta[t + 1] + emissions[t + 1]
        step = F.pad(nxt[1:], (0, 1), value=-torch.inf)
        jump = F.pad(nxt[2:].masked_fill(~skip[2:], -torch.inf), (0, 2), value=-torch.inf) if len(states) > 1 else nxt.new_full(nxt.shape, -torch.inf)
        beta[t] = torch.logsumexp(torch.stack((nxt, step, jump)), 0)
    occupancy = (alpha + beta - z).exp()
    result = lp.new_zeros(lp.shape)
    result.scatter_add_(1, states.expand(len(lp), -1), occupancy)
    return result.float()

"""Factorized CTC residuals and an exactly matched shared-feature control.

The encoder and baseline classifier are frozen observations. Timing changes
blank/emission odds; identity changes only the conditional nonblank distribution.
Both branches still receive gradients from the same transcript CTC lattice.
"""
import copy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

SEED = 1729


class TimingIdentityHead(nn.Module):
    def __init__(self, dim: int, vocab: int, hidden: int = 64, shared: bool = False):
        super().__init__()
        if dim < 1 or vocab < 3 or hidden < 1:
            raise ValueError("Invalid timing/identity dimensions")
        self.shared = shared
        self.timing_features = nn.Linear(dim, hidden)
        self.identity_features = nn.Linear(dim, hidden)
        self.timing = nn.Linear(hidden, 1)
        self.identity = nn.Linear(hidden, vocab - 1)
        for layer in (self.timing, self.identity):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # A fixed orthogonal DCT routing mixes every hidden feature into both
        # readouts in the shared arm, without adding trainable/padding parameters.
        # Apply it AFTER tanh: rotating the affine weights before tanh would
        # merely reparameterize two still-disjoint nonlinear feature blocks.
        size = 2 * hidden
        rows = torch.arange(size, dtype=torch.float64)[:, None]
        cols = torch.arange(size, dtype=torch.float64)[None, :]
        routing = torch.cos(math.pi / size * (cols + .5) * rows) * math.sqrt(2 / size)
        routing[0] /= math.sqrt(2)
        self.register_buffer("routing", routing.float())

    def forward(self, features):
        timing = torch.tanh(self.timing_features(features))
        identity = torch.tanh(self.identity_features(features))
        if self.shared:
            mixed = torch.cat((timing, identity), dim=-1) @ self.routing.T
            timing, identity = mixed.chunk(2, dim=-1)
        return self.timing(timing), self.identity(identity)


def heads(dim: int, vocab: int, hidden: int = 64):
    # Exactly the same trainable tensors, not just the same initial output.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        separate = TimingIdentityHead(dim, vocab, hidden)
    shared = copy.deepcopy(separate)
    shared.shared = True
    return {"separated": separate, "shared": shared}


def phone_log_probs(head, features, shipped, center, scale):
    x = (features.detach() - center.detach()) / scale.detach()
    timing, identity = head(x)
    base = shipped.detach().log_softmax(-1)
    nonblank = base[..., 1:]
    mass = torch.logsumexp(nonblank, dim=-1, keepdim=True)
    odds = mass - base[..., :1] + timing
    # Bernoulli timing times categorical identity is an ordinary normalized
    # CTC distribution; no speech-probability freeze or extra inference rule.
    return torch.cat((F.logsigmoid(-odds),
                      F.logsigmoid(odds) + (nonblank + identity).log_softmax(-1)), dim=-1)


def partition(speakers: list[str], folds: int = 3) -> dict[str, int]:
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if folds < 2 or len(unique) < folds:
        raise ValueError("Not enough training speakers for grouped selection")
    return {speaker: i % folds for i, speaker in enumerate(unique)}


def objective(log_probs, target, shipped, preservation: float = 1.):
    """Full-recording transcript CTC plus shipped-output KL, including pauses."""
    minimum = len(target) + sum(a == b for a, b in zip(target.tolist(), target.tolist()[1:]))
    if not len(target) or len(log_probs) < minimum:
        raise ValueError("Utterance cannot carry its complete phone transcript")
    length = torch.tensor([len(log_probs)], dtype=torch.long)
    labels = torch.tensor([len(target)], dtype=torch.long)
    ctc = F.ctc_loss(log_probs[:, None], target, length, labels, blank=0,
                     reduction="sum", zero_infinity=False) / len(target)
    anchor = shipped.detach().log_softmax(-1)
    kl = (anchor.exp() * (anchor - log_probs)).sum(-1).mean()
    return ctc + preservation * kl

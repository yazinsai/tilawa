"""Frozen-encoder phone heads with genuinely acoustic-conditioned code interactions.

The residual is added to the shipped logits; its zero initialization preserves
the shipped distribution. No bitwise Bernoulli likelihood is normalized here.
All choices and the matched dense control are independent of evaluation audio.
"""
import hashlib
import itertools
import math
import random

import torch
from torch import nn


def fixed_code(vocab: int, bits: int = 16) -> torch.Tensor:
    """Unique constant-weight codes, including blank, without phone feature labels.

    Greedy column balancing and Hamming separation use only inventory size and
    a fixed seed. Neither a transcript nor an acoustic observation enters.
    """
    if bits % 2 or vocab < 2 or math.comb(bits, bits // 2) // 2 < vocab:
        raise ValueError("Code requires an even width and enough non-complementary rows")
    pool = list(itertools.combinations(range(bits), bits // 2))
    random.Random(1729).shuffle(pool)
    chosen, counts = [], [0] * bits
    masks = {row: sum(1 << i for i in row) for row in pool}
    chosen_masks = []
    for _ in range(vocab):
        def rank(row):
            distance = min(((masks[row] ^ old).bit_count() for old in chosen_masks), default=bits)
            imbalance = 2 * sum(counts[i] for i in row)
            return distance, -imbalance
        row = max(pool, key=rank)
        pool.remove(row)
        # Complementary signed rows have identical pair products. Never give
        # two phones that same nonlinear score dictionary, especially blank.
        pool.remove(tuple(i for i in range(bits) if i not in row))
        chosen.append(row)
        chosen_masks.append(masks[row])
        counts = [n + int(i in row) for i, n in enumerate(counts)]
    return torch.tensor([[1. if i in row else -1. for i in range(bits)] for row in chosen])


def matched_widths(dim: int, vocab: int, bits: int = 16) -> tuple[int, int]:
    """Find small exact parameter counts; no unused/padded trainable parameters.

    Code: bias-free D->K singles, biased D->B bottleneck, bias-free B->P
    interaction coefficients. Dense: biased D->H, bias-free H->V.
    For the shipped D=256,V=43,K=16 this gives B=52,H=79 (23,700 each).
    """
    pairs = bits * (bits - 1) // 2
    for bottleneck in range(16, 513):
        count = dim * bits + (dim + 1 + pairs) * bottleneck
        if count % (dim + 1 + vocab) == 0:
            return bottleneck, count // (dim + 1 + vocab)
    raise ValueError("No bounded exact-parameter dense control for this encoder")


class InteractingCodeHead(nn.Module):
    def __init__(self, dim: int, code: torch.Tensor, bottleneck: int):
        super().__init__()
        bits = code.shape[1]
        self.register_buffer("code", code.clone() / math.sqrt(bits))
        pairs = torch.stack([code[:, i] * code[:, j]
                             for i, j in itertools.combinations(range(bits), 2)], dim=1)
        self.register_buffer("pairs", pairs / math.sqrt(pairs.shape[1]))
        self.single = nn.Linear(dim, bits, bias=False)
        self.bottleneck = nn.Linear(dim, bottleneck)
        self.interaction = nn.Linear(bottleneck, pairs.shape[1], bias=False)
        nn.init.zeros_(self.single.weight)
        nn.init.zeros_(self.interaction.weight)

    def forward(self, features):
        singles = self.single(features) @ self.code.T
        coefficients = self.interaction(torch.tanh(self.bottleneck(features)))
        return singles + coefficients @ self.pairs.T


class DenseHead(nn.Module):
    def __init__(self, dim: int, vocab: int, hidden: int):
        super().__init__()
        self.hidden = nn.Linear(dim, hidden)
        self.output = nn.Linear(hidden, vocab, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, features):
        return self.output(torch.tanh(self.hidden(features)))


def heads(dim: int, vocab: int):
    bottleneck, hidden = matched_widths(dim, vocab)
    # Independent reproducible initializations; both residual functions start at 0.
    torch.manual_seed(1729)
    code = InteractingCodeHead(dim, fixed_code(vocab), bottleneck)
    torch.manual_seed(1729)
    dense = DenseHead(dim, vocab, hidden)
    assert sum(p.numel() for p in code.parameters()) == sum(p.numel() for p in dense.parameters())
    return {"interacting": code, "dense": dense}


def partition(speakers: list[str], folds: int = 3) -> dict[str, int]:
    """Stable group split: all recordings/views of a speaker stay together."""
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(unique) < folds:
        raise ValueError("Not enough training speakers for grouped selection")
    return {speaker: i % folds for i, speaker in enumerate(unique)}


def phone_log_probs(head, features, shipped, center, scale):
    return (shipped + head((features - center) / scale)).log_softmax(-1)


def objective(log_probs, target, shipped, preservation: float):
    """Canonical CTC plus output preservation on the same full real utterance.

    The encoder and shipped outputs are detached observations, not parameters.
    Reject impossible alignments rather than silently dropping their CTC loss.
    """
    from torch.nn import functional as F
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

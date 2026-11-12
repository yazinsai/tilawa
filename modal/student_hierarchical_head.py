"""Independent nonlinear conditional readouts over one ordinary CTC inventory.

Frozen shipped branch masses initialize the conditional priors. Zero residual
outputs reproduce shipped probabilities, including real speech and pauses.
Training learns discrimination at each node; leaf log probabilities are sums
along their unique paths, not another flat normalization of phone scores.
"""
import hashlib

import torch
from torch import nn

from student_hierarchical_tree import matched_widths, phonetic_tree


class NonlinearNode(nn.Module):
    def __init__(self, dim, hidden, arity):
        super().__init__()
        self.hidden = nn.Linear(dim, hidden)
        self.output = nn.Linear(hidden, arity, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, features):
        return self.output(torch.tanh(self.hidden(features)))


class HierarchicalHead(nn.Module):
    def __init__(self, dim, tokens, hidden):
        super().__init__()
        self.nodes, self.paths = phonetic_tree(tokens)
        self.branches = nn.ModuleList([NonlinearNode(dim, hidden, len(n.children)) for n in self.nodes])

    def forward(self, features, shipped):
        conditional = []
        for node, classifier in zip(self.nodes, self.branches):
            priors = torch.stack([torch.logsumexp(shipped[..., list(child)], -1)
                                  for child in node.children], -1)
            conditional.append((priors + classifier(features)).log_softmax(-1))
        return torch.stack([sum(conditional[node][..., child] for node, child in path)
                            for path in self.paths], -1)


class FlatHead(NonlinearNode):
    def forward(self, features, shipped):
        return (shipped + super().forward(features)).log_softmax(-1)


def heads(dim, tokens):
    node_hidden, flat_hidden, count = matched_widths(dim, tokens)
    torch.manual_seed(1729)
    tree = HierarchicalHead(dim, tokens, node_hidden)
    torch.manual_seed(1729)
    flat = FlatHead(dim, flat_hidden, len(tokens) + 1)
    assert sum(p.numel() for p in tree.parameters()) == sum(p.numel() for p in flat.parameters()) == count
    return {"hierarchical": tree, "flat": flat}


def partition(speakers: list[str], folds: int = 3) -> dict[str, int]:
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if folds < 2 or len(unique) < folds or any(not isinstance(s, str) or not s for s in unique):
        raise ValueError("Not enough identified training speakers for grouped selection")
    return {speaker: i % folds for i, speaker in enumerate(unique)}


def phone_log_probs(head, features, shipped, center, scale):
    return head((features.detach() - center) / scale, shipped.detach())


def objective(log_probs, target, shipped, preservation: float):
    """Ordinary complete-transcript CTC plus KL to frozen speech/pause outputs."""
    from torch.nn import functional as F
    ids = target.tolist()
    minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
    if not ids or len(log_probs) < minimum:
        raise ValueError("Utterance cannot carry its complete phone transcript")
    length = torch.tensor([len(log_probs)], dtype=torch.long)
    labels = torch.tensor([len(ids)], dtype=torch.long)
    ctc = F.ctc_loss(log_probs[:, None], target, length, labels, blank=0,
                     reduction="sum", zero_infinity=False) / len(ids)
    anchor = shipped.detach().log_softmax(-1)
    kl = (anchor.exp() * (anchor - log_probs)).sum(-1).mean()
    return ctc + preservation * kl

"""Audio-only specialization of the last Zipformer residual feed-forward block.

The ordinary CTC head and all streaming caches are unchanged. Three identical
copies start as the shipped block, so every routing distribution initially
produces the shipped function. Transcript-conditioned occupancy is a training
target only; the exported router receives acoustic activations alone.
"""

import copy

import torch
from torch import nn


REGIMES = ("transient", "sustained", "silence")
# Inventory sound classes, not reference text or a list of evaluation words.
SUSTAINED = frozenset("منوياۦۥَُِںٲ۾۪ـ")
SILENCE = frozenset("ؙۜ")


class RoutedFeedforward(nn.Module):
    def __init__(self, original):
        super().__init__()
        dim = original.in_proj.in_features
        self.experts = nn.ModuleList(copy.deepcopy(original) for _ in REGIMES)
        self.router = nn.Linear(dim, len(REGIMES))
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.route_logits = None

    def forward(self, x):
        logits = self.router(x)
        if self.training:
            self.route_logits = logits
        weights = logits.softmax(-1)
        return sum(expert(x) * weights[..., i:i + 1]
                   for i, expert in enumerate(self.experts))


class DenseFeedforward(nn.Module):
    """An identity-initialized dense control with exactly the same parameter count.

Three concatenated hidden groups reproduce the original at initialization.
Two extra hidden channels, an input gain and per-output gains account for the
router and the mixture's extra output biases. Every parameter affects the
function; there is no unused parameter budget or regime input in this arm.
"""
    def __init__(self, original):
        super().__init__()
        from zipformer import FeedforwardModule

        dim, hidden = original.in_proj.in_features, original.in_proj.out_features
        self.dense = FeedforwardModule(dim, hidden * 3 + 2,
                                       copy.deepcopy(original.out_proj.dropout_p))
        self.input_gain = nn.Parameter(torch.ones(()))
        self.output_gain = nn.Parameter(torch.ones(dim))
        with torch.no_grad():
            for i in range(3):
                sl = slice(i * hidden, (i + 1) * hidden)
                self.dense.in_proj.weight[sl].copy_(original.in_proj.weight)
                self.dense.in_proj.bias[sl].copy_(original.in_proj.bias)
                self.dense.out_proj.weight[:, sl].copy_(original.out_proj.weight / 3)
            self.dense.out_proj.weight[:, hidden * 3:].zero_()
            self.dense.out_proj.bias.copy_(original.out_proj.bias)

    def forward(self, x):
        return self.dense(x * self.input_gain) * self.output_gain


def replace_upper_block(model, arm="routed"):
    """Replace just the final layer's last residual FF, in both forward paths."""
    if model.encoder.downsampling_factor[-1] != model.encoder.output_downsampling_factor:
        raise ValueError("The upper block must share the ordinary CTC frame clock")
    layers = [m for m in model.encoder.modules()
              if m.__class__.__name__ == "Zipformer2EncoderLayer"]
    if not layers:
        raise ValueError("No Zipformer upper block found")
    layer = layers[-1]
    if not hasattr(layer.feed_forward3, "in_proj"):
        raise ValueError("Upper block was already replaced")
    cls = {"routed": RoutedFeedforward, "dense": DenseFeedforward}[arm]
    layer.feed_forward3 = cls(layer.feed_forward3)
    return layer.feed_forward3


def regime_map(tokens):
    return torch.tensor([2] + [2 if p in SILENCE else 1 if p in SUSTAINED else 0 for p in tokens])


@torch.no_grad()
def ctc_regime_occupancy(log_probs, target, classes):
    """Exact forward-backward occupancy conditional on an authorized transcript.

Input is one (T,V) trajectory. Blank states retain a silence target, repeated
phones require intervening blank, and every frame is consumed exactly once.
Returns None for an impossible transcript; it never fabricates alignment labels.
"""
    lp = log_probs.detach().float()
    labels = torch.zeros(2 * target.numel() + 1, dtype=torch.long, device=lp.device)
    labels[1::2] = target.to(lp.device)
    times, states = lp.shape[0], labels.numel()
    if not times:
        return None
    emission = lp[:, labels]
    skip = torch.zeros(states, dtype=torch.bool, device=lp.device)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    alpha = torch.full_like(emission, -torch.inf)
    alpha[0, 0] = emission[0, 0]
    if states > 1:
        alpha[0, 1] = emission[0, 1]
    for t in range(1, times):
        prev = alpha[t - 1]
        one = torch.cat((prev.new_full((1,), -torch.inf), prev[:-1]))
        two = torch.cat((prev.new_full((min(2, states),), -torch.inf), prev[:-2]))
        alpha[t] = emission[t] + torch.logsumexp(
            torch.stack((prev, one, two.masked_fill(~skip, -torch.inf))), 0)
    z = torch.logsumexp(alpha[-1, max(0, states - 2):], 0)
    if not torch.isfinite(z):
        return None
    beta = torch.full_like(emission, -torch.inf)
    beta[-1, max(0, states - 2):] = 0
    for t in range(times - 2, -1, -1):
        nxt = emission[t + 1] + beta[t + 1]
        one = torch.cat((nxt[1:], nxt.new_full((1,), -torch.inf)))
        two = torch.cat((nxt[2:].masked_fill(~skip[2:], -torch.inf),
                         nxt.new_full((min(2, states),), -torch.inf)))
        beta[t] = torch.logsumexp(torch.stack((nxt, one, two)), 0)
    posterior = (alpha + beta - z).exp()
    grouped = lp.new_zeros(times, len(REGIMES))
    grouped.scatter_add_(1, classes.to(lp.device)[labels].expand(times, states), posterior)
    return grouped / grouped.sum(-1, keepdim=True).clamp_min(1e-12)


def router_loss(logits, occupancy, lengths):
    """Supervise the final internal block on its ordinary 40 ms CTC clock.

    The last downsampled encoder works at factor two before upsampling its
    residual output; the final CTC downsampler also has factor two.
    Normalize within each regime before averaging, so blank/held frames cannot
    overwhelm the rarer consonant objective. Padding contributes no supervision.
    """
    # logits (upper time,batch,regime), occupancy (batch,CTC time,regime)
    prediction = logits.permute(1, 0, 2).float().log_softmax(-1)
    if prediction.shape != occupancy.shape:
        raise ValueError("Router and CTC occupancy have different frame clocks")
    valid = torch.arange(prediction.shape[1], device=logits.device)[None] < lengths[:, None]
    target = occupancy * valid[..., None]
    counts = target.sum((0, 1))
    active = counts > 0
    if not active.any():
        return logits.sum() * 0
    loss = -(target * prediction).sum((0, 1)) / counts.clamp_min(1)
    return loss[active].mean()

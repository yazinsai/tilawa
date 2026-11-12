"""Continuous integrate-and-fire acoustic phone events and a matched framewise CTC control.

Only training-time identity supervision rescales mass to the transcript length. Export and free
decoding always use raw learned mass. The affine event classifier commutes with weighted pooling:
the SDK accumulates its *logits*, never their softmax or the auxiliary CTC probabilities.
"""

import torch
from torch import nn
from torch.nn import functional as F


def attach_heads(model):
    dim, vocab = model.ctc_output[1].weight.shape[1], model.ctc_output[1].weight.shape[0]
    model.phone_mass = nn.Linear(dim, 1)
    model.phone_output = nn.Linear(dim, vocab)
    with torch.no_grad():
        model.phone_mass.weight.zero_()
        model.phone_mass.bias.fill_(0.25)
        model.phone_output.load_state_dict(model.ctc_output[1].state_dict())
    return model


def acoustic_outputs(model, features):
    # Exact zeros allow pauses to stay outside event intervals; each frame contributes at most one unit.
    return model.phone_mass(features).squeeze(-1).clamp(0, 1), model.phone_output(features)


def pool(mass, features, lengths):
    """Differentiable overlap between cumulative acoustic mass and unit event bins.

    lengths are the requested number of bins, not a normalization. Callers may rescale mass
    explicitly for *training*, but inference callers must supply the raw acoustic mass.
    """
    maximum = int(lengths.max().item()) if lengths.numel() else 0
    ends = mass.cumsum(1)
    starts = ends - mass
    bins = torch.arange(maximum, device=mass.device, dtype=mass.dtype)
    weights = (torch.minimum(ends[..., None], bins + 1) - torch.maximum(starts[..., None], bins)).clamp_min(0)
    weights = weights * (bins[None, None] < lengths[:, None, None])
    return torch.bmm(weights.transpose(1, 2), features), weights


def objective(model, batch, arm):
    features, lengths = model.encode(batch["feats"], batch["feat_lens"])
    features = features.float()
    valid = torch.arange(features.shape[1], device=features.device)[None] < lengths[:, None]
    mass, logits = acoustic_outputs(model, features)
    mass = mass * valid
    count = ((mass.sum(1) - batch["target_lens"]).abs() / batch["target_lens"].clamp_min(1)).mean()
    denom = batch["target_lens"].sum().clamp_min(1)

    def ctc(lp):
        return F.ctc_loss(lp.transpose(0, 1), batch["targets"], lengths, batch["target_lens"],
                          blank=0, reduction="sum", zero_infinity=False) / denom

    auxiliary = ctc(model.ctc_output(features).log_softmax(-1))
    # Both arms execute the same heads and pooling work, with equal parameter counts and data.
    # Target-count scaling belongs solely to the teacher-forced training identity objective.
    scaled = mass * (batch["target_lens"] / mass.sum(1).clamp_min(1e-8))[:, None]
    pooled, _ = pool(scaled, features, batch["target_lens"])
    event_logits = model.phone_output(pooled)
    mask = torch.arange(pooled.shape[1], device=features.device)[None] < batch["target_lens"][:, None]
    event_identity = F.cross_entropy(event_logits[mask], batch["targets"], reduction="sum") / denom if mask.any() else logits.sum() * 0
    frame_identity = ctc(logits.log_softmax(-1))
    if arm == "cif":
        identity = event_identity + frame_identity * 0
    elif arm == "ctc":
        identity = frame_identity + event_identity * 0
    else:
        raise ValueError(arm)
    return dict(loss=identity + auxiliary + count, identity=identity, auxiliary=auxiliary, count=count)


def decode_events(mass, scores):
    """Reference exported-stream decoder; returns phone ids and exact frame contributions."""
    if not len(mass):
        return []
    events, accumulated, weighted, contributions = [], 0.0, torch.zeros_like(scores[0]), []

    def fire():
        nonlocal accumulated, weighted, contributions
        best = int((weighted / accumulated).argmax())
        if best:
            events.append(dict(id=best, mass=accumulated, contributions=contributions))
        accumulated, weighted, contributions = 0.0, torch.zeros_like(weighted), []

    for frame, value in enumerate(mass.tolist()):
        if not 0 <= value <= 1:
            raise ValueError("invalid mass")
        while value > 0:
            take = min(value, 1 - accumulated)
            weighted += take * scores[frame]
            accumulated += take
            contributions.append(dict(frame=frame, mass=take))
            value -= take
            if accumulated >= 1:
                fire()
    if accumulated >= 0.5:
        fire()
    return events

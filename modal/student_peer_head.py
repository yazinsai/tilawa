"""Reciprocal stop-gradient learning of three independent acoustic readouts.

Only one head is exported. Frozen encoder observations and preservation outputs
never receive gradients. Agreement targets are recomputed before each joint
update, not carried forward from a teacher or from an earlier minibatch.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from student_peer_data import SEED, recording_orders

PRESERVATION = .1


class PhoneHead(nn.Module):
    def __init__(self, width, vocab, hidden=64):
        super().__init__()
        self.hidden = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, vocab)

    def forward(self, x):
        return self.output(torch.tanh(self.hidden(x)))


def heads(width, vocab, seed=SEED):
    # fork_rng avoids mutating the caller's augmentation/sampling RNG state.
    result = []
    with torch.random.fork_rng():
        for i in range(3):
            torch.manual_seed(seed + i)
            result.append(PhoneHead(width, vocab))
    return nn.ModuleList(result)


def phone_log_probs(head, features, center, scale):
    return head((features.detach() - center.detach()) / scale.detach()).log_softmax(-1)


def transcript_loss(lp, target):
    phones = target.tolist()
    minimum = len(phones) + sum(a == b for a, b in zip(phones, phones[1:]))
    if not phones or len(lp) < minimum:
        raise ValueError("Utterance cannot carry its complete phone transcript")
    return F.ctc_loss(lp[:, None], target, torch.tensor([len(lp)]),
                      torch.tensor([len(target)]), blank=0, reduction="sum",
                      zero_infinity=False) / len(target)


def preservation_loss(lp, shipped):
    anchor = shipped.detach().log_softmax(-1)
    return (anchor.exp() * (anchor - lp)).sum(-1).mean()


def agreement_loss(lp, peer_log_probs):
    """KL from the arithmetic probability mean of the two other readouts.

    Detach both the probabilities and entropy. Averaging log probabilities
    instead would favor consensus modes and intensify blank collapse.
    """
    if len(peer_log_probs) != 2 or any(p.shape != lp.shape for p in peer_log_probs):
        raise ValueError("Agreement requires exactly two peers on the same frames")
    target = torch.stack([p.detach().exp() for p in peer_log_probs]).mean(0)
    return (target * (target.clamp_min(torch.finfo(target.dtype).tiny).log() - lp)).sum(-1).mean()


def frozen_peers(models):
    snapshot = copy.deepcopy(models).eval()
    for p in snapshot.parameters():
        p.requires_grad_(False)
    return snapshot


def joint_update(models, optimizers, recordings, center, scale, strength,
                 target_models=None, agreement_enabled=True):
    """Compute all three losses before stepping any optimizer.

    Recipients may have different recordings, but their peers always predict
    that recipient's exact cached views. Independent and frozen arms execute
    the same six detached peer forwards and three differentiable forwards per
    view as the live arm; they differ only in target source/loss activation.
    """
    if len(models) != 3 or len(optimizers) != 3 or len(recordings) != 3 or strength < 0:
        raise ValueError("Joint updates require three heads, optimizers and recordings")
    peers = models if target_models is None else target_models
    if len(peers) != 3:
        raise ValueError("Joint updates require three target readouts")
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    losses = []
    for i, recording in enumerate(recordings):
        values = []
        for view in recording["views"]:
            lp = phone_log_probs(models[i], view["features"], center, scale)
            with torch.no_grad():
                target = [phone_log_probs(peers[j], view["features"], center, scale)
                          for j in range(3) if i != j]
            # Compute agreement even for independent/warmup arms to match work.
            agreement = agreement_loss(lp, target)
            loss = transcript_loss(lp, view["target"]) + PRESERVATION * preservation_loss(lp, view["shipped"])
            values.append(loss + (strength if agreement_enabled else 0.) * agreement)
        losses.append(torch.stack(values).mean())
    if not all(torch.isfinite(value) for value in losses):
        raise ValueError("Non-finite reciprocal acoustic loss")
    for loss in losses:
        loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all()
           for model in models for p in model.parameters()):
        raise ValueError("Non-finite reciprocal acoustic gradient")
    for model, optimizer in zip(models, optimizers):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    return [float(loss.detach()) for loss in losses]


def fit_heads(arm, rows, width, vocab, center, scale, strength, epochs=18, warmup=6):
    if arm not in ("live", "independent", "frozen") or not 0 < warmup < epochs:
        raise ValueError("Unknown arm or invalid matched warmup/exposure")
    models = heads(width, vocab)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.01) for m in models]
    snapshots = None
    exposure = [[0] * len(rows) for _ in models]
    for epoch in range(epochs):
        if epoch == warmup:
            # All arms have identical CTC warmup and matched snapshot work.
            snapshots = frozen_peers(models)
        orders = recording_orders(len(rows), epoch)
        for indices in zip(*orders):
            joint_update(models, optimizers, [rows[i] for i in indices], center, scale, strength,
                         target_models=snapshots if arm == "frozen" and epoch >= warmup else None,
                         agreement_enabled=arm != "independent" and epoch >= warmup)
            for peer, index in enumerate(indices):
                exposure[peer][index] += 1
    if any(n != epochs for counts in exposure for n in counts):
        raise AssertionError("Readouts did not receive matched complete exposure")
    accounting = dict(recordings=len(rows), views=sum(len(r["views"]) for r in rows), epochs=epochs,
                      warmup=warmup, optimizerStepsPerHead=epochs * len(rows),
                      differentiableForwards=3 * epochs * sum(len(r["views"]) for r in rows),
                      detachedPeerForwards=6 * epochs * sum(len(r["views"]) for r in rows),
                      exposures=exposure)
    return models.eval(), accounting

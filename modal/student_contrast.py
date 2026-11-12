"""Identifiable phone-head coordinates for a matched frozen-encoder refit.

The fixed Helmert basis includes CTC blank. ScaledAdam receives only C-1
contrast rows, rather than a redundant C-row classifier. Ordinary controls
retain the original coordinates and differ only by a common-logit shift.
This module reads neither audio nor practice evidence.
"""

import copy
import hashlib
import math
import random

import torch
from torch import nn
from torch.nn import functional as F


def contrast_basis(classes, *, dtype=torch.float32, device=None):
    if classes < 2:
        raise ValueError("Need blank and at least one phone")
    basis = torch.zeros(classes, classes - 1, dtype=dtype, device=device)
    for j in range(classes - 1):
        scale = math.sqrt((j + 1) * (j + 2))
        basis[:j + 1, j] = 1 / scale
        basis[j + 1, j] = -(j + 1) / scale
    return basis


class ContrastHead(nn.Module):
    def __init__(self, linear):
        super().__init__()
        if not isinstance(linear, nn.Linear) or linear.bias is None:
            raise ValueError("Need the complete existing affine phone head")
        basis = contrast_basis(linear.out_features, dtype=linear.weight.dtype,
                               device=linear.weight.device)
        self.register_buffer("basis", basis)
        self.out_features = linear.out_features
        # Center in double precision before projection to avoid cancellation
        # from a large, functionally irrelevant common component.
        w, b = linear.weight.detach().double(), linear.bias.detach().double()
        self.weight = nn.Parameter((basis.double().T @ (w - w.mean(0))).to(linear.weight.dtype))
        self.bias = nn.Parameter((basis.double().T @ (b - b.mean())).to(linear.bias.dtype))

    def forward(self, features):
        return F.linear(features, self.basis @ self.weight, self.basis @ self.bias)

    def folded(self):
        """The export uses the existing ordinary affine head and alphabet."""
        linear = nn.Linear(self.weight.shape[1], self.basis.shape[0],
                           device=self.weight.device, dtype=self.weight.dtype)
        with torch.no_grad():
            linear.weight.copy_(self.basis @ self.weight)
            linear.bias.copy_(self.basis @ self.bias)
        return linear


def ordinary_controls(linear, seed):
    """Original, centered and RMS-matched shifted ordinary classifiers.

    The shift direction and magnitude use only the source head and a fixed
    random draw, never training outcomes or target evidence.
    """
    centered = ContrastHead(linear).folded()
    shifted = copy.deepcopy(centered)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(linear.in_features, generator=rng, dtype=torch.float64)
    direction /= direction.square().mean().sqrt()
    shift = direction.to(linear.weight) * centered.weight.detach().square().mean().sqrt()
    with torch.no_grad():
        shifted.weight.add_(shift[None])
        shifted.bias.add_(centered.bias.detach().square().mean().sqrt())
    return {"ordinary": copy.deepcopy(linear), "centered": centered, "shifted": shifted}


def speaker_partition(speakers):
    if any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Every recording needs an identified training speaker")
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(unique) < 5:
        raise ValueError("Need at least five training speakers")
    return {s: "selection" if i % 5 == 0 else "fit" for i, s in enumerate(unique)}


def admit_recordings(records, excluded):
    """Fail closed on missing provenance; all non-train speakers are excluded.

    None-labelled takes use canonical transcripts. Other recordings require
    authorized performed transcripts, never a planted-mistake script.
    """
    excluded = set(excluded) | {r.get("speaker") for r in records if r.get("split") != "train"}
    admitted, seen = [], set()
    for r in records:
        if not r.get("speaker") or r["speaker"] in excluded or r.get("split") != "train":
            continue
        if r.get("exclude") or r.get("excluded"):
            continue
        if r.get("source") == "help-none":
            if (r.get("kind") != "none" or r.get("scored") is not True or
                    any(r.get(k) != [] for k in ("events", "non_errors", "unsupported"))):
                continue
        elif r.get("source") == "transcribed":
            if r.get("authorized") is not True or r.get("verified") is not True:
                raise ValueError("Recording lacks an authorized performed transcript")
        else:
            raise ValueError("Unknown training source")
        if not r.get("audioSha256") or not r.get("phones"):
            raise ValueError("Missing audio identity or complete transcript")
        if r["audioSha256"] in seen:
            raise ValueError("Duplicate audio can leak across speaker partitions")
        seen.add(r["audioSha256"])
        admitted.append(r)
    if {r["source"] for r in admitted} != {"help-none", "transcribed"}:
        raise ValueError("Need authorized transcripts and training-speaker none takes")
    return admitted


def recording_loss(head, views, *, dropout, ctc_weight, kd_weight, preserve_weight):
    """Complete CTC and ordinary-reading/pause preservation, equal in all arms."""
    if not views:
        raise ValueError("Need complete recording views")
    losses = []
    for v in views:
        x, target, shipped = v["features"].detach(), v["target"], v["shipped"].detach()
        if (x.ndim != 2 or shipped.shape != (len(x), head.out_features) or
                target.ndim != 1 or target.dtype != torch.int64 or not len(target) or
                (target <= 0).any() or (target >= shipped.shape[1]).any() or
                len(x) < len(target) + int((target[1:] == target[:-1]).sum())):
            raise ValueError("Invalid complete recording transcript or frame clock")
        lp = head(F.dropout(x, dropout, training=head.training)).log_softmax(-1).float()
        ctc = F.ctc_loss(lp[:, None], target, torch.tensor([len(lp)]),
                         torch.tensor([len(target)]), blank=0, reduction="sum",
                         zero_infinity=False) / len(lp)
        q = shipped.float().log_softmax(-1)
        preservation = (q.exp() * (q - lp)).sum(-1).mean()
        loss = ctc_weight * ctc + preserve_weight * preservation
        if kd_weight:
            teacher = v["teacher"].detach().float()
            if teacher.shape != lp.shape:
                raise ValueError("Teacher targets must share the complete student clock")
            loss = loss - kd_weight * (teacher.log_softmax(-1).exp() * lp).sum(-1).mean()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite recording objective")
        losses.append(loss)
    return torch.stack(losses).mean()


def matched_refit(linear, records, optimizer_factory, scheduler_factory, *, epochs,
                  seed, dropout=.1, ctc_weight=1., kd_weight=0., preserve_weight=1., emit=None):
    """One shared selected epoch, identical recordings and gradient evaluations.

    Factories supply the pinned ScaledAdam and Eden settings. All optimizers
    start fresh. RNG is reset per recording in each arm so dropout masks match.
    No gradient is evaluated for selection speakers. Epoch zero is eligible.
    """
    if epochs < 1 or not 0 <= dropout < 1:
        raise ValueError("Invalid bounded refit schedule")
    weights = (ctc_weight, kd_weight, preserve_weight)
    if any(not math.isfinite(w) or w < 0 for w in weights) or not ctc_weight:
        raise ValueError("Invalid matched objective weights")
    groups = speaker_partition([r["speaker"] for r in records])
    fit = [r for r in records if groups[r["speaker"]] == "fit"]
    selection = [r for r in records if groups[r["speaker"]] == "selection"]
    for rows in (fit, selection):
        if {r["source"] for r in rows} != {"help-none", "transcribed"}:
            raise ValueError("Both speaker partitions need both training sources")
    heads = ordinary_controls(linear, seed)
    heads["contrast"] = ContrastHead(linear)
    objective = dict(dropout=dropout, ctc_weight=ctc_weight,
                     kd_weight=kd_weight, preserve_weight=preserve_weight)
    # Check all cached views, not just a synthetic vector. A dequantized head
    # that fails shipped-probability parity must not start this experiment.
    worst = {}
    with torch.no_grad():
        for arm, head in heads.items():
            worst[arm] = 0.
            for r in records:
                for v in r["views"]:
                    got = head(v["features"]).softmax(-1)
                    want = v["shipped"].softmax(-1)
                    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
                    worst[arm] = max(worst[arm], float((got - want).abs().max()))
    optimizers = {a: optimizer_factory(h.named_parameters()) for a, h in heads.items()}
    schedulers = {a: scheduler_factory(o) for a, o in optimizers.items()}

    def validate(head):
        head.eval()
        speakers = {}
        with torch.no_grad():
            for r in selection:
                value = float(recording_loss(head, r["views"], **objective))
                speakers.setdefault(r["speaker"], []).append(value)
        return sum(sum(v) / len(v) for v in speakers.values()) / len(speakers)

    history = [{"epoch": 0, "objectives": {a: validate(h) for a, h in heads.items()}}]
    best_score = sum(history[0]["objectives"].values()) / len(heads)
    best_epoch, best_states = 0, {a: copy.deepcopy(h.state_dict()) for a, h in heads.items()}
    evaluations = {a: 0 for a in heads}
    for epoch in range(1, epochs + 1):
        order = list(range(len(fit)))
        random.Random(seed + epoch).shuffle(order)
        for arm, head in heads.items():
            head.train()
            schedulers[arm].step_epoch(epoch - 1)
            for index in order:
                torch.manual_seed(seed + evaluations[arm])
                optimizers[arm].zero_grad(set_to_none=True)
                loss = recording_loss(head, fit[index]["views"], **objective)
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                    raise ValueError("Non-finite applied head gradient")
                optimizers[arm].step()
                evaluations[arm] += 1
                schedulers[arm].step_batch(evaluations[arm])
        metrics = {a: validate(h) for a, h in heads.items()}
        history.append(dict(epoch=epoch, objectives=metrics))
        if emit:
            emit(history[-1])
        score = sum(metrics.values()) / len(heads)
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_states = {a: copy.deepcopy(h.state_dict()) for a, h in heads.items()}
    for arm, head in heads.items():
        head.load_state_dict(best_states[arm])
        head.eval()
    return heads, dict(history=history, groups=groups, selectedEpoch=best_epoch,
                      selectedUpdates=best_epoch * len(fit), gradientEvaluations=evaluations,
                      initializationMaxProbabilityError=worst,
                      selectionRule="shared-epoch-minimum-equal-arm-mean-equal-speaker-objective")

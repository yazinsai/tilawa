"""Class-local ScaledAdam units for a matched, frozen-encoder phone-head refit.

Only parameter boundaries change. The classifier remains one affine projection,
including blank; the objective, gradient coordinates and global clipping rule
are shared with the tensor-grouped control. No practice evidence is read here.
"""

import copy
import hashlib
import math
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("tensor", "class_local")


class ClassLocalHead(nn.Module):
    """One Parameter per class weight row and one per scalar class bias.

    All parameters belong to ONE ScaledAdam group. Its leading batching axis
    then indexes classes, so its existing RMS reduction excludes other rows.
    Biases use the pinned optimizer's ordinary scalar-parameter rule. Building
    the affine tensors with stack preserves every gradient coordinate.
    """

    def __init__(self, linear):
        super().__init__()
        if not isinstance(linear, nn.Linear) or linear.bias is None:
            raise ValueError("Need the complete existing affine phone head")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.rows = nn.ParameterList([nn.Parameter(w.detach().clone()) for w in linear.weight])
        self.biases = nn.ParameterList([nn.Parameter(b.detach().clone()) for b in linear.bias])

    def forward(self, features):
        return F.linear(features, torch.stack(tuple(self.rows)), torch.stack(tuple(self.biases)))

    def folded(self):
        """Return the ordinary classifier used by the unmodified SDK export."""
        result = nn.Linear(self.in_features, self.out_features, device=self.rows[0].device,
                           dtype=self.rows[0].dtype)
        with torch.no_grad():
            result.weight.copy_(torch.stack(tuple(self.rows)))
            result.bias.copy_(torch.stack(tuple(self.biases)))
        return result


def speaker_partition(speakers):
    if any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Every recording needs an identified training speaker")
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(unique) < 5:
        raise ValueError("Need at least five training speakers")
    return {s: "selection" if i % 5 == 0 else "fit" for i, s in enumerate(unique)}


def admit_recordings(records, excluded):
    """Exclude non-training speakers globally before reading enrolled audio."""
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


def recording_loss(head, views, *, dropout, ctc_weight, kd_weight, kd_delay=0):
    """The shipped frame-normalized CTC and top-five KD on complete recordings.

    Ordinary speech and real pause frames are retained in every view. There is
    no new loss term or supervision partition. Teacher frame mapping and top-5
    renormalization match student_train.losses and student.soft_target_loss.
    """
    if not views:
        raise ValueError("Need complete recording views")
    values = []
    for v in views:
        x, target = v["features"].detach(), v["target"]
        if (x.ndim != 2 or not len(x) or target.ndim != 1 or target.dtype != torch.int64 or
                not len(target) or (target <= 0).any() or (target >= head.out_features).any() or
                len(x) < len(target) + int((target[1:] == target[:-1]).sum())):
            raise ValueError("Invalid complete recording transcript or frame clock")
        lp = head(F.dropout(x, dropout, training=head.training)).log_softmax(-1).float()
        loss = ctc_weight * F.ctc_loss(lp[:, None], target, torch.tensor([len(lp)]),
            torch.tensor([len(target)]), blank=0, reduction="sum", zero_infinity=True) / len(lp)
        if kd_weight:
            teacher = v["teacher"].detach().float()
            if (teacher.ndim != 2 or not len(teacher) or teacher.shape[1] != head.out_features or
                    not torch.isfinite(teacher).all()):
                raise ValueError("Invalid complete teacher distribution")
            idx = (torch.arange(len(lp)) - kd_delay).clamp(min=0, max=len(teacher) - 1)
            probs, ids = teacher.log_softmax(-1).exp()[idx].topk(min(5, head.out_features), -1)
            probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-6)
            loss = loss - kd_weight * (lp.gather(-1, ids) * probs).sum(-1).mean()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite recording objective")
        values.append(loss)
    return torch.stack(values).mean()


def matched_refit(linear, records, optimizer_factory, scheduler_factory, *, epochs,
                  seed, dropout=.1, ctc_weight=1., kd_weight=0., kd_delay=0,
                  deadline=None, emit=None):
    """Two arms share data, updates, dropout draws and a training-only epoch.

    The tensor control selects one update count for both arms, including epoch
    zero. Fresh optimizers start with zero per-coordinate moments. The factory
    must supply pinned ScaledAdam with one whole-head group and clipping scale 2.
    """
    if (epochs < 1 or not 0 <= dropout < 1 or
            any(not math.isfinite(w) or w < 0 for w in (ctc_weight, kd_weight)) or
            not ctc_weight or not isinstance(kd_delay, int)):
        raise ValueError("Invalid matched refit configuration")
    def check_deadline():
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Bounded refit deadline exceeded")
    groups = speaker_partition([r["speaker"] for r in records])
    fit = [r for r in records if groups[r["speaker"]] == "fit"]
    selection = [r for r in records if groups[r["speaker"]] == "selection"]
    for rows in (fit, selection):
        if {r["source"] for r in rows} != {"help-none", "transcribed"}:
            raise ValueError("Both speaker partitions need both training sources")
    heads = {"tensor": copy.deepcopy(linear), "class_local": ClassLocalHead(linear)}
    objective = dict(dropout=dropout, ctc_weight=ctc_weight, kd_weight=kd_weight, kd_delay=kd_delay)
    worst = {}
    with torch.no_grad():
        for arm, head in heads.items():
            worst[arm] = 0.
            for r in records:
                check_deadline()
                for v in r["views"]:
                    got = head(v["features"]).softmax(-1)
                    want = v["shipped"].softmax(-1)
                    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
                    worst[arm] = max(worst[arm], float((got - want).abs().max()))
    optimizers = {a: optimizer_factory(h.named_parameters()) for a, h in heads.items()}
    for opt in optimizers.values():
        if len(opt.param_groups) != 1 or opt.param_groups[0].get("clipping_scale") != 2.:
            raise ValueError("Need one globally clipped whole-head ScaledAdam group")
    schedulers = {a: scheduler_factory(o) for a, o in optimizers.items()}

    def validate(head):
        head.eval()
        speakers = {}
        with torch.no_grad():
            for r in selection:
                check_deadline()
                value = float(recording_loss(head, r["views"], **objective))
                speakers.setdefault(r["speaker"], []).append(value)
        return sum(sum(v) / len(v) for v in speakers.values()) / len(speakers)

    history = [{"epoch": 0, "objectives": {a: validate(h) for a, h in heads.items()}}]
    best_score, best_epoch = history[0]["objectives"]["tensor"], 0
    best_states = {a: copy.deepcopy(h.state_dict()) for a, h in heads.items()}
    evaluations = {a: 0 for a in heads}
    for epoch in range(1, epochs + 1):
        order = list(range(len(fit)))
        random.Random(seed + epoch).shuffle(order)
        for arm, head in heads.items():
            head.train()
            schedulers[arm].step_epoch(epoch - 1)
            for index in order:
                check_deadline()
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
        if metrics["tensor"] < best_score:
            best_score, best_epoch = metrics["tensor"], epoch
            best_states = {a: copy.deepcopy(h.state_dict()) for a, h in heads.items()}
    for arm, head in heads.items():
        head.load_state_dict(best_states[arm])
        head.eval()
    return heads, dict(history=history, groups=groups, selectedEpoch=best_epoch,
        selectedUpdates=best_epoch * len(fit), gradientEvaluations=evaluations,
        initializationMaxProbabilityError=worst,
        selectionRule="shared-epoch-minimum-tensor-control-equal-speaker-objective")

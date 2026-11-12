"""Finite-view SVRG for the existing frozen-encoder affine phone head.

Uniform recording sampling preserves the ordinary continuation's objective.
The reference and current gradient use identical cached views, including fixed
head-dropout masks. Compute is counted in complete recording-loss gradients,
not optimizer steps: a reference pass costs N and a paired update costs two.
"""
import copy
import hashlib
import math
import random
import time

import numpy as np


def speaker_partition(speakers):
    if any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Every recording needs an identified training speaker")
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(unique) < 5:
        raise ValueError("Need at least five identified training speakers")
    return {s: "selection" if i % 5 == 0 else "fit" for i, s in enumerate(unique)}


def admitted_none(labels, excluded=()):
    excluded = set(excluded) | {r.get("speaker") for r in labels if r.get("split") != "train"}
    selected, seen = [], set()
    for row in labels:
        if row.get("split") != "train" or row.get("kind") != "none":
            continue
        if not row.get("speaker") or row["speaker"] in excluded:
            continue
        if (row.get("scored") is not True or any(row.get(k) != [] for k in
                ("events", "non_errors", "unsupported")) or row.get("excluded") or row.get("exclude")):
            continue
        if not row.get("id") or not row.get("wav") or not row.get("ayahs"):
            raise ValueError("None take lacks identity, audio or transcript")
        if row["id"] in seen:
            raise ValueError("Duplicate none take")
        seen.add(row["id"])
        selected.append(row)
    return selected


def phone_head(weight, bias):
    import torch
    weight, bias = np.asarray(weight), np.asarray(bias)
    if weight.ndim != 2 or bias.shape != (len(weight),) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Invalid native linear head")
    head = torch.nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        head.weight.copy_(torch.tensor(weight.copy()))
        head.bias.copy_(torch.tensor(bias.copy()))
    return head


def log_probs(head, features):
    return head(features.detach()).log_softmax(-1)


def objectives(predicted, target, teacher, shipped):
    import torch
    from torch.nn import functional as F
    if (predicted.ndim != 2 or target.ndim != 1 or target.dtype != torch.int64 or
            teacher.ndim != 2 or teacher.shape[1] != predicted.shape[1] or
            not 0 < len(teacher) <= len(predicted) or shipped.shape != predicted.shape):
        raise ValueError("Complete transcript, teacher and student clocks must agree")
    if (not len(target) or (target <= 0).any() or (target >= predicted.shape[1]).any() or
            len(predicted) < len(target) + int((target[1:] == target[:-1]).sum())):
        raise ValueError("Impossible complete transcript alignment")
    if any(not torch.isfinite(x).all() for x in (predicted, teacher, shipped)):
        raise ValueError("Non-finite recording evidence")
    # Keep the shipped trainer's valid-frame CTC units, including real pauses.
    ctc = F.ctc_loss(predicted[:, None], target,
        torch.tensor([len(predicted)]), torch.tensor([len(target)]),
        blank=0, reduction="sum", zero_infinity=False) / len(predicted)
    kd = -(teacher.detach().log_softmax(-1).exp() * predicted[:len(teacher)]).sum(-1).mean()
    q = shipped.detach().log_softmax(-1)
    preserve = (q.exp() * (q - predicted)).sum(-1).mean()
    return ctc, kd, preserve


def recording_loss(head, views):
    import torch
    if not views:
        raise ValueError("Need nonempty fixed recording views")
    losses = [sum(objectives(log_probs(head, v["fitFeatures"]), v["target"],
                            v["teacher"], v["shipped"])) for v in views]
    return torch.stack(losses).mean()


def gradient(head, row):
    import torch
    loss = recording_loss(head, row["views"])
    result = torch.autograd.grad(loss, tuple(head.parameters()))
    if any(not torch.isfinite(g).all() for g in result):
        raise ValueError("Non-finite head gradient")
    return tuple(g.detach() for g in result)


def full_gradient(head, rows):
    import torch
    if not rows:
        raise ValueError("Reference gradient needs fit recordings")
    result = [torch.zeros_like(p) for p in head.parameters()]
    for row in rows:
        for total, value in zip(result, gradient(head, row)):
            total.add_(value / len(rows))
    return tuple(result)


def paired_gradient(current, reference, row, reference_mean):
    a, b = gradient(current, row), gradient(reference, row)
    return tuple(x - y + mean for x, y, mean in zip(a, b, reference_mean))


def apply(head, direction, lr):
    import torch
    if len(direction) != len(tuple(head.parameters())):
        raise ValueError("Incomplete head update")
    with torch.no_grad():
        for p, g in zip(head.parameters(), direction):
            if p.shape != g.shape or not torch.isfinite(g).all():
                raise ValueError("Invalid head update")
            p.sub_(lr * g)
            if not torch.isfinite(p).all():
                raise ValueError("Non-finite updated head")


def matched_refit(initial, rows, groups, validate, refresh_multiples=(1, 4),
                  budget_blocks=4, lr=.02, seed=1729, checkpoint=None, emit=None,
                  ordinary_only=False, wall_seconds=23 * 3600):
    """Select refresh/checkpoint on whole held-aside TRAINING speakers only.

    All paths sample uniformly with replacement from the same finite cache.
    A complete SVRG cycle costs N + 2*m*N gradient evaluations. The LCM of
    cycle costs defines shared compute checkpoints with no discarded work.
    Ordinary SGD spends every evaluation on an update; its intermediate heads
    are retained for the independent acoustic premise audit, never selection.
    """
    if (not refresh_multiples or any(type(m) is not int or m < 1 for m in refresh_multiples) or
            len(set(refresh_multiples)) != len(refresh_multiples) or
            type(budget_blocks) is not int or budget_blocks < 1 or not math.isfinite(lr) or lr <= 0):
        raise ValueError("Invalid compute-matched schedule")
    if not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 24 * 3600:
        raise ValueError("Refit must stay within the one-day wall budget")
    deadline = time.monotonic() + wall_seconds
    if set(groups.values()) != {"fit", "selection"} or any(r["speaker"] not in groups for r in rows):
        raise ValueError("Need identified fit and selection speakers")
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    if not fit or not selection:
        raise ValueError("Need separate nonempty training partitions")
    n = len(fit)
    unit = math.lcm(*(n * (1 + 2 * m) for m in refresh_multiples))
    budget = unit * budget_blocks
    arms = ("ordinary",) if ordinary_only else ("ordinary", *(str(m) for m in refresh_multiples))
    heads = {arm: copy.deepcopy(initial) for arm in arms}
    history, states, updates = {}, {}, {}
    for arm, head in heads.items():
        rng = random.Random(seed)
        used = steps = 0
        history[arm] = [dict(evaluations=0, updates=0, **validate(head, selection))]
        states[arm] = {0: copy.deepcopy(head.state_dict())}
        if checkpoint:
            checkpoint(arm, 0, head)
        def record():
            if arm == "ordinary" and checkpoint and used % n == 0:
                checkpoint(arm, used, head)
            if used % unit == 0:
                metrics = dict(evaluations=used, updates=steps, **validate(head, selection))
                history[arm].append(metrics)
                states[arm][used] = copy.deepcopy(head.state_dict())
                if emit:
                    emit(dict(arm=arm, **metrics))
        while used < budget:
            if time.monotonic() >= deadline:
                raise TimeoutError("One-day refit budget exhausted; no selected bundle is complete")
            if arm == "ordinary":
                apply(head, gradient(head, fit[rng.randrange(n)]), lr)
                used += 1
                steps += 1
                record()
            else:
                reference = copy.deepcopy(head)
                mean = full_gradient(reference, fit)
                used += n
                for _ in range(int(arm) * n):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("One-day refit budget exhausted; no selected bundle is complete")
                    apply(head, paired_gradient(head, reference, fit[rng.randrange(n)], mean), lr)
                    used += 2
                    steps += 1
                    record()
        updates[arm] = steps
    if ordinary_only:
        return heads, dict(history=history, gradientEvaluationsPerArm=budget,
                           updates=updates, premiseOnly=True)
    # Include initialization, so preservation can select no change.
    choices = [(float((a["objective"] + b["objective"]) / 2), a["evaluations"], m)
               for m in refresh_multiples for a, b in zip(history["ordinary"], history[str(m)])]
    score, selected_budget, multiple = min(choices)
    selected = {}
    for arm, key in (("ordinary", "ordinary"), ("variance", str(multiple))):
        selected[arm] = copy.deepcopy(initial)
        selected[arm].load_state_dict(states[key][selected_budget])
        selected[arm].eval()
    return selected, dict(history=history, selectedEvaluations=selected_budget,
        gradientEvaluationsPerArm=budget, refreshRecordings=multiple * n,
        selectedRefreshMultiple=multiple, updates=updates, selectionObjective=score,
        selectionRule="minimum-pair-mean-objective-at-equal-gradient-compute-on-selection-speakers")


def consonant_instability(heads, features, teacher, planted, baseline):
    """Audit annotated consonant spans on ONE fixed teacher-supported frame.

    `planted` supplies (start frame, exclusive end, phone ID) occurrences from
    diagnostic labels. It cannot enter training or checkpoint selection.
    Positive/negative crossings must occur across continuation checkpoints;
    baseline absence and teacher support are prerequisites, not guessed labels.
    """
    import torch
    if len(heads) < 2 or teacher.shape != baseline.shape or len(features) != len(teacher):
        raise ValueError("Need matched diagnostic frames and continuation checkpoints")
    if (features.ndim != 2 or teacher.ndim != 2 or not len(features) or
            any(not torch.isfinite(v).all() for v in (features, teacher, baseline))):
        raise ValueError("Invalid diagnostic acoustic evidence")
    results = []
    with torch.no_grad():
        for start, end, phone in planted:
            if not 0 <= start < end <= len(features) or not 0 < phone < teacher.shape[1]:
                raise ValueError("Invalid planted consonant occurrence")
            absent = bool((baseline[start:end].argmax(-1) != phone).all())
            supported = teacher[start:end].argmax(-1) == phone
            if not absent or not bool(supported.any()):
                results.append(dict(missing=absent, teacherSupported=bool(supported.any()), changesSign=False))
                continue
            competitors = teacher[start:end].clone()
            competitors[:, phone] = -torch.inf
            strength = teacher[start:end, phone] - competitors.max(-1).values
            strength[~supported] = -torch.inf
            frame = start + int(strength.argmax())
            margins = []
            for head in heads:
                logits = head(features[frame:frame + 1].detach())[0]
                other = logits.clone()
                other[phone] = -torch.inf
                margins.append(float(logits[phone] - other.max()))
            results.append(dict(missing=True, teacherSupported=True, frame=frame, margins=margins,
                changesSign=min(margins) < 0 < max(margins)))
    return dict(occurrences=results, changesSign=any(r["changesSign"] for r in results))

"""Unconditional CTC-collapse counts and matched frozen-encoder head refits.

Frame labels are independent draws from the CTC distribution. A nonblank
label starts an occurrence at frame zero, or when the preceding frame differs
(including blank). No transcript-conditioned alignment enters this statistic.
Only training-speaker evidence enters the refit or checkpoint selection.
"""
import copy
import hashlib
import math
import random

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


def reference_articulations(phones):
    """Convert ref-v2 held notation using the SDK's collapseHeld semantics.

    Madd/ikhfa runs are one occurrence. Duration-expanded mushaddad noon/meem
    is two, as is a real doubled consonant. Collapsing ALL adjacent labels
    would erase gemination and give the auxiliary loss incorrect counts.
    Performed CTC transcripts already express occurrences; do not apply this
    conversion to them.
    """
    out, i = [], 0
    while i < len(phones):
        j = i + 1
        while j < len(phones) and phones[j] == phones[i]:
            j += 1
        run = j - i
        keep = 1 if phones[i] in "اۥۦں۾" else 2 if phones[i] in "نم" and run >= 3 else run
        out.append(phones[i] * keep)
        i = j
    return "".join(out)


def expected_counts(probabilities, lengths=None, blank=0):
    """Return (..., V-1) expected occurrences, excluding blank and padding.

    Accept (T,V) or (B,T,V). Lengths are valid frame counts, not transcript
    lengths. The first valid frame has an implicit preceding blank. A zero
    length example returns zero counts without breaking the gradient graph.
    """
    import torch
    p = probabilities
    if p.ndim not in (2, 3) or not p.is_floating_point() or not 0 <= blank < p.shape[-1]:
        raise ValueError("Need categorical frame probabilities and a valid blank")
    single = p.ndim == 2
    p = p[None] if single else p
    if not torch.isfinite(p).all() or (p < 0).any() or not torch.allclose(
            p.sum(-1), torch.ones_like(p[..., 0]), atol=1e-6, rtol=1e-6):
        raise ValueError("Invalid categorical frame probabilities")
    if lengths is None:
        lengths = torch.full((len(p),), p.shape[1], dtype=torch.int64, device=p.device)
    else:
        lengths = torch.as_tensor(lengths, device=p.device)
        if single and lengths.ndim == 0:
            lengths = lengths[None]
    if (lengths.shape != (len(p),) or lengths.dtype not in (torch.int32, torch.int64) or
            (lengths < 0).any() or (lengths > p.shape[1]).any()):
        raise ValueError("Invalid valid-frame lengths")
    phones = torch.cat((p[..., :blank], p[..., blank + 1:]), -1)
    previous = torch.cat((torch.zeros_like(phones[:, :1]), phones[:, :-1]), 1)
    starts = phones * (1 - previous)
    valid = torch.arange(p.shape[1], device=p.device)[None] < lengths[:, None]
    counts = (starts * valid[..., None]).sum(1)
    return counts[0] if single else counts


def transcript_counts(target, vocabulary, blank=0):
    """Count articulated labels, retaining real repetitions in CTC targets.

    Duration-expanded reference notation must be converted at enrollment, not
    inferred here: [k,k] means two articulations and needs a separating blank.
    """
    import torch
    if (target.ndim != 1 or target.dtype != torch.int64 or vocabulary < 2 or
            not 0 <= blank < vocabulary or (target < 0).any() or
            (target >= vocabulary).any() or (target == blank).any()):
        raise ValueError("Invalid articulated transcript")
    counts = torch.bincount(target, minlength=vocabulary)
    return torch.cat((counts[:blank], counts[blank + 1:])).to(torch.float32)


def count_loss(predicted, target):
    """Mean count error, scaled by target count + 1; absent phones also count.

    The scale prevents frequent labels dominating, while zero-target silence
    still penalizes every expected phone occurrence. No duration normalization
    or transcript forward/backward probabilities are used.
    """
    wanted = transcript_counts(target, predicted.shape[-1]).to(predicted)
    got = expected_counts(predicted.exp())
    return ((got - wanted).square() / (wanted + 1)).mean()


def phone_head(weight, bias):
    import torch
    weight, bias = np.asarray(weight), np.asarray(bias)
    if (weight.ndim != 2 or bias.shape != (len(weight),) or
            not np.isfinite(weight).all() or not np.isfinite(bias).all()):
        raise ValueError("Invalid native linear head")
    head = torch.nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        head.weight.copy_(torch.tensor(weight.copy()))
        head.bias.copy_(torch.tensor(bias.copy()))
    return head


def log_probs(head, features):
    return head(features.detach()).log_softmax(-1)


def objectives(predicted, target, shipped):
    import torch
    from torch.nn import functional as F
    transcript_counts(target, predicted.shape[-1])
    if (predicted.ndim != 2 or shipped.shape != predicted.shape or not len(predicted) or
            len(predicted) < len(target) + int((target[1:] == target[:-1]).sum())):
        raise ValueError("Impossible complete transcript alignment or mismatched clocks")
    if not torch.isfinite(predicted).all() or not torch.isfinite(shipped).all():
        raise ValueError("Non-finite recording evidence")
    # Empty targets are actual silence: CTC is exactly the all-blank path.
    ctc = F.ctc_loss(predicted[:, None], target,
        torch.tensor([len(predicted)]), torch.tensor([len(target)]), blank=0,
        reduction="sum", zero_infinity=False) / max(1, len(target))
    if not len(target):
        ctc = ctc / len(predicted)  # equal recording weight, independent of silence duration
    q = shipped.detach().log_softmax(-1)
    preserve = (q.exp() * (q - predicted)).sum(-1).mean()
    return ctc, count_loss(predicted, target), preserve


def recording_loss(head, views, strength):
    import torch
    if not views or not math.isfinite(strength) or strength < 0:
        raise ValueError("Need fixed recording views and nonnegative count strength")
    losses = [objectives(log_probs(head, v["features"]), v["target"], v["shipped"]) for v in views]
    ctc, count, preserve = [torch.stack([x[i] for x in losses]).mean() for i in range(3)]
    return ctc + strength * count + preserve, dict(ctc=float(ctc.detach()),
        count=float(count.detach()), preservation=float(preserve.detach()))


def matched_refit(initial, rows, groups, validate, strengths, epochs, lr, seed, emit=None):
    """Select count pressure and one shared checkpoint on selection speakers.

    Each arm gets the same initialization, fixed views, order, SGD, KL pull,
    gradient bound and update count; ordinary differs only by count strength 0.
    A common validation objective compares both arms at the same epoch. Epoch
    zero is eligible, so selection can honestly reject every trained head.
    """
    import torch
    if (epochs < 1 or lr <= 0 or not math.isfinite(lr) or not strengths or
            len(set(strengths)) != len(strengths) or
            any(not math.isfinite(s) or s <= 0 for s in strengths)):
        raise ValueError("Invalid matched training schedule")
    if any(groups.get(r["speaker"]) not in ("fit", "selection") for r in rows):
        raise ValueError("Unknown training-speaker partition")
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    if not fit or not selection:
        raise ValueError("Need separate fit and selection speakers")
    arms = {"ordinary": 0., **{str(s): s for s in strengths}}
    heads = {arm: copy.deepcopy(initial) for arm in arms}
    history = {arm: [dict(epoch=0, **validate(head, selection))] for arm, head in heads.items()}
    best_states = {arm: copy.deepcopy(head.state_dict()) for arm, head in heads.items()}
    chosen_epoch, chosen_strength = 0, str(strengths[0])
    best_score = history["ordinary"][0]["objective"]
    for epoch in range(1, epochs + 1):
        order = list(range(len(fit)))
        random.Random(seed + epoch).shuffle(order)
        for arm, head in heads.items():
            diagnostics = []
            for i in order:
                head.zero_grad(set_to_none=True)
                loss, diag = recording_loss(head, fit[i]["views"], arms[arm])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
                if any(p.grad is None for p in head.parameters()):
                    raise ValueError("Missing applied head gradient")
                with torch.no_grad():
                    for p in head.parameters():
                        p.sub_(lr * p.grad)
                diagnostics.append(diag)
            metrics = dict(epoch=epoch, **validate(head, selection), recordings=len(order),
                losses={k: float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]})
            history[arm].append(metrics)
            if emit:
                emit(dict(arm=arm, **metrics))
        for strength in map(str, strengths):
            score = (history["ordinary"][-1]["objective"] + history[strength][-1]["objective"]) / 2
            if score < best_score:
                best_score, chosen_epoch, chosen_strength = score, epoch, strength
                best_states = {arm: copy.deepcopy(head.state_dict()) for arm, head in heads.items()}
    selected = {}
    for arm, key in (("ordinary", "ordinary"), ("count", chosen_strength)):
        selected[arm] = copy.deepcopy(initial)
        selected[arm].load_state_dict(best_states[key])
        selected[arm].eval()
    return selected, dict(history=history, selectedEpoch=chosen_epoch,
        countStrength=float(chosen_strength), selectionObjective=float(best_score),
        selectedUpdates=chosen_epoch * len(fit), trainingUpdates=epochs * len(fit),
        selectionRule="minimum-pair-mean-common-objective-at-shared-epoch-on-selection-speakers")

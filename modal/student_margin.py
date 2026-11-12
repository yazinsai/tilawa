"""Exact max-path CTC margin and matched frozen-encoder phone-head refits.

The loss-augmented competitor receives unit cost iff its collapsed transcript
differs. This bounded binary sequence cost makes the highest-scoring incorrect
frame path the competitor, including incomplete correct prefixes. No beam,
transcript forward sum, or vocabulary shortlist enters the path comparison.
"""
import copy
import hashlib
import math
import random
import time

import numpy as np

# Fixed before training or practice inspection; reject oversized recordings
# rather than silently truncate transcripts or approximate the path search.
MAX_FRAMES, MAX_TARGET = 1500, 512


def best_paths(scores, target, blank=0):
    """Return (correct, incorrect) maximizing complete frame-label scores.

    Prefix states alternate blank and the last emitted transcript label.
    A mismatch enters an absorbing reject state; arbitrary future labels
    cannot repair an already incorrect transcription. At stream end, every
    incomplete prefix also rejects. At most 2*len(target)+2 states are needed.
    Time is O(T*U*V), backpointer memory O(T*U), with fixed T and U bounds.
    """
    scores, target = np.asarray(scores), np.asarray(target)
    if (scores.ndim != 2 or not np.issubdtype(scores.dtype, np.floating) or
            not 0 < len(scores) <= MAX_FRAMES or scores.shape[1] < 2 or
            not np.isfinite(scores).all() or not 0 <= blank < scores.shape[1]):
        raise ValueError("Invalid or oversized frame scores")
    if (target.ndim != 1 or not np.issubdtype(target.dtype, np.integer) or
            len(target) > MAX_TARGET or np.any(target == blank) or
            np.any(target < 0) or np.any(target >= scores.shape[1])):
        raise ValueError("Invalid or oversized transcript")
    if len(scores) < len(target) + np.count_nonzero(target[1:] == target[:-1]):
        raise ValueError("Complete transcript has no CTC frame path")
    T, V = scores.shape
    S = 2 * len(target) + 1
    labels = np.full(S, blank, dtype=np.int64)
    labels[1::2] = target
    incoming = np.full((S, 3), S, dtype=np.int64)
    incoming[:, 0] = np.arange(S)
    incoming[1:, 1] = np.arange(S - 1)
    for s in range(3, S, 2):
        if labels[s] != labels[s - 2]:
            incoming[s, 2] = s - 2
    # A nonblank label is safe only as a same-label continuation or the next
    # transcript occurrence. Blank-separated equal labels are NEW occurrences.
    next_label = np.full(S, -1, dtype=np.int64)
    for s in range(S):
        emitted = (s + 1) // 2
        if emitted < len(target):
            next_label[s] = target[emitted]
    vocabulary = np.arange(V)[None]
    mismatch = ((vocabulary != blank) & (vocabulary != labels[:, None]) &
                (vocabulary != next_label[:, None]))
    parents = np.empty((T, S + 1), dtype=np.int32)
    emissions = np.empty((T, S + 1), dtype=np.int32)
    previous = np.full(S + 1, -np.inf)
    previous[0] = 0.
    for t, frame in enumerate(scores):
        # The sentinel S used for missing good predecessors must remain -inf,
        # independently of the finite score in the actual rejecting state.
        good_previous = np.append(previous[:S], -np.inf)
        options = good_previous[incoming]
        which = options.argmax(1)
        parents[t, :S] = incoming[np.arange(S), which]
        emissions[t, :S] = labels
        current = np.empty(S + 1)
        current[:S] = options[np.arange(S), which] + frame[labels]
        illegal = np.where(mismatch, frame[None], -np.inf)
        bad_label = illegal.argmax(1)
        bad_score = previous[:S] + illegal[np.arange(S), bad_label]
        greedy = int(frame.argmax())
        candidates = np.append(bad_score, previous[S] + frame[greedy])
        source = int(candidates.argmax())
        current[S] = candidates[source]
        parents[t, S] = source
        emissions[t, S] = greedy if source == S else bad_label[source]
        previous = current
    accepts = [S - 1] if S == 1 else [S - 2, S - 1]
    correct_end = max(accepts, key=lambda s: previous[s])
    rejects = [s for s in range(S + 1) if s not in accepts]
    incorrect_end = max(rejects, key=lambda s: previous[s])
    def traceback(state):
        if not np.isfinite(previous[state]):
            raise ValueError("No finite complete path")
        path = np.empty(T, dtype=np.int64)
        for t in range(T - 1, -1, -1):
            path[t] = emissions[t, state]
            state = int(parents[t, state])
        if state != 0:
            raise AssertionError("Path does not originate at the initial blank")
        return path
    return traceback(correct_end), traceback(incorrect_end)


def path_margin(predicted, target, blank=0):
    """Unit loss-augmented structured hinge, with exact max-path subgradients.

    Search is detached. Gather scores from the live tensor for the selected
    complete paths; common softmax normalization cancels. One denominator
    scales all transcript losses, with real-silence loss scaled by frame count.
    Padding never enters the input and therefore receives no gradient.
    """
    import torch
    good, bad = best_paths(predicted.detach().cpu().numpy(), target.detach().cpu().numpy(), blank)
    frames = torch.arange(len(predicted), device=predicted.device)
    good = torch.as_tensor(good, device=predicted.device)
    bad = torch.as_tensor(bad, device=predicted.device)
    difference = (predicted[frames, bad] - predicted[frames, good]).sum()
    return (1. + difference).clamp_min(0.) / max(1, len(target) if len(target) else len(predicted))


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
    """Convert SDK duration notation, preserving articulated gemination."""
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
    margin = path_margin(predicted, target)
    if shipped.shape != predicted.shape or not torch.isfinite(shipped).all():
        raise ValueError("Mismatched or non-finite preservation evidence")
    denominator = max(1, len(target) if len(target) else len(predicted))
    ctc = F.ctc_loss(predicted[:, None], target,
        torch.tensor([len(predicted)]), torch.tensor([len(target)]), blank=0,
        reduction="sum", zero_infinity=False) / denominator
    q = shipped.detach().log_softmax(-1)
    preserve = (q.exp() * (q - predicted)).sum(-1).mean()
    return ctc, margin, preserve


def recording_loss(head, views, strength):
    import torch
    if not views or not math.isfinite(strength) or strength < 0:
        raise ValueError("Need fixed recording views and nonnegative margin strength")
    # Evaluate the SAME path DP and losses even in the ordinary arm. Its zero
    # coefficient disconnects the extra gradient, giving matched compute.
    losses = [objectives(log_probs(head, v["features"]), v["target"], v["shipped"]) for v in views]
    ctc, margin, preserve = [torch.stack([x[i] for x in losses]).mean() for i in range(3)]
    return ctc + strength * margin + preserve, dict(ctc=float(ctc.detach()),
        margin=float(margin.detach()), preservation=float(preserve.detach()))


def matched_refit(initial, rows, groups, validate, strength, epochs, lr, seed, emit=None, deadline=None):
    """Select one shared epoch for two identical-data, matched-compute arms.

    Both compute CTC, path-margin and preservation losses on identical views
    and recording order, starting from identical native affine parameters.
    Only the margin coefficient differs. Epoch zero can reject all updates.
    """
    import torch
    def check_budget():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("One-day matched-refit budget exhausted; no selected pair is exported")
    check_budget()
    if (epochs < 1 or not math.isfinite(lr) or lr <= 0 or
            not math.isfinite(strength) or strength <= 0):
        raise ValueError("Invalid matched training schedule")
    if any(groups.get(r["speaker"]) not in ("fit", "selection") for r in rows):
        raise ValueError("Unknown training-speaker partition")
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    if not fit or not selection:
        raise ValueError("Need separate fit and selection speakers")
    arms = {"ordinary": 0., "margin": strength}
    heads = {arm: copy.deepcopy(initial) for arm in arms}
    history = {arm: [dict(epoch=0, **validate(head, selection))] for arm, head in heads.items()}
    best_states = {arm: copy.deepcopy(head.state_dict()) for arm, head in heads.items()}
    chosen_epoch = 0
    best_score = sum(history[a][0]["objective"] for a in arms) / 2
    if not math.isfinite(best_score):
        raise ValueError("Non-finite initial selection objective")
    for epoch in range(1, epochs + 1):
        order = list(range(len(fit)))
        random.Random(seed + epoch).shuffle(order)
        for arm, head in heads.items():
            diagnostics = []
            for i in order:
                check_budget()
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
        check_budget()
        score = sum(history[a][-1]["objective"] for a in arms) / 2
        if not math.isfinite(score):
            raise ValueError("Non-finite selection objective")
        if score < best_score:
            best_score, chosen_epoch = score, epoch
            best_states = {arm: copy.deepcopy(head.state_dict()) for arm, head in heads.items()}
    for arm, head in heads.items():
        head.load_state_dict(best_states[arm])
        head.eval()
    return heads, dict(history=history, selectedEpoch=chosen_epoch,
        marginStrength=strength, selectionObjective=float(best_score),
        selectedUpdates=chosen_epoch * len(fit), trainingUpdates=epochs * len(fit),
        selectionRule="minimum-pair-mean-common-objective-at-shared-epoch-on-selection-speakers")

"""Bounded, reference-free CTC transcription distributions for phone-head refits.

Proposal beams are approximate; every retained transcription is rescored by the
exact CTC forward sum. The complementary OTHER bucket keeps discarded mass in
the training objective rather than renormalizing a tiny candidate set to one.
No transcript, lexicon or language model participates in candidate generation.
"""
import hashlib
import math


def validate_rows(rows):
    """Reject logits or partial-alphabet scores passed as probabilities."""
    vocab = len(rows[0]) if len(rows) else 0
    if not vocab:
        raise ValueError("Need nonempty full-alphabet posterior rows")
    for row in rows:
        if (len(row) != vocab or any(not math.isfinite(float(p)) for p in row)
                or abs(logadd(*map(float, row))) > 1e-5):
            raise ValueError("Need normalized full-alphabet log probabilities")


def logadd(*values):
    high = max(values, default=-math.inf)
    return high if high == -math.inf else high + math.log(sum(math.exp(v - high) for v in values))


def collapse(path):
    return tuple(p for i, p in enumerate(path) if p and (not i or p != path[i - 1]))


def ctc_score(rows, labels):
    """Log P(complete transcription), summing every valid frame path once."""
    labels = tuple(labels)
    vocab = len(rows[0]) if len(rows) else max(labels, default=0) + 1
    if any(not isinstance(p, int) or not 0 < p < vocab for p in labels):
        raise ValueError("Transcription must contain inventory IDs excluding blank")
    states = (0,) + tuple(x for p in labels for x in (p, 0))
    alpha = [0.] + [-math.inf] * (len(states) - 1)
    for row in rows:
        if len(row) != vocab:
            raise ValueError("Changing posterior alphabet")
        following = []
        for i, p in enumerate(states):
            incoming = [alpha[i]]
            if i:
                incoming.append(alpha[i - 1])
            if i > 1 and p and p != states[i - 2]:
                incoming.append(alpha[i - 2])
            following.append(logadd(*incoming) + float(row[p]))
        alpha = following
    return logadd(*alpha[-2:]) if labels else alpha[0]


def transcriptions(rows, beam=16, retain=16):
    """Deterministic prefix beam, followed by exact rescoring and a fixed cap.

    Blank/nonblank prefix states remain separate: equal labels can extend a
    transcription only after blank. All output classes are visited each frame.
    """
    if beam < 1 or not 1 <= retain <= beam:
        raise ValueError("Need 1 <= retain <= beam")
    validate_rows(rows)
    active = {(): (0., -math.inf)}
    for row in rows:
        next_prefixes = {}

        def add(prefix, blank=-math.inf, phone=-math.inf):
            b, p = next_prefixes.get(prefix, (-math.inf, -math.inf))
            next_prefixes[prefix] = (logadd(b, blank), logadd(p, phone))

        for prefix, (b, p) in active.items():
            add(prefix, blank=logadd(b, p) + float(row[0]))
            for label in range(1, len(row)):
                emission = float(row[label])
                if prefix and label == prefix[-1]:
                    add(prefix, phone=p + emission)
                    add(prefix + (label,), phone=b + emission)
                else:
                    add(prefix + (label,), phone=logadd(b, p) + emission)
        active = dict(sorted(next_prefixes.items(), key=lambda x: (-logadd(*x[1]), x[0]))[:beam])
    # Greedy is an acoustic-only safeguard against a beam's intermediate pruning.
    greedy = collapse([max(range(len(row)), key=lambda p: row[p]) for row in rows])
    candidates = set(active) | {greedy}
    return tuple(sorted(candidates, key=lambda p: (-ctc_score(rows, p), p))[:retain])


def candidate_union(teacher, student, beam=16, retain=16):
    if not len(teacher) or not len(student) or len(teacher[0]) != len(student[0]):
        raise ValueError("Teacher and student must share the output alphabet")
    return tuple(sorted(set(transcriptions(teacher, beam, retain)) |
                        set(transcriptions(student, beam, retain))))


def speaker_groups(speakers, excluded=()):
    """Stable training-only fit/selection partition, with whole-speaker exclusion."""
    speakers = set(speakers)
    if not speakers or any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Every recording needs a speaker identity")
    if speakers & set(excluded):
        raise ValueError("Excluded speaker in training cache")
    return {s: "selection" if int(hashlib.sha256(s.encode()).hexdigest(), 16) % 5 == 0 else "fit"
            for s in sorted(speakers)}


def sequence_probabilities(log_probs, candidates):
    """Differentiable exact CTC scores plus the mass outside the bounded union.

    CPU float64 arithmetic is used for this small head objective. Candidate
    lengths and each model's frame count can differ; their emission timing is
    never aligned for sequence supervision.
    """
    import torch
    import torch.nn.functional as F
    candidates = tuple(tuple(p) for p in candidates)
    if len(set(candidates)) != len(candidates) or not candidates:
        raise ValueError("Need a nonempty set of distinct transcriptions")
    lp = log_probs.double()
    if lp.ndim != 2 or not len(lp) or not torch.isfinite(lp).all():
        raise ValueError("Need finite, full-vocabulary valid-frame posteriors")
    if not torch.allclose(lp.logsumexp(-1), torch.zeros(len(lp), dtype=lp.dtype), atol=1e-5, rtol=0):
        raise ValueError("Need normalized full-alphabet log probabilities")
    feasible = [i for i, p in enumerate(candidates)
                if len(p) + sum(a == b for a, b in zip(p, p[1:])) <= len(lp)]
    if any(any(not isinstance(p, int) or not 0 < p < lp.shape[1] for p in seq) for seq in candidates):
        raise ValueError("Candidate ID differs from the output alphabet")
    probs = lp.new_zeros(len(candidates))
    if feasible:
        labels = torch.tensor([p for i in feasible for p in candidates[i]], dtype=torch.long)
        lengths = torch.tensor([len(candidates[i]) for i in feasible], dtype=torch.long)
        scores = -F.ctc_loss(lp[:, None, :].expand(-1, len(feasible), -1).contiguous(),
            labels, torch.full((len(feasible),), len(lp), dtype=torch.long), lengths,
            blank=0, reduction="none", zero_infinity=False)
        probs = probs.index_copy(0, torch.tensor(feasible), scores.exp())
    # Only protect roundoff at unit total mass; no probability floor is placed
    # on any retained transcription, and discarded mass is not copied to blank.
    other = (1. - probs.sum()).clamp_min(torch.finfo(lp.dtype).eps)
    result = torch.cat((probs, other[None]))
    return result / result.sum()


def sequence_disagreement(student, teacher, candidates):
    import torch
    q = sequence_probabilities(teacher.detach(), candidates).detach()
    p = sequence_probabilities(student, candidates)
    positive = q > 0
    return (q[positive] * (q[positive].log() - p[positive].clamp_min(torch.finfo(p.dtype).tiny).log())).sum()


def head_objective(head, item, arm, candidates):
    """Identical transcript/preservation objectives; only the KD unit changes."""
    import torch
    import torch.nn.functional as F
    lp = head(item["features"].detach()).log_softmax(-1)
    target = item["target"]
    if (not len(target) or len(target) + int((target[1:] == target[:-1]).sum()) > len(lp)
            or bool((target <= 0).any()) or bool((target >= lp.shape[1]).any())):
        raise ValueError("Complete transcript cannot align to native output frames")
    ctc = F.ctc_loss(lp[:, None, :], target, torch.tensor([len(lp)]),
        torch.tensor([len(target)]), reduction="sum", zero_infinity=False) / len(lp)
    original = item["shipped"].detach()
    if original.shape != lp.shape or item["teacher"].ndim != 2 or item["teacher"].shape[1] != lp.shape[1]:
        raise ValueError("Cached evidence must retain the installed output alphabet and native clock")
    preserve = F.kl_div(lp, original.exp(), reduction="sum") / len(lp)
    teacher = item["teacher"].detach()
    if arm == "sequence":
        kd = sequence_disagreement(lp, teacher, candidates) / len(lp)
    elif arm == "framewise":
        # Match the shipped clamped native-clock teacher mapping in the control.
        t = teacher[torch.arange(len(lp)).clamp_max(len(teacher) - 1)]
        kd = F.kl_div(lp, t.exp(), reduction="sum") / len(lp)
    else:
        raise ValueError("Unknown distillation arm")
    return ctc + preserve + kd


def admitted_none(rows, excluded=()):
    """Enrollment is deliberately stricter than the help scorer's none label."""
    excluded = set(excluded) | {r.get("speaker") for r in rows if r.get("split") != "train"}
    result, seen = [], set()
    for r in rows:
        if (r.get("split") != "train" or r.get("speaker") in excluded
                or r.get("kind") != "none" or r.get("scored") is not True
                or r.get("excluded") or r.get("events") != []
                or r.get("non_errors") != [] or r.get("unsupported") != []):
            continue
        if not all(r.get(k) for k in ("id", "speaker", "wav", "ayahs")):
            raise ValueError("Incomplete none take enrollment")
        if r["id"] in seen:
            raise ValueError("Duplicate none take")
        seen.add(r["id"])
        result.append(r)
    return result


def phone_head(weight, bias):
    import torch
    head = torch.nn.Linear(len(weight[0]), len(weight))
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(weight))
        head.bias.copy_(torch.as_tensor(bias))
    return head


def matched_refit(initial, rows, groups, validate, beam, retain, epochs=12,
                  learning_rate=.02, seed=1729, emit=lambda report: None,
                  cpu_seconds=86400):
    """One order, initialization, data and selected update count for both arms.

    Candidates are regenerated from the current detached student, never from
    transcripts. Selection receives only disjoint training-speaker recordings
    and applies the same complete-sequence objective to both arms.
    """
    import copy
    import random
    import time
    import torch
    started = time.process_time()
    def budget():
        if time.process_time() - started >= cpu_seconds:
            raise TimeoutError("One-day CPU budget exhausted before paired validation")
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    if not fit or not selection:
        raise ValueError("Need nonempty whole-speaker fit and selection groups")
    heads = {arm: copy.deepcopy(initial) for arm in ("sequence", "framewise")}
    optimizers = {arm: torch.optim.SGD(h.parameters(), lr=learning_rate) for arm, h in heads.items()}
    history, selected, best, updates = [], None, None, 0
    rng = random.Random(seed)
    for epoch in range(epochs + 1):
        budget()
        if epoch:
            order = rng.sample(fit, len(fit))
            for r in order:
                budget()
                for arm, head in heads.items():
                    optimizers[arm].zero_grad()
                    losses = []
                    for view in r["views"]:
                        with torch.no_grad():
                            lp = head(view["features"].detach()).log_softmax(-1)
                        candidates = candidate_union(view["teacher"].tolist(), lp.tolist(), beam, retain)
                        losses.append(head_objective(head, view, arm, candidates))
                    loss = torch.stack(losses).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite matched refit loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
                    optimizers[arm].step()
                updates += 1
        metrics = {arm: validate(h, selection, beam, retain) for arm, h in heads.items()}
        key = (sum(m["per"] for m in metrics.values()), sum(m["objective"] for m in metrics.values()), epoch)
        if best is None or key < best:
            best, selected = key, copy.deepcopy(heads)
            chosen_epoch, chosen_updates = epoch, updates
        entry = dict(epoch=epoch, updates=updates, arms=metrics)
        history.append(entry)
        emit(entry)
    budget()
    return selected, dict(history=history, selectedEpoch=chosen_epoch,
        selectedUpdates=chosen_updates, trainingUpdates=updates, beam=beam, retain=retain)

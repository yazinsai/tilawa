"""Bounded, reference-free CTC transcription distributions for phone-head refits.

Proposal beams are approximate; every retained transcription is rescored by the
exact CTC forward sum. Conditional expected edit distance is accompanied by exact complementary mass,
full-risk bounds and a coverage cost. The bounded conditional objective is an
approximation, not the full unbounded transcription risk.
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
            if blank == phone == -math.inf:
                return
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


def candidate_union(baseline, student, beam=16, retain=16):
    if not len(baseline) or not len(student) or len(baseline[0]) != len(student[0]):
        raise ValueError("Baseline and student must share the output alphabet")
    return tuple(sorted(set(transcriptions(baseline, beam, retain)) |
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


def sequence_log_probabilities(log_probs, candidates):
    """Exact differentiable CTC forward sums, stable even for tiny lattice mass.

    A transcription is one event, regardless of how many proposal paths collapse
    to it. Impossible transcriptions have log probability -inf. No candidate or
    omitted probability receives an artificial floor.
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
    if any(any(not isinstance(p, int) or not 0 < p < lp.shape[1] for p in seq) for seq in candidates):
        raise ValueError("Candidate ID differs from the output alphabet")
    feasible = [i for i, p in enumerate(candidates)
                if len(p) + sum(a == b for a, b in zip(p, p[1:])) <= len(lp)]
    scores = lp.new_full((len(candidates),), -float("inf"))
    if feasible:
        labels = torch.tensor([p for i in feasible for p in candidates[i]], dtype=torch.long)
        lengths = torch.tensor([len(candidates[i]) for i in feasible], dtype=torch.long)
        values = -F.ctc_loss(lp[:, None, :].expand(-1, len(feasible), -1).contiguous(),
            labels, torch.full((len(feasible),), len(lp), dtype=torch.long), lengths,
            blank=0, reduction="none", zero_infinity=False)
        scores = scores.index_copy(0, torch.tensor(feasible), values)
    return scores


def levenshtein(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def edit_risk(log_probs, candidates, target):
    """Conditional expected edit distance and explicit complementary mass.

    Candidate generation never sees `target`. Exact CTC scores are normalized
    within their retained set for graded edit credit. The outside distribution
    is unknown: report its mass and rigorous lower/upper full-risk bounds,
    rather than claiming the conditional risk is the full expectation.
    A coverage cost (-log retained mass per frame) discourages improving the
    conditional objective merely by moving probability outside the lattice.
    """
    import torch
    target = tuple(target)
    if any(not isinstance(p, int) or not 0 < p < log_probs.shape[-1] for p in target):
        raise ValueError("Reference must use the installed nonblank inventory")
    scores = sequence_log_probabilities(log_probs, candidates)
    log_mass = scores.logsumexp(0)
    if not torch.isfinite(log_mass):
        raise ValueError("Lattice contains no feasible transcription")
    if log_mass.detach() > 1e-5:
        raise ValueError("Disjoint CTC transcription mass exceeds one")
    # Scores may have tiny roundoff above log(1); only the accounting is capped.
    mass = log_mass.clamp_max(0).exp()
    other = -torch.expm1(log_mass.clamp_max(0))
    costs = scores.new_tensor([levenshtein(p, target) for p in candidates])
    conditional = (torch.softmax(scores, 0) * costs).sum() / max(1, len(target))
    lower = mass * conditional
    upper = lower + other * max(len(log_probs), len(target)) / max(1, len(target))
    coverage = -log_mass.clamp_max(0) / len(log_probs)
    return dict(conditional=conditional, retainedMass=mass, otherMass=other,
                lower=lower, upper=upper, coverage=coverage,
                objective=conditional + coverage)


def head_objective(head, item, arm, candidates=()):
    """Same full transcript CTC and ordinary-output preservation in both arms."""
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
    if original.shape != lp.shape:
        raise ValueError("Preservation evidence must keep the installed inventory and clock")
    preserve = F.kl_div(lp, original.exp(), reduction="sum") / len(lp)
    if arm == "risk":
        auxiliary = edit_risk(lp, candidates, tuple(target.tolist()))["objective"]
    elif arm == "ctc":
        auxiliary = lp.new_zeros(())
    else:
        raise ValueError("Unknown refit arm")
    return ctc + preserve + auxiliary


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
    and chooses a shared update count by paired speaker-balanced greedy PER,
    then the shared ordinary CTC/preservation objective.
    """
    import copy
    import random
    import time
    import torch
    started, wall_started = time.process_time(), time.monotonic()
    def budget():
        if max(time.process_time() - started, time.monotonic() - wall_started) >= cpu_seconds:
            raise TimeoutError("One-day elapsed/CPU budget exhausted before paired validation")
    fit = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    if not fit or not selection:
        raise ValueError("Need nonempty whole-speaker fit and selection groups")
    heads = {arm: copy.deepcopy(initial) for arm in ("risk", "ctc")}
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
                        candidates = candidate_union(view["shipped"].tolist(), lp.tolist(), beam, retain)
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

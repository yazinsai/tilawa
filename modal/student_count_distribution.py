"""Unconditional occurrence distributions after ordinary CTC collapse.

For each requested nonblank class, the previous frame is either that class or
another class (including blank). Only entering from the latter increments the
count. The final bin holds ALL counts above max_count, without renormalizing
away zero or surplus paths. This is a training statistic, not a decoder rule.
"""


def _logadd(a, b):
    import torch
    unreachable = torch.isneginf(a) & torch.isneginf(b)
    # logaddexp(-inf, -inf) has undefined derivatives. Mask its operands too.
    return torch.logaddexp(a.masked_fill(unreachable, 0),
                          b.masked_fill(unreachable, 0)).masked_fill(unreachable, -torch.inf)


def collapsed_count_log_distribution(log_probs, classes, max_count):
    """Return (classes, max_count+2) log mass: 0..max_count, then overflow.

    log_probs is one recording's VALID (T,V) categorical CTC frames. The
    implicit previous frame at the left edge is blank. Passing no frames
    returns a point mass at zero. Requested classes must exclude blank id 0.
    The result sums over unrestricted paths, not transcript alignments.
    """
    import torch
    if (log_probs.ndim != 2 or not log_probs.is_floating_point() or
            not isinstance(max_count, int) or max_count < 0):
        raise ValueError('Need categorical log frames and a nonnegative count cap')
    classes = torch.as_tensor(classes, dtype=torch.int64, device=log_probs.device)
    if (classes.ndim != 1 or not len(classes) or classes.unique().numel() != len(classes) or
            (classes <= 0).any() or (classes >= log_probs.shape[1]).any()):
        raise ValueError('Need distinct nonblank class ids')
    if (torch.isnan(log_probs).any() or torch.isposinf(log_probs).any() or
            not torch.allclose(log_probs.logsumexp(-1), torch.zeros_like(log_probs[:, 0]),
                               atol=1e-5, rtol=1e-5)):
        raise ValueError('Invalid normalized categorical log probabilities')
    off = log_probs.new_full((len(classes), max_count + 2), -torch.inf)
    off[:, 0] = 0
    on = torch.full_like(off, -torch.inf)
    mask = torch.arange(log_probs.shape[1], device=log_probs.device)[None] == classes[:, None]
    for frame in log_probs:
        q = frame[classes, None]
        other = frame[None].expand(len(classes), -1).masked_fill(mask, -torch.inf).logsumexp(-1)[:, None]
        entered = torch.cat((torch.full_like(off[:, :1], -torch.inf), off[:, :-1]), -1)
        # Entering from overflow remains in overflow, as does entering from
        # the last exact count. Continuing a run never increments its count.
        entered = torch.cat((entered[:, :-1], _logadd(off[:, -2:-1], off[:, -1:])), -1)
        off, on = _logadd(off, on) + other, _logadd(on, entered) + q
    return _logadd(off, on)


def expected_collapsed_counts(log_probs, classes):
    """Exact first moment; overflow is not approximated by its lower bound."""
    import torch
    p = log_probs[:, classes].exp()
    previous = torch.cat((torch.zeros_like(p[:1]), p[:-1]), 0)
    return (p * (1 - previous)).sum(0)


def distribution_count_loss(log_probs, target, consonants):
    """Mean negative log probability of every consonant's transcript count.

    target contains performed articulations, retaining true gemination but
    with duration-expanded reference notation converted at enrollment. Absent
    consonants participate, so real pauses penalize surplus emissions too.
    Transcript CTC remains a separate, unchanged objective.
    """
    import torch
    if (target.ndim != 1 or target.dtype != torch.int64 or (target <= 0).any() or
            (target >= log_probs.shape[1]).any()):
        raise ValueError('Need an articulated nonblank transcript')
    classes = torch.as_tensor(consonants, dtype=torch.int64, device=log_probs.device)
    counts = torch.bincount(target, minlength=log_probs.shape[1])[classes]
    result = collapsed_count_log_distribution(log_probs, classes, int(counts.max()))
    return -result.gather(-1, counts[:, None]).mean()


def recording_loss(head, views, mode, strength, consonants):
    import torch
    from student_count import log_probs, objectives, transcript_counts
    if mode not in ('ordinary', 'expected', 'distribution') or not views or strength < 0:
        raise ValueError('Invalid matched recording objective')
    losses = []
    for v in views:
        lp = log_probs(head, v['features'])
        ctc, _, preserve = objectives(lp, v['target'], v['shipped'])
        wanted = transcript_counts(v['target'], lp.shape[-1]).to(lp)[
            torch.as_tensor(consonants, device=lp.device) - 1]
        expected = ((expected_collapsed_counts(lp, consonants) - wanted).square() / (wanted + 1)).mean()
        distribution = distribution_count_loss(lp, v['target'], consonants)
        # Every arm evaluates both statistics, including their zero-weighted
        # gradients, so ordinary CTC receives matched computational work too.
        losses.append(torch.stack((ctc, expected, distribution, preserve)))
    ctc, expected, distribution, preserve = torch.stack(losses).mean(0)
    total = ctc + preserve + (strength if mode == 'expected' else 0.) * expected + (
        strength if mode == 'distribution' else 0.) * distribution
    return total, {k: float(v.detach()) for k, v in zip(
        ('ctc', 'expected', 'distribution', 'preservation'), (ctc, expected, distribution, preserve))}


def matched_refit(initial, rows, groups, validate, strengths, epochs, lr, seed,
                  consonants, emit=None):
    """Fit three objective families, choosing a shared epoch on training speakers.

    At each epoch select each auxiliary family's regularization by the SAME
    ordinary CTC + shipped KL validation objective. Select the shared epoch by
    the mean of those two minima and the ordinary control. Epoch zero is
    eligible. All arms use identical cached views, update order and clipping.
    """
    import copy
    import math
    import random
    import time
    import torch
    if (epochs < 1 or not math.isfinite(lr) or lr <= 0 or not strengths or
            len(set(strengths)) != len(strengths) or
            any(not math.isfinite(s) or s <= 0 for s in strengths)):
        raise ValueError('Invalid matched training schedule')
    if any(groups.get(r['speaker']) not in ('fit', 'selection') for r in rows):
        raise ValueError('Unknown training-speaker partition')
    fit = [r for r in rows if groups[r['speaker']] == 'fit']
    selection = [r for r in rows if groups[r['speaker']] == 'selection']
    if not fit or not selection:
        raise ValueError('Need distinct fit and selection speakers')
    arms = {'ordinary': ('ordinary', 0.), **{
        f'{mode}:{s}': (mode, s) for mode in ('expected', 'distribution') for s in strengths}}
    heads = {name: copy.deepcopy(initial) for name in arms}
    history = {name: [] for name in arms}
    best, selected, chosen_epoch = math.inf, None, None
    started = time.monotonic()
    for epoch in range(epochs + 1):
        order = list(range(len(fit)))
        random.Random(seed + epoch).shuffle(order)
        for name, head in heads.items():
            if epoch:
                head.train()
                for i in order:
                    if time.monotonic() - started > 24 * 3600:
                        raise TimeoutError('Matched comparison exceeded one day; no head selected')
                    head.zero_grad(set_to_none=True)
                    loss, _ = recording_loss(head, fit[i]['views'], *arms[name], consonants)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
                    if any(p.grad is None for p in head.parameters()):
                        raise ValueError('Missing applied phone-head gradient')
                    with torch.no_grad():
                        for p in head.parameters():
                            p.sub_(lr * p.grad)
            head.eval()
            metrics = dict(epoch=epoch, **validate(head, selection))
            history[name].append(metrics)
            if emit:
                emit(dict(arm=name, **metrics))
        if time.monotonic() - started > 24 * 3600:
            raise TimeoutError('Matched comparison exceeded one day; no head selected')
        chosen = {'ordinary': 'ordinary'}
        for mode in ('expected', 'distribution'):
            chosen[mode] = min((name for name in arms if arms[name][0] == mode),
                key=lambda name: (history[name][-1]['objective'], arms[name][1]))
        score = sum(history[name][-1]['objective'] for name in chosen.values()) / 3
        if score < best:
            best, chosen_epoch = score, epoch
            selected = {mode: copy.deepcopy(heads[name]) for mode, name in chosen.items()}
            selected_strengths = {mode: arms[name][1] for mode, name in chosen.items()}
    return selected, dict(history=history, selectedEpoch=chosen_epoch,
        selectedStrengths=selected_strengths, selectionObjective=best,
        selectedUpdates=chosen_epoch * len(fit), trainingUpdates=epochs * len(fit),
        elapsedSeconds=time.monotonic() - started,
        selectionRule='minimum-three-arm-mean-CTC-plus-KL-at-shared-epoch-on-selection-speakers')

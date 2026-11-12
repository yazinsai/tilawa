"""Training-only CTC occupancies and recording-preserving batch grouping."""
from collections import Counter

import numpy as np


def ctc_occupancies(log_probs, target, blank=0):
    """P(frame phone | complete transcript, acoustics), including CTC blank.

    Repeated transcript labels require a blank between their articulation
    states. Forward/backward sums every legal placement; no hard word cuts or
    winning-frame labels become training supervision.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    if lp.ndim != 2 or not len(lp) or not np.isfinite(lp).all():
        raise ValueError('Invalid acoustic rows')
    if (target.ndim != 1 or np.any(target == blank) or
            np.any(target < 0) or np.any(target >= lp.shape[1])):
        raise ValueError('Invalid complete transcript')
    labels = np.full(2*len(target)+1, blank, dtype=np.int64)
    labels[1::2] = target
    emit = lp[:, labels]
    skip = np.zeros(len(labels), dtype=bool)
    skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    alpha = np.full_like(emit, -np.inf)
    alpha[0, :min(2, len(labels))] = emit[0, :min(2, len(labels))]
    for t in range(1, len(lp)):
        previous = alpha[t-1]
        one = np.r_[-np.inf, previous[:-1]]
        two = np.full(len(labels), -np.inf)
        two[2:] = previous[:-2]
        alpha[t] = emit[t] + np.logaddexp(np.logaddexp(previous, one), np.where(skip, two, -np.inf))
    z = np.logaddexp.reduce(alpha[-1, -2:])
    if not np.isfinite(z):
        raise ValueError('Transcript cannot be completely realized in acoustic rows')
    beta = np.full_like(emit, -np.inf)
    beta[-1, -min(2, len(labels)):] = 0.
    for t in range(len(lp)-2, -1, -1):
        following = beta[t+1] + emit[t+1]
        one = np.r_[following[1:], -np.inf]
        two = np.full(len(labels), -np.inf)
        two[:-2] = np.where(skip[2:], following[2:], -np.inf)
        beta[t] = np.logaddexp(np.logaddexp(following, one), two)
    state = np.exp(alpha+beta-z)
    occupied = np.zeros_like(lp)
    for s, label in enumerate(labels):
        occupied[:, label] += state[:, s]
    np.testing.assert_allclose(occupied.sum(1), 1., atol=1e-7, rtol=0)
    return occupied


def recording_descriptors(views):
    """Equal mean of cached views' soft phone occupancies per recording."""
    result = {}
    for key, rows in views.items():
        if not rows:
            raise ValueError('Recording has no acoustic views')
        result[key] = np.mean([ctc_occupancies(lp, target).mean(0) for lp, target in rows], axis=0)
    return result


def mine_confusion_pairs(training_decodes, consonants, limit):
    """Rank symmetric confusions from fit-speaker decodes, never validation.

    Each whole recording contributes its soft transcript occupancy against
    decoded frame winners with unit total weight. The geometric mean of both
    directed error rates requires observed mistakes in both directions.
    """
    consonants = sorted(set(consonants))
    table = None
    for lp, target in training_decodes:
        occupancy = ctc_occupancies(lp, target)
        winning = np.eye(lp.shape[1])[np.argmax(lp, axis=1)]
        update = occupancy.T @ winning / len(lp)
        table = update if table is None else table+update
    if table is None:
        raise ValueError('No fit-speaker decodes for mining')
    totals = table.sum(1)
    ranked = []
    for i, a in enumerate(consonants):
        for b in consonants[i+1:]:
            if totals[a] > 0 and totals[b] > 0:
                score = np.sqrt((table[a, b]/totals[a])*(table[b, a]/totals[b]))
                if score > 0:
                    ranked.append(dict(phones=[a, b], score=float(score),
                                       support=[float(totals[a]), float(totals[b])]))
    return sorted(ranked, key=lambda p: (-p['score'], p['phones']))[:limit]


def batch_quality(batches, descriptors, pairs):
    """Normalized mass balance; a one-sided active contrast has cost one."""
    if not pairs:
        return dict(imbalance=0., both_sides=0., active=0)
    keys = [k for batch in batches for k in batch]
    mean = np.mean([descriptors[k] for k in keys], axis=0)
    differences, covered = [], []
    for batch in batches:
        mass = np.sum([descriptors[k] for k in batch], axis=0)
        for a, b in pairs:
            x, y = mass[a]/mean[a], mass[b]/mean[b]
            if x+y:
                differences.append(abs(x-y)/(x+y))
                covered.append(x > 0 and y > 0)
    return dict(imbalance=float(np.mean(differences)) if differences else 0.,
                both_sides=float(np.mean(covered)) if covered else 0., active=len(covered))


def batch_plan(keys, lengths, descriptors, pairs, batch_size, seed, mode):
    """Permute complete recordings; never resample, drop, crop or duplicate.

    Greedy contrast grouping uses soft masses scaled by their training-population
    means, so common sounds cannot dominate every pairing. It favors having
    both alternatives in one update and then balances their normalized masses.
    Every arm has the same final partial batch size.
    """
    keys = list(keys)
    if len(set(keys)) != len(keys) or not keys or batch_size < 1:
        raise ValueError('Expected unique recordings and positive batch size')
    rng = np.random.default_rng(seed)
    order = [keys[i] for i in rng.permutation(len(keys))]
    if mode == 'length':
        order.sort(key=lambda k: lengths[k])  # random ties from the shared seed
    elif mode == 'balanced':
        if not pairs or batch_size < 2:
            raise ValueError('Contrast grouping requires pairs and at least two recordings')
        index = np.asarray(pairs, dtype=int)
        values = np.asarray([descriptors[k] for k in order])[:, index]
        means = values.mean(0)
        if np.any(means <= 0):
            raise ValueError('A mined contrast lacks an observed alternative')
        values /= means
        unused = np.ones(len(order), dtype=bool)
        grouped = []
        for start in range(0, len(order), batch_size):
            # Fixed random anchors prevent text/speaker sorting from defining
            # the update order. Candidate additions can only change grouping.
            first = int(np.flatnonzero(unused)[0])
            unused[first] = False
            batch, mass = [order[first]], values[first].copy()
            for _ in range(min(batch_size, len(order)-start)-1):
                candidates = np.flatnonzero(unused)
                total = values[candidates]+mass
                denominator = total.sum(-1)
                imbalance = np.divide(np.abs(total[..., 0]-total[..., 1]), denominator,
                                      out=np.zeros_like(denominator), where=denominator > 0)
                both = (total[..., 0] > 0) & (total[..., 1] > 0)
                # Coverage first, imbalance second; seeded order breaks ties.
                winner = min(range(len(candidates)), key=lambda i: (-int(both[i].sum()), float(imbalance[i].sum()), i))
                chosen = int(candidates[winner])
                unused[chosen] = False
                batch.append(order[chosen])
                mass += values[chosen]
            grouped.append(batch)
        batches = grouped
    elif mode != 'shuffled':
        raise ValueError('Unknown grouping procedure')
    if mode != 'balanced':
        batches = [order[i:i+batch_size] for i in range(0, len(order), batch_size)]
    # Shuffle full batches without changing the position of the partial update.
    full = len(keys)//batch_size
    batches[:full] = [batches[i] for i in rng.permutation(full)]
    verify_matched_plans(keys, {'candidate': batches}, batch_size)
    return batches


def verify_matched_plans(keys, plans, batch_size):
    expected = Counter(keys)
    sizes = [batch_size]*(len(keys)//batch_size)
    if len(keys) % batch_size:
        sizes.append(len(keys) % batch_size)
    for batches in plans.values():
        if Counter(k for batch in batches for k in batch) != expected:
            raise ValueError('Grouping changed the recording multiset')
        if [len(batch) for batch in batches] != sizes:
            raise ValueError('Grouping changed effective batch sizes or update counts')

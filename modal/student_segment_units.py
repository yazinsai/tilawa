"""Matched acoustic/forced segmentation codebooks and a marginalized phone-to-unit transducer.

The independent codebook receives only acoustic arrays. All eligible training takes enter
that stage, including mistakes and repeats. Canonical labels enter only the forced control
and the none-take transducer. Speaker-grouped validation never sees practice observations.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.cluster.vq import kmeans2

MAX_DURATION, UNITS, DIMENSIONS, PENALTY = 12, 32, 8, 4.0


def descriptor(x, start, end):
    positions = start + np.array([1/6, 1/2, 5/6]) * (end-start-1)
    lo, hi = np.floor(positions).astype(int), np.ceil(positions).astype(int)
    fraction = (positions-lo)[:, None]
    return (x[lo]*(1-fraction)+x[hi]*fraction).reshape(-1)


def descriptors(x):
    """All candidate durations in one batch; start-indexed arrays, exact normalized clock."""
    values = []
    for length in range(1, MAX_DURATION+1):
        starts = np.arange(max(0, len(x)-length+1))[:, None]
        if not len(starts):
            values.append(np.empty((0, 3*x.shape[1])))
            continue
        positions = starts + np.array([1/6, 1/2, 5/6])[None, :] * (length-1)
        lo, hi = np.floor(positions).astype(int), np.ceil(positions).astype(int)
        f = (positions-lo)[..., None]
        values.append((x[lo]*(1-f)+x[hi]*f).reshape(len(starts), -1))
    return values


def distances(x, centers):
    return np.maximum(0, (x*x).sum(1)[:, None]+(centers*centers).sum(1)[None, :]-2*x@centers.T)


def acoustic_segments(x, centers, variance):
    """Joint minimum-description-cost segmentation and unit assignments; no labels or visits."""
    desc = descriptors(x)
    dist = [distances(d, centers) for d in desc]
    costs = [d.min(1)*length/(2*variance)+PENALTY+np.log(MAX_DURATION)
             for length, d in enumerate(dist, 1)]
    best = np.full(len(x)+1, np.inf); best[0] = 0
    back = np.zeros(len(x)+1, dtype=int)
    for end in range(1, len(x)+1):
        possible = np.array([best[end-length]+costs[length-1][end-length]
                             for length in range(1, min(MAX_DURATION, end)+1)])
        back[end] = int(possible.argmin())+1; best[end] = possible.min()
    end, result = len(x), []
    while end:
        start = end-back[end]; result.append((start, end)); end = start
    return result[::-1]


def ctc_edges(ids):
    labels = np.array([0, *[v for p in ids for v in (p, 0)]])
    edges = [[s, *([s+1] if s+1 < len(labels) else []),
              *([s+2] if s+2 < len(labels) and labels[s+2] and labels[s+2] != p else [])]
             for s, p in enumerate(labels)]
    return labels, edges


def forced_segments(log_probs, ids):
    """Transcript-forced phone/blank placement, bounded with the same maximum duration."""
    labels, edges = ctc_edges(ids)
    parents = [[] for _ in labels]
    for s, outgoing in enumerate(edges):
        for n in outgoing: parents[n].append(s)
    a = np.full(len(labels), -np.inf); a[:2] = log_probs[0, labels[:2]]
    back = np.zeros((len(log_probs), len(labels)), dtype=np.int32)
    for t in range(1, len(log_probs)):
        b = np.full(len(labels), -np.inf)
        for s, p in enumerate(parents):
            winner = p[int(a[p].argmax())]; back[t, s] = winner
            b[s] = a[winner]+log_probs[t, labels[s]]
        a = b
    state = len(labels)-2+int(a[-1] > a[-2])
    if not np.isfinite(a[state]): raise ValueError('Incomplete forced training pronunciation')
    path = [state]
    for t in range(len(log_probs)-1, 0, -1):
        state = int(back[t, state]); path.append(state)
    path.reverse()
    cuts = [0, *[t for t in range(1, len(path)) if path[t] != path[t-1]], len(path)]
    result = []
    for start, end in zip(cuts, cuts[1:]):
        for lo in range(start, end, MAX_DURATION): result.append((lo, min(end, lo+MAX_DURATION)))
    return result


def fit_codebook(acoustics, initial, forced=None):
    centers, variance = initial.copy(), 1.0
    for iteration in range(5):
        observations, weights = [], []
        for i, x in enumerate(acoustics):
            boundaries = forced[i] if forced is not None else acoustic_segments(x, centers, variance)
            observations.extend(descriptor(x, lo, hi) for lo, hi in boundaries)
            weights.extend(hi-lo for lo, hi in boundaries)
        d, weight = np.array(observations), np.array(weights)
        assignment = distances(d, centers).argmin(1)
        for k in range(UNITS):
            mask = assignment == k
            if mask.any(): centers[k] = np.average(d[mask], axis=0, weights=weight[mask])
        residual = ((d-centers[assignment])**2).sum(1)
        variance = float(np.clip(np.average(residual, weights=weight)/d.shape[1], .25, 4))
        print(json.dumps(dict(stage='forced' if forced is not None else 'independent', iteration=iteration,
                              segments=len(d), variance=variance)), flush=True)
    return dict(centers=centers.tolist(), variance=variance)


def unit_observations(x, boundaries, arm):
    d = np.array([descriptor(x, lo, hi) for lo, hi in boundaries])
    logits = -distances(d, np.array(arm['centers']))/(2*arm['variance'])
    return np.exp(logits-logsumexp(logits, axis=1)[:, None]), np.array([hi-lo for lo, hi in boundaries])


def correspondence(emissions, ids):
    """Forward/backward monotone phone correspondence with normalized outgoing arcs."""
    labels, edges = ctc_edges(ids)
    outgoing = np.log([len(e) for e in edges])
    skip = np.array([bool(labels[s] and labels[s] != labels[s-2]) if s >= 2 else False for s in range(len(labels))])
    a = np.full((len(emissions), len(labels)), -np.inf)
    a[0, :2] = emissions[0, labels[:2]]
    for t in range(1, len(a)):
        source = a[t-1]-outgoing
        current = source.copy()
        current[1:] = np.logaddexp(current[1:], source[:-1])
        jump = np.full(len(labels), -np.inf); jump[2:] = source[:-2]
        current = np.logaddexp(current, np.where(skip, jump, -np.inf))
        a[t] = current+emissions[t, labels]
    total = np.logaddexp(a[-1, -1], a[-1, -2])
    if not np.isfinite(total): return None
    b = np.full_like(a, -np.inf); b[-1, -2:] = 0
    for t in range(len(a)-2, -1, -1):
        target = b[t+1]+emissions[t+1, labels]
        current = target.copy()
        current[:-1] = np.logaddexp(current[:-1], target[1:])
        jump = np.full(len(labels), -np.inf); jump[:-2] = np.where(skip[2:], target[2:], -np.inf)
        b[t] = np.logaddexp(current, jump)-outgoing
    gamma = np.exp(a+b-total)
    occupancy = np.zeros((len(a), emissions.shape[1]))
    for s, p in enumerate(labels): occupancy[:, p] += gamma[:, s]
    return occupancy


def fit_transducer(rows, x, arm, mode, tokens, shrinkage):
    vocab = len(tokens)+1
    cached = []
    counts = np.ones((vocab, UNITS))*shrinkage/UNITS
    centers = np.array(arm['centers'])
    for row, acoustic in zip(rows, x):
        if not row['canonical']: continue
        ids = [tokens.index(p)+1 for p in row['pronunciation']]
        boundaries = (forced_segments(row['log_probs'], ids) if mode == 'forced'
                      else acoustic_segments(acoustic, centers, arm['variance']))
        q, lengths = unit_observations(acoustic, boundaries, arm)
        # Soft transcript occupancies initialize both arms; no hard unit-to-phone labels.
        occupancy = correspondence(row['log_probs'], ids)
        if occupancy is None: continue
        for j, (lo, hi) in enumerate(boundaries): counts += occupancy[lo:hi].sum(0)[:, None]*q[j]
        # A learned unit may span more than one phone. Expand its observed duration,
        # not its acoustic boundaries, so phone correspondence can move inside it.
        # Otherwise recordings with fewer units than phones disappear from EM.
        expanded = np.repeat(q, lengths, axis=0)
        cached.append((ids, expanded, np.ones(len(expanded))))
    if not cached: raise ValueError('No complete canonical training correspondence')
    coverage = np.maximum(0, counts.sum(1)-shrinkage)
    theta = counts/counts.sum(1)[:, None]
    for _ in range(6):
        counts = np.ones_like(theta)*shrinkage/UNITS
        for ids, q, length in cached:
            probability = np.maximum(q@theta.T, 1e-300)
            occupancy = correspondence(length[:, None]*np.log(UNITS*probability), ids)
            if occupancy is None: raise ValueError('Incomplete canonical transducer correspondence')
            expected = occupancy*length[:, None]/probability
            counts += theta*(expected.T@q)
        theta = counts/counts.sum(1)[:, None]
    return {**arm, 'phones': theta.tolist(), 'counts': coverage.tolist()}


def score(arm, x, ids):
    """Same complete segmental marginal as the SDK (likelihood ratio to a uniform unit model)."""
    if not len(x) or not ids: return -np.inf
    labels, edges = ctc_edges(ids)
    theta, centers = np.array(arm['phones']), np.array(arm['centers'])
    desc = descriptors(x)
    emission, background = [], []
    for length, d in enumerate(desc, 1):
        logits = -distances(d, centers)/(2*arm['variance'])
        q = np.exp(logits-logsumexp(logits, axis=1)[:, None])
        bg = length*(logsumexp(logits, axis=1)-np.log(UNITS))
        background.append(bg)
        emission.append(length*np.log(np.maximum(UNITS*q@theta.T, 1e-300))+bg[:, None])
    a = np.full((len(x)+1, len(labels)), -np.inf); prior = a.copy()
    outgoing = np.log([len(e) for e in edges])
    skip = np.array([bool(labels[s] and labels[s] != labels[s-2]) if s >= 2 else False for s in range(len(labels))])
    cost = -PENALTY-np.log(MAX_DURATION)
    def advance(values):
        source = values-outgoing
        result = source.copy()
        result[1:] = np.logaddexp(result[1:], source[:-1])
        jump = np.full(len(labels), -np.inf); jump[2:] = source[:-2]
        return np.logaddexp(result, np.where(skip, jump, -np.inf))
    for start in range(len(x)):
        arrived = advance(a[start]) if start else None
        background_arrived = advance(prior[start]) if start else None
        for length in range(1, min(MAX_DURATION, len(x)-start)+1):
            e = emission[length-1][start, labels]; end = start+length
            timing_cost = cost+background[length-1][start]
            if start == 0:
                a[end, :2] = np.logaddexp(a[end, :2], cost+e[:2])
                prior[end, :2] = np.logaddexp(prior[end, :2], timing_cost)
            else:
                a[end] = np.logaddexp(a[end], arrived+cost+e)
                prior[end] = np.logaddexp(prior[end], background_arrived+timing_cost)
    likelihood, timing = np.logaddexp(a[-1, -1], a[-1, -2]), np.logaddexp(prior[-1, -1], prior[-1, -2])
    return float(likelihood-timing) if np.isfinite(likelihood) and np.isfinite(timing) else -np.inf


def load_rows(paths):
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r['source'] != 'help-train' or r['split'] != 'train' or not r['speaker']: raise ValueError('Unsafe training input')
            r['features'] = np.frombuffer(base64.b64decode(r.pop('featuresBase64')), dtype='<f4').reshape(r['frames'], r['width'])
            r['log_probs'] = np.array(r['log_probs'])
            rows.append(r)
    return rows


def fit(args):
    meta = json.loads(Path(args.metadata).read_text())
    rows = load_rows(args.rows)
    validation = {r['speaker'] for r in rows if int(hashlib.sha256(r['speaker'].encode()).hexdigest(), 16) % 5 == 0}
    train = [r for r in rows if r['speaker'] not in validation]
    valid = [r for r in rows if r['speaker'] in validation and r['canonical']]
    if not train or not valid: raise ValueError('Missing speaker-grouped canonical validation')
    # Only training partition acoustics determine normalization, projection and both codebooks.
    frames = np.concatenate([r['features'][::4] for r in train]).astype(np.float64)
    center, scale = frames.mean(0), np.maximum(frames.std(0), .1)
    standardized = (frames-center)/scale
    eigenvalues, vectors = np.linalg.eigh(standardized.T@standardized/len(standardized))
    projection = vectors[:, -DIMENSIONS:].T / np.sqrt(np.maximum(eigenvalues[-DIMENSIONS:], .1))[:, None]
    transform = lambda r: ((r['features']-center)/scale)@projection.T
    x = [transform(r) for r in train]
    rng = np.random.default_rng(1729)
    initial_descriptors = np.array([descriptor(a, start, min(len(a), start+4)) for a in x for start in range(0, len(a), 8)])
    initial, _ = kmeans2(initial_descriptors, UNITS, iter=15, minit='++', seed=rng)
    independent = fit_codebook(x, initial)  # Only acoustic arrays; no text/scripts or engine visits.
    forced_boundaries = [forced_segments(r['log_probs'], [meta['tokens'].index(p)+1 for p in r['pronunciation']]) for r in train]
    forced = fit_codebook(x, initial, forced_boundaries)
    selections = []
    for shrinkage in [1., 10.]:
        arms = {mode:fit_transducer(train, x, arm, mode, meta['tokens'], shrinkage)
                for mode, arm in [('independent', independent), ('forced', forced)]}
        losses = []
        for r in valid:
            acoustic, ids = transform(r), [meta['tokens'].index(p)+1 for p in r['pronunciation']]
            losses.extend(-score(arm, acoustic, ids)/len(acoustic) for arm in arms.values())
        loss = float(np.mean(losses)); selections.append((loss, shrinkage))
        print(json.dumps(dict(stage='validation', shrinkage=shrinkage, loss=loss)), flush=True)
    selected = min(selections)[1]
    # Freeze complexity and selected regularization, then refit with every eligible training speaker.
    all_x = [transform(r) for r in rows]
    independent = fit_codebook(all_x, np.array(independent['centers']))
    forced = fit_codebook(all_x, np.array(forced['centers']),
        [forced_segments(r['log_probs'], [meta['tokens'].index(p)+1 for p in r['pronunciation']]) for r in rows])
    model = {k:meta[k] for k in ['schema', 'inventory', 'tokens', 'width', 'featureOutput', 'primarySha256']}
    model.update(center=center.tolist(), scale=scale.tolist(), projection=projection.tolist(), maxDuration=MAX_DURATION, penalty=PENALTY,
        independent=fit_transducer(rows, all_x, independent, 'independent', meta['tokens'], selected),
        forced=fit_transducer(rows, all_x, forced, 'forced', meta['tokens'], selected))
    Path(args.out).write_text(json.dumps(model, allow_nan=False))
    audit = dict(recordings=len(rows), speakers=len({r['speaker'] for r in rows}), canonical=sum(r['canonical'] for r in rows),
        validationSpeakers=sorted(validation), validationCanonical=len(valid), selection=selections, selectedShrinkage=selected,
        units=UNITS, dimensions=DIMENSIONS, maxDuration=MAX_DURATION, penalty=PENALTY,
        inputs=[dict(id=r['id'], speaker=r['speaker'], canonical=r['canonical']) for r in rows])
    Path(args.out+'.audit.json').write_text(json.dumps(audit, allow_nan=False))
    print(json.dumps({k:v for k,v in audit.items() if k!='inputs'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rows', nargs='+', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--out', required=True)
    fit(p.parse_args())

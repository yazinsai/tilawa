"""CPU-bounded Fisher-score pronunciation readout and equal-capacity pooling control.

All fitting/selection uses eligible help-train training speakers. Each background
GMM is pronunciation independent. Both arms have identical diagonal-Gaussian
parameter counts and semi-Markov topology; no practice windows choose a model.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

DIMENSION = 6
READOUT = 24
MAX_DURATION = 64
CHECKPOINTS = (0, 1, 2)
MIXTURES = (2, 4)
STRENGTHS = (1., 16.)
CPU_LIMIT = 86400

try:
    from numba import njit
except ImportError:
    def njit(f):
        return f


import math
import numpy as np

@njit
def logadd(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = max(a, b)
    return m+math.log1p(math.exp(min(a, b)-m))


@njit
def boundary_kernel(values, labels, duration):
    tmax = values.shape[0]
    a = np.full((len(labels)+1, tmax+1), -np.inf)
    prior = np.full_like(a, -np.inf)
    a[0, 0] = prior[0, 0] = 0.
    for s in range(len(labels)):
        p = labels[s]
        minimum = 0 if p == 0 else 1
        cost = math.log(duration-minimum+1)
        for end in range(tmax+1):
            for n in range(minimum, min(duration, end)+1):
                v = 0. if n == 0 else values[end-1, n-1, p]
                a[s+1, end] = logadd(a[s+1, end], a[s, end-n]+v-cost)
                prior[s+1, end] = logadd(prior[s+1, end], prior[s, end-n]-cost)
    b = np.full_like(a, -np.inf)
    b[-1, -1] = 0.
    for s in range(len(labels)-1, -1, -1):
        p = labels[s]
        minimum = 0 if p == 0 else 1
        cost = math.log(duration-minimum+1)
        for start in range(tmax+1):
            for n in range(minimum, min(duration, tmax-start)+1):
                v = 0. if n == 0 else values[start+n-1, n-1, p]
                b[s, start] = logadd(b[s, start], v-cost+b[s+1, start+n])
    return a, b, prior[-1, -1]


def labels_for(ids):
    return np.asarray([0]+[p for v in ids for p in (v, 0)], dtype=np.int64)


def boundary(values, ids, posterior=False):
    labels = labels_for(ids)
    a, b, timing = boundary_kernel(values, labels, values.shape[1])
    z = a[-1, -1]
    score = float(z-timing) if np.isfinite(z) and np.isfinite(timing) else -np.inf
    if not posterior:
        return score
    if not np.isfinite(z):
        raise ValueError('Complete training pronunciation cannot fit acoustic clock')
    weights = np.zeros_like(values)
    # Sum repeated occurrences only after computing their distinct state posteriors.
    for s, p in enumerate(labels):
        for n in range(1, values.shape[1]+1):
            ends = np.arange(n, len(values)+1)
            cost = math.log(values.shape[1]+(p == 0))
            logw = a[s, ends-n]+values[ends-1, n-1, p]-cost+b[s+1, ends]-z
            weights[ends-1, n-1, p] += np.exp(logw)
    return score, weights


def background(rows, k):
    """PCA whitening and unsupervised diagonal mixture, fitted on this fold only."""
    from scipy.special import logsumexp
    raw = np.concatenate([r['raw'] for r in rows])
    center = raw.mean(0)
    _, singular, axes = np.linalg.svd(raw-center, full_matrices=False)
    project = axes[:DIMENSION]/np.maximum(singular[:DIMENSION, None]/math.sqrt(len(raw)), .01)
    x = (raw-center) @ project.T
    # Bounded deterministic subsample, independent of all labels/pronunciations.
    x = x[np.linspace(0, len(x)-1, min(50000, len(x)), dtype=int)]
    rng = np.random.default_rng(1729)
    mean = x[rng.choice(len(x), k, replace=False)].copy()
    variance = np.tile(np.maximum(x.var(0), .05), (k, 1))
    weight = np.full(k, 1./k)
    for _ in range(40):
        logq = np.log(weight)[None, :]-.5*(np.log(2*np.pi*variance)[None, :, :]+(x[:, None, :]-mean[None, :, :])**2/variance[None, :, :]).sum(2)
        q = np.exp(logq-logsumexp(logq, axis=1)[:, None])
        count = np.maximum(q.sum(0), 1e-8)
        mean = q.T @ x/count[:, None]
        variance = np.maximum(q.T @ (x*x)/count[:, None]-mean*mean, .05)
        weight = count/count.sum()
    return dict(center=center.tolist(), projection=project.tolist(), dimensions=READOUT,
        maxDuration=MAX_DURATION,
        mixture=[dict(weight=float(w), mean=m.tolist(), variance=v.tolist()) for w, m, v in zip(weight, mean, variance)],
        fisherProjection=(rng.normal(size=(READOUT, 2*k*DIMENSION))/math.sqrt(2*k*DIMENSION)).tolist(),
        controlProjection=(rng.normal(size=(READOUT, DIMENSION+DIMENSION*(DIMENSION+1)//2))/math.sqrt(DIMENSION+DIMENSION*(DIMENSION+1)//2)).tolist())


def frame_gradients(model, x):
    from scipy.special import logsumexp
    weight = np.array([g['weight'] for g in model['mixture']])
    mean = np.array([g['mean'] for g in model['mixture']])
    variance = np.array([g['variance'] for g in model['mixture']])
    u = (x[:, None, :]-mean[None, :, :])/np.sqrt(variance)[None, :, :]
    logq = np.log(weight)[None, :]-.5*(np.log(2*np.pi*variance)[None, :, :]+u*u).sum(2)
    q = np.exp(logq-logsumexp(logq, axis=1)[:, None])
    return np.concatenate([q[:, :, None]*u/np.sqrt(weight)[None, :, None],
        q[:, :, None]*(u*u-1)/np.sqrt(2*weight)[None, :, None]], axis=2).reshape(len(x), -1)


def normalized_projection(x, projection):
    y = x @ np.asarray(projection).T
    y = np.sign(y)*np.sqrt(np.abs(y))
    norm = np.linalg.norm(y, axis=-1, keepdims=True)
    return y/np.maximum(norm, 1e-12)


def segments(model, raw):
    """All candidate intervals, computed from prefix sufficient statistics."""
    x = (raw-model['center']) @ np.asarray(model['projection']).T
    g = frame_gradients(model, x)
    prefix = lambda v: np.concatenate([np.zeros_like(v[:1]), np.cumsum(v, axis=0)])
    px, pg = prefix(x), prefix(g)
    pairs = [(i, j) for i in range(DIMENSION) for j in range(i+1)]
    pc = prefix(np.column_stack([x[:, i]*x[:, j] for i, j in pairs]))
    a = np.zeros((len(x), MAX_DURATION, READOUT))
    b = np.zeros_like(a)
    for n in range(1, MAX_DURATION+1):
        ends = np.arange(n, len(x)+1)
        mean = (px[ends]-px[ends-n])/n
        covariance = (pc[ends]-pc[ends-n])/n-np.column_stack([mean[:, i]*mean[:, j] for i, j in pairs])
        a[ends-1, n-1] = normalized_projection((pg[ends]-pg[ends-n])/n, model['fisherProjection'])
        b[ends-1, n-1] = normalized_projection(np.concatenate([mean, covariance], axis=1), model['controlProjection'])
    return {'fisher':a, 'control':b}


def emissions(arm, descriptors):
    from scipy.special import logsumexp
    values = []
    for g in arm['phones']:
        variance = np.array(g['variance'])
        values.append(-.5*(np.log(2*np.pi*variance)+ (descriptors-g['mean'])**2/variance).sum(-1))
    v = np.stack(values, axis=-1)
    v = v-logsumexp(v, axis=-1, keepdims=True)+math.log(len(values))
    v *= np.arange(1, MAX_DURATION+1)[None, :, None]
    for end in range(len(v)):
        v[end, end+1:] = -np.inf
    return v


def initial_segments(log_probs, ids):
    """CTC Viterbi supplies starting phone order/centers, never final boundaries."""
    labels = labels_for(ids)
    scores = np.full((len(log_probs), len(labels)), -np.inf)
    back = np.zeros_like(scores, dtype=np.int64)
    scores[0, :2] = log_probs[0, labels[:2]]
    for t in range(1, len(log_probs)):
        for s, p in enumerate(labels):
            sources = [s]+([s-1] if s else [])+([s-2] if s>1 and p and p != labels[s-2] else [])
            previous = max(sources, key=lambda j:scores[t-1, j])
            scores[t, s] = scores[t-1, previous]+log_probs[t, p]
            back[t, s] = previous
    state = len(labels)-2+int(scores[-1, -1]>scores[-1, -2])
    if not np.isfinite(scores[-1, state]):
        raise ValueError('Unalignable none-labelled take')
    path = [state]
    for t in range(len(log_probs)-1, 0, -1):
        state = back[t, state]
        path.append(state)
    path = np.asarray(path[::-1])
    centers = [int(np.median(np.flatnonzero(path == s))) for s in range(1, len(labels), 2)]
    cuts = [0]+[(a+b)//2+1 for a, b in zip(centers, centers[1:])]+[len(path)]
    result = []
    # Conservatively reserve only outer parent blank runs for initial background examples.
    onset = int(np.flatnonzero(path == 1)[0])
    offset = int(np.flatnonzero(path == len(labels)-2)[-1])+1
    if onset:
        result.append((0, onset, 0))
        cuts[0] = onset
    if offset<len(path):
        result.append((offset, len(path), 0))
        cuts[-1] = offset
    for i, p in enumerate(ids):
        start, end = cuts[i], cuts[i+1]
        if end>start:
            result.append((start, end, p))
    return result


def update(count, sums, squares, strength):
    total = count.sum()
    if total <= 0:
        raise ValueError('No usable canonical acoustic intervals')
    global_mean = sums.sum(0)/total
    global_variance = np.maximum(squares.sum(0)/total-global_mean**2, .001)
    phones = []
    for p, n in enumerate(count):
        mean = (sums[p]+strength*global_mean)/(n+strength)
        variance = (squares[p]+strength*(global_variance+global_mean**2))/(n+strength)-mean**2
        phones.append(dict(mean=mean.tolist(), variance=np.maximum(variance, .001).tolist(), count=float(n)))
    return {'phones':phones}


def fit(rows, descriptors, key, vocab, strength, epoch=0, previous=None):
    count = np.zeros(vocab)
    sums = np.zeros((vocab, READOUT))
    squares = np.zeros_like(sums)
    for row, both in zip(rows, descriptors):
        x = both[key]
        if epoch == 0:
            for start, end, p in row['initial']:
                # Very long initial intervals are partitioned solely for initialization.
                for first in range(start, end, MAX_DURATION):
                    last = min(first+MAX_DURATION, end)
                    v = x[last-1, last-first-1]
                    count[p] += 1
                    sums[p] += v
                    squares[p] += v*v
        else:
            _, w = boundary(emissions(previous, x), row['ids'], True)
            count += w.sum((0, 1))
            sums += np.einsum('tnp,tnd->pd', w, x)
            squares += np.einsum('tnp,tnd->pd', w, x*x)
    return update(count, sums, squares, strength)


def validation(arm, held, descriptors, key, tokens):
    from scipy.special import logsumexp
    grouped = {}
    for row, both in zip(held, descriptors):
        values = emissions(arm, both[key])
        canonical = boundary(values, row['ids'])
        rivals = [boundary(values, [tokens.index(p)+1 for p in text]) for text in row['controlPronunciations']]
        if not rivals:
            continue
        loss = logsumexp([canonical]+rivals)-canonical
        grouped.setdefault(row['speaker'], []).append(float(loss))
    if not grouped:
        raise ValueError('No training-only pronunciation validation controls')
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def train(args):
    import resource
    # Linux supervisor: a hard per-process CPU-day ceiling also bounds BLAS work.
    used = math.ceil(time.process_time())
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    limit = used+CPU_LIMIT if hard == resource.RLIM_INFINITY else min(used+CPU_LIMIT, hard)
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
    started = time.process_time()
    metadata = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = []
    for line in Path(args.input).read_text().splitlines():
        r = json.loads(line)
        if r.get('source') != 'help-train' or r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Only eligible help-train training-speaker none takes may fit the scorer')
        r['ids'] = [metadata['tokens'].index(p)+1 for p in r['pronunciation']]
        for key, width in [('raw', metadata['width']), ('log_probs', len(metadata['tokens'])+1)]:
            r[key] = np.asarray(r[key], dtype=float)
            if r[key].ndim != 2 or r[key].shape[1] != width or not len(r[key]) or not np.isfinite(r[key]).all():
                raise ValueError('Invalid frozen acoustic observations')
        if len(r['raw']) != len(r['log_probs']):
            raise ValueError('Feature and primary clocks differ')
        r['initial'] = initial_segments(r['log_probs'], r['ids'])
        rows.append(r)
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers) < 3:
        raise ValueError('At least three independent eligible training speakers required')
    folds = {s:i%3 for i, s in enumerate(speakers)}
    results = {}
    def budget():
        if time.process_time()-started>CPU_LIMIT:
            raise TimeoutError('Fisher fitting exceeded its one-CPU-day bound')
    for k in MIXTURES:
        for fold in range(3):
            training = [r for r in rows if folds[r['speaker']] != fold]
            held = [r for r in rows if folds[r['speaker']] == fold]
            bg = background(training, k)
            descriptors = [segments(bg, r['raw']) for r in training]
            validation_descriptors = [segments(bg, r['raw']) for r in held]
            for strength in STRENGTHS:
                arms = {}
                for epoch in CHECKPOINTS:
                    budget()
                    for key in ('fisher', 'control'):
                        arms[key] = fit(training, descriptors, key, len(metadata['tokens'])+1, strength, epoch, arms.get(key))
                    losses = {key:validation(arms[key], held, validation_descriptors, key, metadata['tokens']) for key in arms}
                    item = dict(mixture=k, strength=strength, epoch=epoch, fold=fold, **losses)
                    results.setdefault((k, strength, epoch), []).append(item)
                    print(json.dumps(item), flush=True)
    summaries = [dict(mixture=k, strength=s, epoch=e, folds=v,
        matchedLoss=float(np.mean([(r['fisher']+r['control'])/2 for r in v]))) for (k, s, e), v in results.items()]
    selected = min(summaries, key=lambda r:(r['matchedLoss'], r['mixture'], r['strength'], r['epoch']))
    bg = background(rows, selected['mixture'])
    descriptors = [segments(bg, r['raw']) for r in rows]
    arms = {}
    for epoch in range(selected['epoch']+1):
        budget()
        for key in ('fisher', 'control'):
            arms[key] = fit(rows, descriptors, key, len(metadata['tokens'])+1, selected['strength'], epoch, arms.get(key))
    fitted = {**metadata, **bg, **arms}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extractor = out/'fisher-pronunciation.onnx'
    if not extractor.is_file():
        raise ValueError('Verified frozen extractor is missing')
    asset = out/'fisher-pronunciation-emissions.json'
    asset.write_text(json.dumps(fitted, allow_nan=False))
    files = [dict(file=p.name, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (asset, extractor)]
    (out/'fisher-pronunciation-asset.json').write_text(json.dumps(dict(schema=1, files=files)))
    report = dict(recordings=len(rows), speakers=speakers, speakerFolds=folds, excludedSpeakers=sorted(excluded),
        selected=selected, validation=summaries, cpuSeconds=time.process_time()-started,
        matchedReadoutParameters=2*READOUT*(len(metadata['tokens'])+1),
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(), files=files,
        limitation='Counterfactual lexical pronunciations on none-labelled training speech; scripted mistakes are not treated as verified spoken transcripts.')
    (out/'fisher-pronunciation-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--excluded-speakers', required=True)
    p.add_argument('--out', required=True)
    train(p.parse_args())

"""Training-only soft-labelled acoustic exemplar likelihoods; no practice input.

Whitening, neighbourhood size and smoothing are chosen by speaker-grouped
canonical-sequence likelihood. Each recording has unit total weight. Blank
versus speech remains supplied by the primary model in both comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path

DIM = 16
GRID = [(ridge, k, alpha) for ridge in (.01, .1) for k in (8, 24) for alpha in (.01, .1)]


def ctc(values, ids, posterior=False):
    import numpy as np
    labels = np.array([0] + [p for q in ids for p in (q, 0)])
    n, states = len(values), len(labels)
    if not n or not ids:
        raise ValueError('Empty complete pronunciation')
    skip = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    a = np.full((n, states), -np.inf)
    a[0, :2] = values[0, labels[:2]]
    for t in range(1, n):
        row = a[t-1].copy()
        row[1:] = np.logaddexp(row[1:], a[t-1, :-1])
        row[2:] = np.logaddexp(row[2:], np.where(skip, a[t-1, :-2], -np.inf))
        a[t] = row + values[t, labels]
    z = np.logaddexp(*a[-1, -2:])
    if not posterior:
        return float(z)
    if not np.isfinite(z):
        raise ValueError('Canonical pronunciation cannot fit the recording')
    b = np.full_like(a, -np.inf)
    b[-1, -2:] = 0
    for t in range(n-2, -1, -1):
        following = b[t+1] + values[t+1, labels]
        row = following.copy()
        row[:-1] = np.logaddexp(row[:-1], following[1:])
        row[:-2] = np.logaddexp(row[:-2], np.where(skip, following[2:], -np.inf))
        b[t] = row
    occupancy = np.zeros_like(values)
    for s, p in enumerate(labels):
        occupancy[:, p] += np.exp(a[:, s] + b[:, s] - z)
    return float(z), occupancy


def prepare(raw):
    import numpy as np
    excluded = {r['speaker'] for r in raw if r.get('split') != 'train'}
    rows = []
    for r in raw:
        if r.get('source') != 'help-train' or r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r['speaker'] in excluded:
            raise ValueError('Only none-labelled help-train training speakers are allowed')
        x, lp = np.asarray(r['features']), np.asarray(r['log_probs'])
        if x.ndim != 2 or lp.ndim != 2 or len(x) != len(lp) or not np.isfinite(x).all() or not np.isfinite(lp).all():
            raise ValueError('Invalid frozen pre-head observations')
        if not r['ids'] or any(not isinstance(p,int) or p < 1 or p >= lp.shape[1] for p in r['ids']):
            raise ValueError('Canonical labels must name supported non-blank phone classes')
        _, gamma = ctc(lp, r['ids'], True)
        # Retain at most 128 regularly spaced frames per recording, without an
        # argmax phone filter; fractional labels include every constrained path.
        pick = np.unique(np.linspace(0, len(x)-1, min(128, len(x))).astype(int))
        rows.append(dict(x=x, lp=lp, ids=r['ids'], gamma=gamma,
                         bank=x[pick], soft=gamma[pick, 1:], speaker=r['speaker']))
    if len({r['speaker'] for r in rows}) < 3:
        raise ValueError('At least three non-excluded training speakers required')
    return rows


def fit(rows, ridge, k, alpha):
    import numpy as np
    # Equal recording contributions also determine the whitening transform.
    mean = np.mean([r['bank'].mean(0) for r in rows], 0)
    cov = np.mean([(r['bank']-mean).T @ (r['bank']-mean)/len(r['bank']) for r in rows], 0)
    eigen, vectors = np.linalg.eigh(cov)
    take = np.argsort(eigen)[-min(DIM, len(mean)):]
    projection = (vectors[:, take]/np.sqrt(eigen[take] + ridge*np.trace(cov)/len(mean))).T
    bank, soft, recording, weights = [], [], [], []
    for i, r in enumerate(rows):
        bank.extend((r['bank']-mean) @ projection.T)
        soft.extend(r['soft'])
        recording.extend([i]*len(r['bank']))
        weights.extend([1/len(r['bank'])]*len(r['bank']))
    bank, soft, weights = np.array(bank), np.array(soft), np.array(weights)
    frequencies = weights @ soft
    return dict(center=mean.tolist(), projection=projection.tolist(), bank=bank.tolist(), soft=soft.tolist(),
                recording=recording, weights=weights.tolist(), frequencies=frequencies.tolist(), k=k, alpha=alpha,
                ridge=ridge, recordingCount=len(rows))


def likelihood(model, raw, lp):
    import numpy as np
    x = (raw-model['center']) @ np.array(model['projection']).T
    bank, soft = np.array(model['bank']), np.array(model['soft'])
    weights, freq = np.array(model['weights']), np.array(model['frequencies'])
    recordings = np.array(model['recording'])
    result = np.zeros_like(lp)
    for t, row in enumerate(x):
        distance = ((bank-row)**2).sum(1)
        # Each phone selects neighbouring RECORDINGS, not frames. For each
        # recording, select its closest supported frame for that class; limit
        # its weighted contribution to its unit recording budget.
        neighbours = []
        for p in range(soft.shape[1]):
            candidates = []
            for r in range(model['recordingCount']):
                eligible = np.flatnonzero((recordings == r) & (soft[:, p] > 0))
                if len(eligible):
                    j = eligible[np.argmin(distance[eligible]-2*np.log(soft[eligible, p]))]
                    candidates.append(int(j))
            candidates.sort(key=lambda j: (distance[j]-2*np.log(soft[j,p]), j))
            neighbours.append(candidates[:model['k']])
        selected = sorted(set(j for ns in neighbours for j in ns))
        bandwidth = max(float(np.median(distance[selected])), 1e-8) if selected else 1
        density = np.zeros(soft.shape[1])
        for p, ns in enumerate(neighbours):
            if ns and freq[p] > 0:
                density[p] = sum(np.exp(-distance[j]/(2*bandwidth))*soft[j, p]*weights[j] for j in ns)/freq[p]
        # Common regularizer toward the class-balanced background; no written
        # word, rival or label popularity enters neighbourhood selection.
        density += model['alpha'] * max(float(density.mean()), 1e-12)
        identity = density/density.sum()
        blank = np.exp(lp[t, 0])
        result[t, 0] = lp[t, 0]
        result[t, 1:] = np.log(np.maximum((1-blank)*identity, 1e-12))
    return result


def train(args):
    import numpy as np
    raw = [json.loads(line) for line in Path(args.input).open() if line.strip()]
    rows = prepare(raw)
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    folds = {s:i % 3 for i,s in enumerate(speakers)}
    trials = []
    for ridge,k,alpha in GRID:
        scores = []
        for fold in range(3):
            model = fit([r for r in rows if folds[r['speaker']] != fold], ridge,k,alpha)
            by_speaker = {}
            for r in rows:
                if folds[r['speaker']] != fold: continue
                score = ctc(likelihood(model,r['x'],r['lp']),r['ids'])/len(r['x'])
                by_speaker.setdefault(r['speaker'],[]).append(score)
            scores.append(float(np.mean([np.mean(v) for v in by_speaker.values()])))
        trial = dict(ridge=ridge,k=k,alpha=alpha,foldScores=scores,score=float(np.mean(scores)))
        trials.append(trial)
        print(json.dumps(trial),flush=True)
    best = max(trials,key=lambda t:t['score'])
    model = fit(rows,best['ridge'],best['k'],best['alpha'])
    model.update(schema=1,inventory=args.inventory,tokens=args.tokens,primarySha256=args.primary_sha256,output=args.output)
    dest = Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    (dest/'exemplar-density.json').write_text(json.dumps(model,allow_nan=False))
    (dest/'exemplar-training-report.json').write_text(json.dumps(dict(recordings=len(rows),speakers=len(speakers),
        folds=folds,trials=trials,selected=best,inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest()),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for key in ('input','out','tokens','inventory','primary-sha256','output'):p.add_argument('--'+key,required=True)
    train(p.parse_args())

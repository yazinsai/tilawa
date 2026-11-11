"""Matched forward/reverse acoustic EM, trained only on eligible none-labelled takes.

The input contains native SDK log-mel observations and canonical pronunciation.
No recognition logits, performed mistake transcripts, or practice examples enter fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path


def topology(ids):
    import numpy as np
    labels = np.array([0] + [q for p in ids for q in (p, 0)])
    n = len(labels)
    step = np.full(n, .2)
    skip = np.zeros(n, dtype=bool)
    for s in range(n):
        skip[s] = s >= 2 and labels[s] != 0 and labels[s] != labels[s-2]
        if s and labels[s-1] != 0 and s+1 < n and labels[s-1] != labels[s+1]:
            step[s] = .1
    return labels, np.log(step), skip


def chain(e, ids, posterior=False):
    """Sum every complete phone/background path, normalized by completion probability."""
    import numpy as np
    labels, step, skip = topology(ids)
    tmax, n = e.shape
    a = np.full((tmax, n), -np.inf)
    a[0, :2] = -np.log(2) + e[0, :2]
    prior = np.full(n, -np.inf)
    prior[:2] = -np.log(2)
    stay, leap = np.log(.8), np.log(.1)
    def advance(v):
        return np.logaddexp(np.logaddexp(v + stay, np.r_[-np.inf, v[:-1]] + step),
            np.where(skip, np.r_[-np.inf, -np.inf, v[:-2]] + leap, -np.inf))
    for t in range(1, tmax):
        a[t] = advance(a[t-1]) + e[t]
        prior = advance(prior)
    likelihood = np.logaddexp(a[-1, -1], a[-1, -2])
    value = likelihood - np.logaddexp(prior[-1], prior[-2])
    if not posterior:
        return float(value)
    b = np.full_like(a, -np.inf)
    b[-1, -2:] = 0
    for t in range(tmax-2, -1, -1):
        v = b[t+1] + e[t+1]
        b[t] = np.logaddexp(np.logaddexp(v + stay, np.r_[v[1:] + step[1:], -np.inf]),
            np.r_[np.where(skip[2:], v[2:] + leap, -np.inf), -np.inf, -np.inf])
    return float(value), np.exp(a + b - likelihood), labels


def emission(model, x, z, labels):
    import numpy as np
    b, a, v = (np.array(model[k])[labels] for k in ('intercept', 'coefficient', 'variance'))
    # Expansion avoids a T x S x 80 temporary; both arms use the same predictor capacity.
    result = x**2 @ (1/v).T + z**2 @ (a*a/v).T + np.sum(b*b/v, 1)[None]
    result -= 2*x @ (b/v).T + 2*(x*z) @ (a/v).T
    result += 2*z @ (a*b/v).T
    return -.5*(result + np.log(2*np.pi*v).sum(1)[None])


def update(stats, shrinkage):
    import numpy as np
    count, sx, sz, sxx, szz, sxz = stats
    def regression(c, x, z, zz, xz, target, penalty):
        aa, bb, cc = c[:, None]+penalty, z, zz+penalty
        y, w = x+penalty*target[0], xz+penalty*target[1]
        det = np.maximum(aa*cc-bb*bb, 1e-12)
        return (cc*y-bb*w)/det, (aa*w-bb*y)/det
    global_b, global_a = regression(count.sum(keepdims=True), sx.sum(0, keepdims=True),
        sz.sum(0, keepdims=True), szz.sum(0, keepdims=True), sxz.sum(0, keepdims=True), (0, 0), 1)
    b, a = regression(count, sx, sz, szz, sxz, (global_b, global_a), shrinkage)
    residual = sxx-2*b*sx-2*a*sxz+b*b*count[:, None]+a*a*szz+2*a*b*sz
    global_var = np.maximum(residual.sum(0)/max(count.sum(), 1), .05)
    variance = np.maximum((residual+shrinkage*global_var)/(count[:, None]+shrinkage), .05)
    return dict(intercept=b.tolist(), coefficient=a.tolist(), variance=variance.tolist(), counts=count.tolist())


def fit(rows, vocab, shrinkage, reverse, epochs=8):
    import numpy as np
    parts = []
    for r in rows:
        x = r['x'][::-1].copy() if reverse else r['x']
        ids = r['ids'][::-1] if reverse else r['ids']
        parts.append((x, np.vstack([np.zeros((1, 80)), x[:-1]]), ids))
    def fresh():
        return [np.zeros(vocab)] + [np.zeros((vocab, 80)) for _ in range(5)]
    def accumulate(stats, x, z, weights):
        stats[0] += weights.sum(0)
        for i, v in enumerate((x, z, x*x, z*z, x*z), 1):
            stats[i] += weights.T @ v
    stats = fresh()
    # Only initialization assigns uniform boundaries; every subsequent fit integrates all boundaries.
    for x, z, ids in parts:
        labels, _, _ = topology(ids)
        state = np.minimum(np.arange(len(x))*len(labels)//len(x), len(labels)-1)
        weights = np.zeros((len(x), vocab))
        weights[np.arange(len(x)), labels[state]] = 1
        accumulate(stats, x, z, weights)
    model = update(stats, shrinkage)
    history = []
    for _ in range(epochs):
        stats, values = fresh(), []
        for x, z, ids in parts:
            labels, _, _ = topology(ids)
            value, gamma, labels = chain(emission(model, x, z, labels), ids, True)
            if not np.isfinite(value):
                raise ValueError('Impossible complete training pronunciation')
            weights = np.zeros((len(x), vocab))
            for s, p in enumerate(labels):
                weights[:, p] += gamma[:, s]
            accumulate(stats, x, z, weights)
            values.append(value/len(x))
        model = update(stats, shrinkage)
        history.append(float(np.mean(values)))
    return model, history


def validation(model, rows, reverse):
    import numpy as np
    speakers = {}
    for r in rows:
        x = r['x'][::-1] if reverse else r['x']
        ids = r['ids'][::-1] if reverse else r['ids']
        z = np.vstack([np.zeros((1, 80)), x[:-1]])
        labels, _, _ = topology(ids)
        value = chain(emission(model, x, z, labels), ids)/len(x)
        speakers.setdefault(r['speaker'], []).append(value)
    return float(np.mean([np.mean(v) for v in speakers.values()]))


def prepare(raw, tokens, excluded):
    import numpy as np
    out = []
    for r in raw:
        if r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r['speaker'] in excluded:
            raise ValueError('Only none-labelled, non-excluded training-speaker recordings may be fitted')
        x = np.array(r['features'], dtype=float)
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        if x.ndim != 2 or x.shape[1] != 80 or not np.isfinite(x).all() or not ids or len(x) < len(ids) or len(x)>8192:
            raise ValueError('Invalid SDK observations or impossible complete pronunciation')
        out.append(dict(x=x, ids=ids, speaker=r['speaker']))
    if len({r['speaker'] for r in out}) < 3:
        raise ValueError('At least three training speakers required')
    return out


def normalization(train, other):
    import numpy as np
    raw = np.concatenate([r['x'] for r in train])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .1)
    normalize = lambda rows: [dict(r, x=(r['x']-center)/scale) for r in rows]
    return normalize(train), normalize(other), center, scale


def main():
    parser = argparse.ArgumentParser()
    for key in ['input', 'out', 'tokens', 'inventory', 'excluded-speakers']:
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = prepare([json.loads(line) for line in Path(args.input).open()], args.tokens, excluded)
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    results = []
    for shrinkage in [8, 32, 128]:
        values = []
        for fold in range(3):
            train, valid, _, _ = normalization([r for r in rows if folds[r['speaker']] != fold],
                [r for r in rows if folds[r['speaker']] == fold])
            scores = []
            for reverse in (False, True):
                model, _ = fit(train, len(args.tokens)+1, shrinkage, reverse)
                scores.append(validation(model, valid, reverse))
            values.append(scores)
            print(json.dumps(dict(shrinkage=shrinkage, fold=fold, forward=scores[0], reverse=scores[1])), flush=True)
        results.append(dict(shrinkage=shrinkage, foldLogP=values, meanLogP=sum(sum(s) for s in values)/6))
    chosen = max(results, key=lambda r: (r['meanLogP'], r['shrinkage']))
    train, _, center, scale = normalization(rows, [])
    forward, fh = fit(train, len(args.tokens)+1, chosen['shrinkage'], False)
    reverse, rh = fit(train, len(args.tokens)+1, chosen['shrinkage'], True)
    model = dict(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40, observationMs=10,
        dimensions=80, maxFrames=8192, center=center.tolist(), scale=scale.tolist(), forward=forward, reverse=reverse)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'reverse-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'reverse-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        speakerFolds=folds, excludedSpeakers=sorted(excluded), validation=results, selected=chosen,
        forwardHistory=fh, reverseHistory=rh, asset=descriptor)
    (out/'reverse-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

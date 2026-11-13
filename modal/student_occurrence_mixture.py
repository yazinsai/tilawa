"""Training-only frozen-feature GMMs with occurrence-tied CTC latent components.

The frame-switching control reuses every fitted emission, prior and timing arc.
No practice observations participate in projection, fitting or model selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

DIMENSIONS = 12
COMPONENTS = 3
EM_STEPS = 3


def boundary(values, weights, ids, tied=True, posterior=False):
    import numpy as np
    from scipy.special import logsumexp
    values, weights = np.asarray(values), np.asarray(weights)
    if not len(values) or not ids:
        return -np.inf
    labels = np.array([0] + [v for p in ids for v in (p, 0)])
    n, k = len(labels), weights.shape[1] if tied else 1
    skip = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    degree = np.ones(n)
    degree[:-1] += 1
    degree[:-2] += skip
    cost = np.log(degree)
    mixed = logsumexp(values + np.log(weights)[None], axis=2)
    prior = np.full((n, k), -np.inf)
    prior[::2, 0] = 0
    if tied:
        prior[1::2] = np.log(weights[labels[1::2]])
        e = values[:, labels].copy()
        e[:, ::2, 0] = mixed[:, labels[::2]]
        e[:, ::2, 1:] = -np.inf
    else:
        prior[:, 0] = 0
        e = mixed[:, labels, None]
    a = np.full_like(e, -np.inf)
    a[0, :2] = e[0, :2] + prior[:2]
    timing = np.full(n, -np.inf)
    timing[:2] = 0
    for t in range(1, len(e)):
        outgoing = logsumexp(a[t-1], axis=1) - cost
        entering = np.full(n, -np.inf)
        entering[1:] = outgoing[:-1]
        entering[2:] = np.logaddexp(entering[2:], np.where(skip, outgoing[:-2], -np.inf))
        a[t] = np.logaddexp(a[t-1]-cost[:, None], entering[:, None]+prior) + e[t]
        out = timing-cost
        timing = out.copy()
        timing[1:] = np.logaddexp(timing[1:], out[:-1])
        timing[2:] = np.logaddexp(timing[2:], np.where(skip, out[:-2], -np.inf))
    z, normalizer = logsumexp(a[-1, -2:]), logsumexp(timing[-2:])
    if not np.isfinite(z) or not np.isfinite(normalizer):
        if posterior:
            raise ValueError('Incomplete canonical pronunciation')
        return -np.inf
    if not posterior:
        return float(z-normalizer)
    b = np.full_like(a, -np.inf)
    b[-1, -2:] = 0
    for t in range(len(e)-2, -1, -1):
        incoming = b[t+1]+e[t+1]
        exits = logsumexp(incoming+prior, axis=1)
        onward = np.full(n, -np.inf)
        onward[:-1] = exits[1:]
        onward[:-2] = np.logaddexp(onward[:-2], np.where(skip, exits[2:], -np.inf))
        b[t] = np.logaddexp(incoming, onward[:, None])-cost[:, None]
    occ = np.exp(a+b-z)
    posts = np.zeros_like(values)
    entry_count = np.zeros_like(weights)
    for s, p in enumerate(labels):
        if p == 0 or not tied:
            resp = np.exp(values[:, p]+np.log(weights[p])-mixed[:, p, None])
            posts[:, p] += occ[:, s].sum(1)[:, None]*resp
        else:
            posts[:, p] += occ[:, s]
            if s == 1:
                entry_count[p] += occ[0, s]
            entering = logsumexp(a[:-1, s-1], axis=1)-cost[s-1]
            if s >= 2 and labels[s] != labels[s-2]:
                entering = np.logaddexp(entering, logsumexp(a[:-1, s-2], axis=1)-cost[s-2])
            entry_count[p] += np.exp(entering[:, None]+prior[s]+e[1:, s]+b[1:, s]-z).sum(0)
    entry_count[0] = posts[:, 0].sum(0)
    return float(z-normalizer), posts, entry_count


def emissions(model, x):
    import numpy as np
    means, variances = model['means'], model['variances']
    return -.5*(np.log(2*np.pi*variances)[None] +
        (x[:, None, None]-means[None])**2/variances[None]).sum(3)


def fit(rows, strength, vocab):
    import numpy as np
    from scipy.special import logsumexp
    raw = np.concatenate([r['features'] for r in rows])
    center = raw.mean(0)
    covariance = (raw-center).T @ (raw-center)/len(raw)
    eig, basis = np.linalg.eigh(covariance)
    projection = (basis[:, -DIMENSIONS:] / np.sqrt(np.maximum(eig[-DIMENSIONS:], .01))).T
    xs = [(r['features']-center) @ projection.T for r in rows]
    initial = [boundary(r['log_probs'][:, :, None], np.ones((vocab, 1)), r['ids'], posterior=True)[1][:, :, 0] for r in rows]
    x = np.concatenate(xs)
    w = np.concatenate(initial)
    global_mean, global_var = x.mean(0), np.maximum(x.var(0), .05)
    means = np.empty((vocab, COMPONENTS, DIMENSIONS))
    variances = np.empty_like(means)
    for p in range(vocab):
        selected = w[:, p] > .05
        xp, wp = x[selected], w[selected, p]
        if not len(xp):
            means[p] = global_mean
            variances[p] = global_var
            continue
        mu = np.average(xp, axis=0, weights=wp)
        var = np.maximum(np.average((xp-mu)**2, axis=0, weights=wp), .05)
        axis = int(var.argmax())
        order = np.argsort(xp[:, axis], kind='stable')
        cumulative = np.cumsum(wp[order])
        seeds = [order[min(np.searchsorted(cumulative, q*cumulative[-1]), len(order)-1)] for q in (1/6, 1/2, 5/6)]
        means[p] = xp[seeds]
        variances[p] = var
    model = dict(center=center, projection=projection, means=means, variances=variances,
        weights=np.full((vocab, COMPONENTS), 1/COMPONENTS), count=w.sum(0))
    def update(posts, entries):
        count = np.zeros((vocab, COMPONENTS))
        sums = np.zeros_like(means)
        squares = np.zeros_like(means)
        for xx, ww in zip(xs, posts):
            count += ww.sum(0)
            sums += np.einsum('tpk,td->pkd', ww, xx)
            squares += np.einsum('tpk,td->pkd', ww, xx*xx)
        prior_count = strength/COMPONENTS
        mu = (sums+prior_count*global_mean)/(count[:, :, None]+prior_count)
        var = (squares+prior_count*(global_var+global_mean**2))/(count[:, :, None]+prior_count)-mu**2
        model.update(means=mu, variances=np.maximum(var, .05), count=count.sum(1))
        counts = sum(entries)+prior_count
        model['weights'] = counts/counts.sum(1, keepdims=True)
    # Deterministic GMM initialization on uncertain parent occupancies.
    for _ in range(3):
        posts = []
        for xx, ww in zip(xs, initial):
            v = emissions(model, xx)+np.log(model['weights'])[None]
            posts.append(np.exp(v-logsumexp(v, axis=2, keepdims=True))*ww[:, :, None])
        update(posts, [p.sum(0) for p in posts])
    # The latent identity and boundaries are then inferred jointly per occurrence.
    for _ in range(EM_STEPS):
        result = [boundary(emissions(model, xx), model['weights'], r['ids'], posterior=True) for r, xx in zip(rows, xs)]
        update([v[1] for v in result], [v[2] for v in result])
    return model


def loss(model, rows):
    import numpy as np
    grouped = {}
    for r in rows:
        v = emissions(model, (r['features']-model['center']) @ model['projection'].T)
        # Complete-pronunciation likelihood; every validation speaker has equal weight.
        score = boundary(v, model['weights'], r['ids'])
        grouped.setdefault(r['speaker'], []).append(-score/len(v))
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def train(args):
    import numpy as np
    meta = json.loads(Path(args.metadata).read_text())
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = []
    for line in Path(args.input).read_text().splitlines():
        r = json.loads(line)
        if r.get('source') != 'help-train' or r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False:
            raise ValueError('Only help-train training-speaker none takes may fit emissions')
        if not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Missing or held-out speaker')
        r['ids'] = [meta['tokens'].index(p)+1 for p in r['pronunciation']]
        for key, width in [('features', meta['width']), ('log_probs', len(meta['tokens'])+1)]:
            r[key] = np.asarray(r[key], dtype=np.float64)
            if r[key].ndim != 2 or r[key].shape[1] != width or not len(r[key]) or not np.isfinite(r[key]).all():
                raise ValueError('Invalid frozen observations')
        if len(r['features']) != len(r['log_probs']):
            raise ValueError('Feature clocks differ')
        rows.append(r)
    speakers = sorted({r['speaker'] for r in rows})
    if len(speakers) < 3:
        raise ValueError('At least three independent training speakers required')
    folds = {s: i%3 for i, s in enumerate(speakers)}
    results = []
    for strength in (4., 16., 64.):
        losses = []
        for fold in range(3):
            training = [r for r in rows if folds[r['speaker']] != fold]
            held = [r for r in rows if folds[r['speaker']] == fold]
            losses.append(loss(fit(training, strength, len(meta['tokens'])+1), held))
        results.append(dict(strength=strength, foldLosses=losses, loss=float(np.mean(losses))))
        print(json.dumps(results[-1]), flush=True)
    selected = min(results, key=lambda r: (r['loss'], r['strength']))
    model = fit(rows, selected['strength'], len(meta['tokens'])+1)
    fitted = dict(**meta, dimensions=DIMENSIONS, center=model['center'].tolist(), projection=model['projection'].tolist(),
        phones=[dict(weights=model['weights'][p].tolist(), means=model['means'][p].tolist(),
            variances=model['variances'][p].tolist(), count=float(model['count'][p])) for p in range(len(meta['tokens'])+1)])
    out = Path(args.out)
    emission = out/'occurrence-mixture-emissions.json'
    emission.write_text(json.dumps(fitted, allow_nan=False))
    extractor = out/'occurrence-mixture.onnx'
    if not extractor.is_file():
        raise ValueError('Verified extractor missing')
    files = [dict(file=p.name, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (emission, extractor)]
    (out/'occurrence-mixture-asset.json').write_text(json.dumps(dict(schema=1, files=files)))
    report = dict(recordings=len(rows), frames=sum(len(r['features']) for r in rows), speakers=speakers, speakerFolds=folds,
        excludedSpeakers=sorted(excluded), validation=results, selected=selected, components=COMPONENTS,
        dimensions=DIMENSIONS, emSteps=EM_STEPS, files=files,
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    (out/'occurrence-mixture-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('metadata', 'input', 'excluded-speakers', 'out'):
        p.add_argument('--'+name, required=True)
    train(p.parse_args())

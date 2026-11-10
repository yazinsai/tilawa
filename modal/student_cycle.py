"""CPU-only latent-boundary fit of phase-conditioned aperiodic high-band energy.

The SDK extracts all observations, so training and deployed filtering/closure
estimation are identical. Only none-labelled, non-excluded help-train training
speakers provide transcripts. No scripted performance is assumed. Speaker folds
select regularization before any practice window is inspected.
"""
import argparse
import hashlib
import json
from pathlib import Path


def chain(emission, stay, posterior=False):
    import numpy as np
    frames, states = emission.shape
    alpha = np.full_like(emission, -np.inf)
    prior = np.full(states, -np.inf)
    alpha[0, 0], prior[0] = emission[0, 0], 0
    for t in range(1, frames):
        alpha[t] = np.logaddexp(alpha[t-1] + np.log(stay),
            np.r_[-np.inf, alpha[t-1, :-1]] + np.log1p(-stay)) + emission[t]
        prior = np.logaddexp(prior + np.log(stay), np.r_[-np.inf, prior[:-1]] + np.log1p(-stay))
    value = alpha[-1, -1] - prior[-1]
    if not posterior:
        return value
    beta = np.full_like(emission, -np.inf)
    beta[-1, -1] = 0
    for t in range(frames-2, -1, -1):
        beta[t] = np.logaddexp(np.log(stay) + emission[t+1] + beta[t+1],
            np.r_[np.log1p(-stay) + emission[t+1, 1:] + beta[t+1, 1:], -np.inf])
    return value, np.exp(alpha + beta - alpha[-1, -1])


def densities(model, x, reliable, ids, conditioned=True):
    import numpy as np
    means, var = np.asarray(model['means'])[ids], np.asarray(model['variances'])[ids]
    d = x[:, None] - means[None]
    full = -.5*(np.log(2*np.pi*var)[None] + d*d/var[None])
    # Both views use the same standardized phase-background observation measure.
    # Missing phase integrates out exactly and contributes no identity evidence.
    full[:, :, 2:] += .5*(np.log(2*np.pi) + x[:, None, 2:]**2)
    full[:, :, 2:] *= reliable[:, None, None] if conditioned else 0
    return full.sum(2)


def update(counts, sums, seconds, regularization):
    import numpy as np
    total = np.maximum(counts.sum(0), 1)
    mu = sums.sum(0)/total
    variance = np.maximum(seconds.sum(0)/total - mu*mu, .05)
    denominator = counts + regularization
    means = (sums + regularization*mu)/denominator
    var = (seconds + regularization*(variance+mu*mu))/denominator - means*means
    return means.tolist(), np.maximum(var, .05).tolist()


def fit(rows, tokens, regularization, epochs=10):
    import numpy as np
    allx = np.concatenate([r['raw'][:, :16] for r in rows])
    allq = np.concatenate([r['raw'][:, 16] for r in rows])
    mask = np.ones_like(allx)
    mask[:, 2:] *= allq[:, None]
    total = np.maximum(mask.sum(0), 1)
    center = (mask*allx).sum(0)/total
    scale = np.maximum(np.sqrt((mask*(allx-center)**2).sum(0)/total), .1)
    normalized = [dict(r, x=(r['raw'][:, :16]-center)/scale, q=r['raw'][:, 16]) for r in rows]
    n = len(tokens)+1

    def fresh():
        return np.zeros((n, 16)), np.zeros((n, 16)), np.zeros((n, 16))

    counts, sums, seconds = fresh()
    for r in normalized:
        states = np.minimum(np.arange(len(r['x']))*len(r['ids'])//len(r['x']), len(r['ids'])-1)
        for s, p in enumerate(r['ids']):
            x, q = r['x'][states == s], r['q'][states == s]
            weights = np.ones_like(x)
            weights[:, 2:] *= q[:, None]
            counts[p] += weights.sum(0)
            sums[p] += (weights*x).sum(0)
            seconds[p] += (weights*x*x).sum(0)
    means, var = update(counts, sums, seconds, regularization)
    stay = (sum(len(r['x'])-len(r['ids']) for r in normalized)+1)/(sum(len(r['x'])-1 for r in normalized)+2)
    model = dict(center=center.tolist(), scale=scale.tolist(), means=means, variances=var, stay=stay)
    history = []
    for _ in range(epochs):
        counts, sums, seconds = fresh()
        values = []
        for r in normalized:
            emission = densities(model, r['x'], r['q'], r['ids'])
            value, gamma = chain(emission, stay, True)
            values.append(float(value)/len(r['x']))
            mask = np.ones_like(r['x'])
            mask[:, 2:] *= r['q'][:, None]
            for s, p in enumerate(r['ids']):
                weight = gamma[:, s, None]*mask
                counts[p] += weight.sum(0)
                sums[p] += (weight*r['x']).sum(0)
                seconds[p] += (weight*r['x']**2).sum(0)
        model['means'], model['variances'] = update(counts, sums, seconds, regularization)
        history.append(sum(values)/len(values))
    model['counts'], model['cycleCounts'] = counts[:, 0].tolist(), counts[:, 2].tolist()
    return model, history


def validation(model, rows):
    import numpy as np
    speakers = {}
    for r in rows:
        x = (r['raw'][:, :16]-model['center'])/model['scale']
        emission = densities(model, x, r['raw'][:, 16], r['ids'])
        value = chain(emission, model['stay'])/len(x)
        speakers.setdefault(r['speaker'], []).append(float(value))
    return float(np.mean([np.mean(values) for values in speakers.values()]))


def prepare(rows, tokens, excluded):
    import numpy as np
    out = []
    if not isinstance(excluded, list) or any(not isinstance(s, str) for s in excluded):
        raise ValueError('An explicit excluded-speaker list is required')
    for r in rows:
        if r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r['speaker'] in excluded:
            raise ValueError('Only explicitly none-labelled, non-excluded help-train training speakers are authorized')
        raw = np.asarray(r['features'], dtype=float)
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        if raw.ndim != 2 or raw.shape[1] != 17 or not np.isfinite(raw).all() or not ids or len(raw) < len(ids) or len(raw)>4096:
            raise ValueError('Invalid SDK observations or impossible complete pronunciation')
        if not np.isin(raw[:, 16], [0, 1]).all():
            raise ValueError('Invalid cycle reliability')
        out.append(dict(raw=raw, ids=ids, speaker=r['speaker']))
    if len({r['speaker'] for r in out}) < 3 or not any(r['raw'][:, 16].sum() for r in out):
        raise ValueError('At least three eligible training speakers and reliable cycles are required')
    return out


def main():
    parser = argparse.ArgumentParser()
    for key in ['input', 'out', 'tokens', 'inventory', 'excluded-speakers']:
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    excluded = json.loads(Path(args.excluded_speakers).read_text())
    rows = prepare([json.loads(line) for line in Path(args.input).open()], args.tokens, excluded)
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    results = []
    for regularization in [16, 64, 256]:
        values = []
        for fold in range(3):
            model, _ = fit([r for r in rows if folds[r['speaker']] != fold], args.tokens, regularization)
            values.append(validation(model, [r for r in rows if folds[r['speaker']] == fold]))
        result = dict(regularization=regularization, foldLogP=values, meanLogP=sum(values)/3)
        results.append(result)
        print(json.dumps(result), flush=True)
    chosen = max(results, key=lambda r: (r['meanLogP'], r['regularization']))
    model, history = fit(rows, args.tokens, chosen['regularization'])
    model.update(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40, observationMs=10, maxFrames=4096)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'cycle-turbulence-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'cycle-turbulence-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        speakerFolds=folds, excludedSpeakers=sorted(excluded), validation=results, selected=chosen, history=history,
        reliableFrames=sum(int(r['raw'][:, 16].sum()) for r in rows), asset=descriptor)
    (out/'cycle-turbulence-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

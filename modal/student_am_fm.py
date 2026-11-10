"""CPU EM of waveform amplitude / instantaneous-frequency joint trajectories.

Input observations come from the SDK's fixed complex filterbank. Only explicitly
none-labelled help-train training speakers supply complete transcripts. Scripts
for mistakes are never converted to performed transcripts. Speaker folds freeze
band selection and covariance regularization before any practice inspection.
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


def densities(model, x, q, ids, joint=True):
    import numpy as np
    bands = len(model['bands'])
    means = np.asarray(model['means'])[ids].reshape(-1, bands, 2)
    variances = np.asarray(model['variances'])[ids].reshape(-1, bands, 2)
    covariance = np.asarray(model['covariance'])[ids] if joint else np.zeros(means.shape[:2])
    d = x[:, None] - means[None]
    va, vf, c = variances[:, :, 0], variances[:, :, 1], covariance
    determinant = va*vf-c*c
    amplitude = -.5*(np.log(2*np.pi*va)[None] + d[:, :, :, 0]**2/va[None])
    full = -np.log(2*np.pi) - .5*np.log(determinant)[None] - .5*(
        vf[None]*d[:, :, :, 0]**2 + va[None]*d[:, :, :, 1]**2 -
        2*c[None]*d[:, :, :, 0]*d[:, :, :, 1])/determinant[None]
    # Shared standardized background for unreliable observed frequency. Subtracting
    # it from both branches preserves comparisons and makes q=0 an exact marginal.
    full += .5*(np.log(2*np.pi) + x[:, None, :, 1]**2)
    with np.errstate(divide='ignore'):
        reliable = np.log(q[:, None]) + full
        missing = np.log1p(-q[:, None]) + amplitude
    mixed = np.logaddexp(reliable, missing)
    responsibility = np.exp(reliable-mixed)
    conditional_mean = means[None, :, :, 1] + c[None]/va[None]*d[:, :, :, 0]
    conditional_variance = vf-c*c/va
    return mixed.sum(2), responsibility, conditional_mean, conditional_variance


def update(counts, sums, seconds, regularization):
    import numpy as np
    total = np.maximum(counts.sum(0), 1)
    mu = sums.sum(0)/total[:, None]
    # A diagonal global prior avoids inventing a shared coupling for unsupported phones.
    variance = np.maximum(np.diagonal(seconds.sum(0)/total[:, None, None], axis1=1, axis2=2)-mu**2, .05)
    prior = mu[:, :, None]*mu[:, None, :]
    prior[:, 0, 0] += variance[:, 0]
    prior[:, 1, 1] += variance[:, 1]
    denominator = counts + regularization
    means = (sums + regularization*mu)/denominator[:, :, None]
    covariance = (seconds + regularization*prior)/denominator[:, :, None, None] - means[:, :, :, None]*means[:, :, None, :]
    var = np.maximum(np.diagonal(covariance, axis1=2, axis2=3), .05)
    cross = covariance[:, :, 0, 1]
    # Positive-definite shrinkage, identical in training and SDK; no practice cut.
    cross = np.clip(cross, -.95*np.sqrt(var[:, :, 0]*var[:, :, 1]), .95*np.sqrt(var[:, :, 0]*var[:, :, 1]))
    return means.reshape(len(means), -1).tolist(), var.reshape(len(means), -1).tolist(), cross.tolist()


def fit(rows, tokens, bands, regularization, epochs=10):
    import numpy as np
    allx = np.concatenate([r['raw'][:, bands, :2] for r in rows])
    allq = np.concatenate([r['raw'][:, bands, 2] for r in rows])
    weights = np.stack([np.ones_like(allq), allq], axis=2)
    center = (weights*allx).sum(0)/np.maximum(weights.sum(0), 1)
    scale = np.maximum(np.sqrt((weights*(allx-center)**2).sum(0)/np.maximum(weights.sum(0), 1)), .1)
    normalized = [dict(r, x=(r['raw'][:, bands, :2]-center)/scale, q=r['raw'][:, bands, 2]) for r in rows]
    n, b = len(tokens)+1, len(bands)
    def fresh():
        return np.zeros((n, b)), np.zeros((n, b, 2)), np.zeros((n, b, 2, 2))
    counts, sums, seconds = fresh()
    for r in normalized:
        states = np.minimum(np.arange(len(r['x']))*len(r['ids'])//len(r['x']), len(r['ids'])-1)
        for s, p in enumerate(r['ids']):
            x = r['x'][states == s]
            counts[p] += len(x)
            sums[p] += x.sum(0)
            seconds[p] += np.einsum('tbi,tbj->bij', x, x)
    means, var, cov = update(counts, sums, seconds, regularization)
    stay = (sum(len(r['x'])-len(r['ids']) for r in normalized)+1)/(sum(len(r['x'])-1 for r in normalized)+2)
    model = dict(bands=bands, center=center.tolist(), scale=scale.tolist(), means=means, variances=var, covariance=cov, stay=stay)
    history = []
    for _ in range(epochs):
        counts, sums, seconds = fresh()
        values = []
        for r in normalized:
            emission, rho, cond, condvar = densities(model, r['x'], r['q'], r['ids'])
            value, gamma = chain(emission, stay, True)
            values.append(float(value)/len(r['x']))
            for s, p in enumerate(r['ids']):
                weight = gamma[:, s, None]
                a = r['x'][:, :, 0]
                f = rho[:, s]*r['x'][:, :, 1] + (1-rho[:, s])*cond[:, s]
                ff = rho[:, s]*r['x'][:, :, 1]**2 + (1-rho[:, s])*(cond[:, s]**2 + condvar[s])
                counts[p] += weight.sum(0)
                sums[p, :, 0] += (weight*a).sum(0)
                sums[p, :, 1] += (weight*f).sum(0)
                seconds[p, :, 0, 0] += (weight*a*a).sum(0)
                seconds[p, :, 1, 1] += (weight*ff).sum(0)
                seconds[p, :, 0, 1] += (weight*a*f).sum(0)
                seconds[p, :, 1, 0] += (weight*a*f).sum(0)
        model['means'], model['variances'], model['covariance'] = update(counts, sums, seconds, regularization)
        history.append(sum(values)/len(values))
    model['counts'] = counts.mean(1).tolist()
    return model, history


def validation(model, rows):
    import numpy as np
    speakers = {}
    for r in rows:
        raw = r['raw'][:, model['bands']]
        x = (raw[:, :, :2]-model['center'])/model['scale']
        emission, *_ = densities(model, x, raw[:, :, 2], r['ids'])
        value = chain(emission, model['stay'])/len(raw)
        speakers.setdefault(r['speaker'], []).append(float(value)/len(model['bands']))
    return float(np.mean([np.mean(values) for values in speakers.values()]))


def prepare(rows, tokens, excluded):
    import numpy as np
    out = []
    for r in rows:
        if r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r['speaker'] in excluded:
            raise ValueError('Only explicitly none-labelled, non-excluded help-train training speakers are authorized')
        raw = np.asarray(r['features'], dtype=float)
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        if raw.ndim != 2 or raw.shape[1] != 24 or not np.isfinite(raw).all() or not ids or len(raw) < len(ids):
            raise ValueError('Invalid SDK observations or impossible complete pronunciation')
        raw = raw.reshape(-1, 8, 3)
        if (raw[:, :, 2] < 0).any() or (raw[:, :, 2] > 1).any():
            raise ValueError('Invalid IF reliability')
        out.append(dict(raw=raw, ids=ids, speaker=r['speaker']))
    if len({r['speaker'] for r in out}) < 3:
        raise ValueError('At least three eligible training speakers required')
    return out


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
    for bands in [[0, 1, 2, 3], list(range(8))]:
        for regularization in [16, 64, 256]:
            values = []
            for fold in range(3):
                model, _ = fit([r for r in rows if folds[r['speaker']] != fold], args.tokens, bands, regularization)
                values.append(validation(model, [r for r in rows if folds[r['speaker']] == fold]))
            result = dict(bands=bands, regularization=regularization, foldLogP=values, meanLogP=sum(values)/3)
            results.append(result)
            print(json.dumps(result), flush=True)
    chosen = max(results, key=lambda r: (r['meanLogP'], r['regularization'], -len(r['bands'])))
    model, history = fit(rows, args.tokens, chosen['bands'], chosen['regularization'])
    model.update(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40, observationMs=10, maxFrames=4096)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'am-fm-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'am-fm-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        speakerFolds=folds, excludedSpeakers=sorted(excluded), validation=results, selected=chosen, history=history, asset=descriptor)
    (out/'am-fm-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

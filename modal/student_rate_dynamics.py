"""CPU EM for spectral targets approached through shared acoustic dynamics.

Only none-labelled help-train training-speaker observations enter estimation.
The control has independent realizations with learned shared dispersion instead
of shared retention: targets*dimensions + dimensions parameters in either arm.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

DIMENSIONS = 20


def emissions(model, x, ids, dynamic, previous=None):
    targets = np.asarray(model['targets'])[ids]
    shared = np.asarray(model['shared'])
    before = np.vstack([np.zeros(DIMENSIONS) if previous is None else previous, x[:-1]])
    mean = (shared[None, None, :]*before[:, None, :] +
            (1-shared)[None, None, :]*targets[None, :, :]) if dynamic else targets[None, :, :]
    variance = np.asarray(model['innovation'])*(1 if dynamic else np.exp(shared))
    return -.5*np.sum((x[:, None, :]-mean)**2/variance+np.log(2*np.pi*variance), axis=2)


def evaluate(model, x, ids, dynamic, posterior=False, previous=None):
    """Exact complete-pronunciation forward/backward and normalized timing prior."""
    tmax, n = len(x), len(ids)
    if tmax < n or not n:
        raise ValueError('Impossible complete pronunciation')
    e = emissions(model, x, ids, dynamic, previous)
    alpha = np.full((tmax, n), -np.inf)
    alpha[0, 0] = e[0, 0]
    prior = np.full(n, -np.inf); prior[0] = 0
    for t in range(1, tmax):
        alpha[t] = alpha[t-1]
        alpha[t, 1:] = np.logaddexp(alpha[t, 1:], alpha[t-1, :-1])
        alpha[t] += e[t]
        following = prior.copy()
        following[1:] = np.logaddexp(following[1:], prior[:-1])
        prior = following
    likelihood = alpha[-1, -1]
    if not posterior:
        return float(likelihood-prior[-1])
    beta = np.full_like(alpha, -np.inf); beta[-1, -1] = 0
    for t in range(tmax-2, -1, -1):
        following = e[t+1]+beta[t+1]
        beta[t] = following
        beta[t, :-1] = np.logaddexp(beta[t, :-1], following[1:])
    return float(likelihood-prior[-1]), np.exp(alpha+beta-likelihood)


def attributes(p):
    return next((name for chars, name in [('َُِاۦۥ', 'vowel'), ('منں۾', 'nasal'),
        ('بتدطكقءڇ', 'stop'), ('ثذفسشصزظخغحعه', 'fricative'), ('رلوي', 'approximant')]
        if p in chars), 'other')


def objective(params, observations, before, phone, mass, innovation, pool, shrinkage, dynamic):
    n = len(pool)
    targets = params[:n*DIMENSIONS].reshape(n, DIMENSIONS)
    shared = params[n*DIMENSIONS:]
    if dynamic:
        mean = shared*before+(1-shared)*targets[phone]
        variance = innovation
    else:
        mean = targets[phone]
        variance = innovation*np.exp(shared)
    residual = mean-observations
    value = .5*np.sum(mass[:, None]*(residual**2/variance+np.log(2*np.pi*variance)))
    derivative = mass[:, None]*residual/variance
    gt = np.zeros_like(targets)
    np.add.at(gt, phone, derivative*(1-shared if dynamic else 1))
    gs = (derivative*(before-targets[phone])).sum(0) if dynamic else (
        .5*mass[:, None]*(1-residual**2/variance)).sum(0)
    # Same prior unit on every free parameter in both models. Target pooling is
    # fixed before EM using phonetic groups, never the practice vocabulary.
    value += .5*shrinkage*np.sum((targets-pool)**2) + .5*shrinkage*np.sum(shared**2)
    gt += shrinkage*(targets-pool); gs += shrinkage*shared
    return float(value), np.r_[gt.ravel(), gs]


def fit(rows, tokens, shrinkage, dynamic, epochs=6):
    n = len(tokens)+1
    allx = np.concatenate([r['x'] for r in rows])
    mean = allx.mean(0); scale = np.maximum(allx.std(0), .1)
    normalized = [dict(r, x=(r['x']-mean)/scale) for r in rows]
    differences = np.concatenate([np.diff(r['x'], axis=0) for r in normalized])
    innovation = np.maximum(np.mean(differences**2, axis=0), .05)
    count = np.zeros(n); sums = np.zeros((n, DIMENSIONS))
    for r in normalized:
        uniform = np.minimum(np.arange(len(r['x']))*len(r['ids'])//len(r['x']), len(r['ids'])-1)
        phone = np.asarray(r['ids'])[uniform]
        np.add.at(sums, phone, r['x']); np.add.at(count, phone, 1)
    pool = np.zeros_like(sums)
    attrs = [attributes(p) for p in '_'+tokens]
    for p in range(1, n):
        members = [q for q in range(1, n) if attrs[q] == attrs[p]]
        pool[p] = sums[members].sum(0)/(count[members].sum()+shrinkage)
    targets = (sums+shrinkage*pool)/(count[:, None]+shrinkage)
    shared = np.zeros(DIMENSIONS)
    model = dict(targets=targets.tolist(), shared=shared.tolist(), counts=count.tolist(), innovation=innovation.tolist())
    history = []
    for epoch in range(epochs):
        observations, before, phone, masses, likelihoods = [], [], [], [], []
        for r in normalized:
            value, gamma = evaluate(model, r['x'], r['ids'], dynamic, True)
            likelihoods.append(value/len(r['x']))
            t, s = np.nonzero(gamma > 1e-9)
            observations.append(r['x'][t])
            prev = np.vstack([np.zeros(DIMENSIONS), r['x'][:-1]])
            before.append(prev[t]); phone.append(np.asarray(r['ids'])[s]); masses.append(gamma[t, s])
        x, prev, ids, mass = map(np.concatenate, (observations, before, phone, masses))
        result = minimize(objective, np.r_[np.asarray(model['targets']).ravel(), model['shared']],
            args=(x, prev, ids, mass, innovation, pool, shrinkage, dynamic), jac=True, method='L-BFGS-B',
            bounds=[(-30, 30)]*(n*DIMENSIONS)+([(0, .995)] if dynamic else [(-10, 10)])*DIMENSIONS,
            options={'maxiter': 60, 'ftol': 1e-8})
        if not np.isfinite(result.fun):
            raise ValueError('Non-finite training objective')
        model['targets'] = result.x[:n*DIMENSIONS].reshape(n, DIMENSIONS).tolist()
        model['shared'] = result.x[n*DIMENSIONS:].tolist()
        history.append(dict(epoch=epoch, logP=float(np.mean(likelihoods)), optimizerSuccess=bool(result.success)))
    return model, mean, scale, history


def prepare(raw, tokens, excluded):
    out = []
    for r in raw:
        if r.get('source') != 'help-train' or r.get('kind') != 'none' or r.get('split') != 'train' or (
                r.get('excluded') is not False or r.get('speaker') in excluded):
            raise ValueError('Only none-labelled non-excluded help-train training speakers may fit this scorer')
        x = np.asarray(r['features'], dtype=float)
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        if not r.get('speaker') or x.ndim != 2 or x.shape[1] != DIMENSIONS or not np.isfinite(x).all() or (
                not ids or len(x) < len(ids) or len(x) > 8192 or not np.allclose(np.exp(x).sum(1), 1, atol=1e-5)):
            raise ValueError('Invalid normalized observations or impossible complete pronunciation')
        out.append(dict(x=x, ids=ids, speaker=r['speaker']))
    if len({r['speaker'] for r in out}) < 3:
        raise ValueError('At least three eligible training speakers are required')
    return out


def validation(model, mean, scale, rows, dynamic):
    by_speaker = {}
    for r in rows:
        by_speaker.setdefault(r['speaker'], []).append(evaluate(model, (r['x']-mean)/scale, r['ids'], dynamic)/len(r['x']))
    return float(np.mean([np.mean(values) for values in by_speaker.values()]))


def main():
    parser = argparse.ArgumentParser()
    for name in ['input', 'out', 'tokens', 'inventory', 'excluded-speakers']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = prepare([json.loads(line) for line in Path(args.input).open()], args.tokens, excluded)
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    results = []
    for shrinkage in [8, 32, 128]:
        values = []
        for fold in range(3):
            train = [r for r in rows if folds[r['speaker']] != fold]
            held = [r for r in rows if folds[r['speaker']] == fold]
            pair = []
            for dynamic in [True, False]:
                model, mean, scale, _ = fit(train, args.tokens, shrinkage, dynamic)
                pair.append(validation(model, mean, scale, held, dynamic))
            values.append(pair)
            print(json.dumps(dict(shrinkage=shrinkage, fold=fold, logP=pair)), flush=True)
        results.append(dict(shrinkage=shrinkage, folds=values, meanLogP=float(np.mean(values))))
    selected = max(results, key=lambda r: (r['meanLogP'], r['shrinkage']))
    dynamic, mean, scale, dh = fit(rows, args.tokens, selected['shrinkage'], True)
    control, cm, cs, ch = fit(rows, args.tokens, selected['shrinkage'], False)
    if not np.array_equal(cm, mean) or not np.array_equal(cs, scale) or dynamic['innovation'] != control['innovation']:
        raise ValueError('Matched controls have different feature normalization')
    model = dict(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40, observationMs=10,
        dimensions=DIMENSIONS, maxFrames=8192, mean=mean.tolist(), scale=scale.tolist(), innovation=dynamic.pop('innovation'),
        dynamic=dynamic, control={k: v for k, v in control.items() if k != 'innovation'})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    asset = out/'rate-dynamics.json'; asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size,
        sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'rate-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), speakerFolds=folds, excludedSpeakers=sorted(excluded),
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(), validation=results, selected=selected,
        dynamicHistory=dh, controlHistory=ch, asset=descriptor, parametersPerArm=(len(args.tokens)+2)*DIMENSIONS,
        timingPrior='uniform over complete nonempty phone segmentations')
    (out/'rate-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

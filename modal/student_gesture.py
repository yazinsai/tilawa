"""CPU EM for normalized overlapping phone gestures and a matched local control.

Inputs are SDK spectral observations from none-labelled help-train training
speakers only. Boundaries are latent, not taken from listener phone spikes.
No practice inputs or script-corrected transcripts may enter fitting/selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

DIMENSIONS = 20
PHASES = 3


def sources(ids, overlap):
    states = np.array([p*3+k for p in ids for k in range(3)])
    left = np.array([0 if i == 0 else ids[i-1]*3+2 if overlap else p*3
                     for i, p in enumerate(ids) for k in range(3)])
    right = np.array([0 if i == len(ids)-1 else ids[i+1]*3 if overlap else p*3+2
                      for i, p in enumerate(ids) for k in range(3)])
    phase = np.tile(np.arange(3)/2, len(ids))
    return states, left, right, phase


def mixtures(spectra, weights, state, left, right, phase):
    a, b = weights[0]*(1-phase), weights[1]*phase
    return (spectra[state]+a[:, None]*spectra[left]+b[:, None]*spectra[right])/(1+a+b)[:, None]


def evaluate(model, x, ids, overlap, posterior=False):
    state, left, right, phase = sources(ids, overlap)
    tmax, n = len(x), len(state)
    if tmax < n:
        raise ValueError('Impossible complete pronunciation')
    mean = np.log(mixtures(np.asarray(model['spectra']), model['weights'], state, left, right, phase))
    var = np.asarray(model['variances'])[state]
    inv = 1/var
    e = -.5*(x*x @ inv.T-2*x @ (mean*inv).T+(mean*mean*inv+np.log(2*np.pi*var)).sum(1))
    stay = np.log(np.asarray(model['stay'])[state])
    advance = np.log1p(-np.asarray(model['stay'])[state])
    alpha = np.full((tmax, n), -np.inf)
    prior = np.full(n, -np.inf)
    alpha[0, 0], prior[0] = e[0, 0], 0
    for t in range(1, tmax):
        alpha[t] = alpha[t-1]+stay
        alpha[t, 1:] = np.logaddexp(alpha[t, 1:], alpha[t-1, :-1]+advance[:-1])
        alpha[t] += e[t]
        following = prior+stay
        following[1:] = np.logaddexp(following[1:], prior[:-1]+advance[:-1])
        prior = following
    likelihood, timing = alpha[-1, -1], prior[-1]
    if not posterior:
        return float(likelihood-timing)
    beta = np.full_like(alpha, -np.inf)
    beta[-1, -1] = 0
    transitions = np.zeros((len(model['stay']), 2))
    for t in range(tmax-2, -1, -1):
        following = e[t+1]+beta[t+1]
        beta[t] = stay+following
        beta[t, :-1] = np.logaddexp(beta[t, :-1], advance[:-1]+following[1:])
        np.add.at(transitions[:, 0], state, np.exp(alpha[t]+stay+following-likelihood))
        np.add.at(transitions[:, 1], state[:-1], np.exp(alpha[t, :-1]+advance[:-1]+following[1:]-likelihood))
    gamma = np.exp(alpha+beta-likelihood)
    return float(likelihood-timing), gamma, transitions


def attributes(p):
    return next((name for chars, name in [('َُِاۦۥ', 'vowel'), ('منں۾', 'nasal'),
        ('بتدطكقءڇ', 'stop'), ('ثذفسشصزظخغحعه', 'fricative'), ('رلوي', 'approximant')]
        if p in chars), 'other')


def objective(params, design, x, mass, variance, target, shrinkage):
    spectra = softmax(params[:-2].reshape(-1, DIMENSIONS), axis=1)
    weights = np.exp(params[-2:])
    state, left, right, phase = design.T
    state, left, right = state.astype(int), left.astype(int), right.astype(int)
    a, b = weights[0]*(1-phase), weights[1]*phase
    norm = (1+a+b)[:, None]
    prediction = (spectra[state]+a[:, None]*spectra[left]+b[:, None]*spectra[right])/norm
    residual = np.log(prediction)-x
    var = variance[state]
    value = .5*np.sum(mass[:, None]*(residual**2/var+np.log(2*np.pi*var)))
    g = mass[:, None]*residual/var/prediction
    gs = np.zeros_like(spectra)
    for s, weight in [(state, 1/norm), (left, a[:, None]/norm), (right, b[:, None]/norm)]:
        np.add.at(gs, s, g*weight)
    gl = spectra*(gs-(gs*spectra).sum(1, keepdims=True))
    value -= shrinkage*np.sum(target*np.log(spectra))
    gl += shrinkage*(spectra-target)
    ga = np.sum(g*(spectra[left]-prediction)/norm*(a[:, None]))
    gb = np.sum(g*(spectra[right]-prediction)/norm*(b[:, None]))
    # Identical log-normal regularization in both models. The normalized kernels
    # are shared across every phone: no candidate-specific explanation weights.
    kernel_penalty = params[-2:]+np.log(4)
    value += .5*np.sum(kernel_penalty**2)
    return float(value), np.r_[gl.ravel(), ga+kernel_penalty[0], gb+kernel_penalty[1]]


def fit(rows, tokens, shrinkage, overlap, epochs=5):
    n = (len(tokens)+1)*3
    sums, count = np.zeros((n, DIMENSIONS)), np.zeros(n)
    for r in rows:
        state = sources(r['ids'], overlap)[0]
        uniform = np.minimum(np.arange(len(r['x']))*len(state)//len(r['x']), len(state)-1)
        np.add.at(sums, state[uniform], np.exp(r['x']))
        np.add.at(count, state[uniform], 1)
    global_spectrum = sums.sum(0)/count.sum()
    attrs = [attributes(p) for p in '_'+tokens]
    target = np.zeros_like(sums)
    for p in range(len(attrs)):
        for phase in range(3):
            pool = [q*3+phase for q in range(1, len(attrs)) if attrs[q] == attrs[p]]
            target[p*3+phase] = (sums[pool].sum(0)+shrinkage*global_spectrum)/(count[pool].sum()+shrinkage)
    target[:3] = global_spectrum
    target = np.maximum(target, 1e-9); target /= target.sum(1, keepdims=True)
    spectra = (sums+shrinkage*target)/(count[:, None]+shrinkage)
    spectra = np.maximum(spectra, 1e-9); spectra /= spectra.sum(1, keepdims=True)
    model = dict(spectra=spectra.tolist(), variances=np.ones_like(spectra).tolist(),
                 stay=np.full(n, .8).tolist(), counts=count.tolist(), weights=[.25, .25])
    history = []
    for epoch in range(epochs):
        designs, observations, masses, likelihoods = [], [], [], []
        transitions = np.zeros((n, 2))
        for r in rows:
            value, gamma, tc = evaluate(model, r['x'], r['ids'], overlap, True)
            transitions += tc
            likelihoods.append(value/len(r['x']))
            t, s = np.nonzero(gamma > 1e-8)
            design = np.column_stack(sources(r['ids'], overlap))
            designs.append(design[s]); observations.append(r['x'][t]); masses.append(gamma[t, s])
        design, x, mass = np.concatenate(designs), np.concatenate(observations), np.concatenate(masses)
        params = np.r_[np.log(model['spectra']).ravel(), np.log(model['weights'])]
        optimized = minimize(objective, params,
            args=(design, x, mass, np.asarray(model['variances']), target, shrinkage),
            method='L-BFGS-B', jac=True, options={'maxiter': 35, 'ftol': 1e-7},
            bounds=[(-25, 5)]*(n*DIMENSIONS)+[(-6, 2), (-6, 2)])
        if not np.isfinite(optimized.fun):
            raise ValueError('Non-finite training objective')
        spectra, weights = softmax(optimized.x[:-2].reshape(n, DIMENSIONS), axis=1), np.exp(optimized.x[-2:])
        state, left, right, phase = design.T
        state, left, right = state.astype(int), left.astype(int), right.astype(int)
        residual = x-np.log(mixtures(spectra, weights, state, left, right, phase))
        seconds, count = np.zeros_like(spectra), np.zeros(n)
        np.add.at(seconds, state, mass[:, None]*residual**2)
        np.add.at(count, state, mass)
        global_var = seconds.sum(0)/count.sum()
        var = np.maximum((seconds+shrinkage*global_var)/(count[:, None]+shrinkage), .05)
        model.update(spectra=spectra.tolist(), weights=weights.tolist(), variances=var.tolist(),
                     counts=count.tolist())
        history.append(dict(epoch=epoch, meanLogP=float(np.mean(likelihoods)), weights=weights.tolist(),
                            optimizerSuccess=bool(optimized.success)))
    return model, history


def prepare(raw, tokens, excluded):
    out = []
    for r in raw:
        if r.get('source') != 'help-train' or r.get('kind') != 'none' or r.get('split') != 'train' or (
                r.get('excluded') is not False or r.get('speaker') in excluded):
            raise ValueError('Only none-labelled, non-excluded help-train training speakers may fit this scorer')
        x = np.asarray(r['features'], dtype=float)
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        if not r.get('speaker') or x.ndim != 2 or x.shape[1] != DIMENSIONS or not np.isfinite(x).all() or (
                not ids or len(x) < len(ids)*3 or len(x) > 8192 or not np.allclose(np.exp(x).sum(1), 1, atol=1e-5)):
            raise ValueError('Invalid normalized observations or impossible complete pronunciation')
        out.append(dict(x=x, ids=ids, speaker=r['speaker']))
    if len({r['speaker'] for r in out}) < 3:
        raise ValueError('At least three eligible training speakers are required')
    return out


def validation(model, rows, overlap):
    by_speaker = {}
    for r in rows:
        by_speaker.setdefault(r['speaker'], []).append(evaluate(model, r['x'], r['ids'], overlap)/len(r['x']))
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
            for overlap in [True, False]:
                fitted, _ = fit(train, args.tokens, shrinkage, overlap)
                pair.append(validation(fitted, held, overlap))
            values.append(pair)
            print(json.dumps(dict(shrinkage=shrinkage, fold=fold, logP=pair)), flush=True)
        results.append(dict(shrinkage=shrinkage, folds=values, meanLogP=float(np.mean(values))))
    selected = max(results, key=lambda r: (r['meanLogP'], r['shrinkage']))
    # Freeze one regularizer for both matched arms using their average validation
    # density; refit on all eligible training speakers, never on practice cases.
    overlap, oh = fit(rows, args.tokens, selected['shrinkage'], True)
    control, ch = fit(rows, args.tokens, selected['shrinkage'], False)
    model = dict(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40, observationMs=10,
                 dimensions=DIMENSIONS, maxFrames=8192, overlap=overlap, control=control)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    asset = out/'gesture-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size,
                      sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'gesture-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), speakerFolds=folds, excludedSpeakers=sorted(excluded),
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(), validation=results,
        selected=selected, overlapHistory=oh, controlHistory=ch, asset=descriptor,
        parametersPerArm=sum(np.asarray(overlap[k]).size for k in ['spectra', 'variances', 'weights']),
        timingPrior='uniform over complete segmentations: fixed equal self-loops in every phase')
    (out/'gesture-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

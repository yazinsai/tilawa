"""CPU EM for separate phoneme and time-varying phonation states.

SDK observations only, from none-labelled help-train training speakers. Model
selection groups speakers, never recordings from the practice set. Both phone
boundaries and the three voice-quality states remain latent after initialization.
"""
import argparse
import hashlib
import json
from pathlib import Path


def chain(emission, transition, initial, posterior=False, switching=True):
    """Exact product-chain marginal with a completion-conditioned phone prior.

    All phone states are mandatory; equal self/advance probabilities make the
    conditional boundary prior uniform. The fixed control uses the same initial
    phonation prior and emissions, replacing only the switching matrix by I.
    """
    import numpy as np
    frames, states, zmax = emission.shape
    if frames < states:
        return -np.inf
    matrix = np.asarray(transition) if switching else np.eye(zmax)
    with np.errstate(divide='ignore'):
        logmatrix = np.log(matrix)
    loghalf = np.log(.5)
    alpha = np.full_like(emission, -np.inf)
    alpha[0, 0] = np.log(initial) + emission[0, 0]
    prior = np.full(states, -np.inf)
    prior[0] = 0
    for t in range(1, frames):
        phone = np.logaddexp(alpha[t-1], np.concatenate([np.full((1, zmax), -np.inf), alpha[t-1, :-1]])) + loghalf
        alpha[t] = np.logaddexp.reduce(phone[:, :, None] + logmatrix[None], axis=1) + emission[t]
        prior = np.logaddexp(prior, np.r_[-np.inf, prior[:-1]]) + loghalf
    logp = np.logaddexp.reduce(alpha[-1, -1])
    value = logp - prior[-1]
    if not posterior:
        return value
    beta = np.full_like(emission, -np.inf)
    beta[-1, -1] = 0
    for t in range(frames-2, -1, -1):
        following = beta[t+1] + emission[t+1]
        phone = np.logaddexp(following, np.concatenate([following[1:], np.full((1, zmax), -np.inf)])) + loghalf
        beta[t] = np.logaddexp.reduce(logmatrix[None] + phone[:, None, :], axis=2)
    gamma = np.exp(alpha + beta - logp)
    zcounts = np.zeros((zmax, zmax))
    for t in range(frames-1):
        following = beta[t+1] + emission[t+1]
        phone = np.logaddexp(following, np.concatenate([following[1:], np.full((1, zmax), -np.inf)])) + loghalf
        zcounts += np.exp(alpha[t, :, :, None] + logmatrix[None] + phone[:, None, :] - logp).sum(0)
    return value, gamma, zcounts


def densities(model, x, ids):
    import numpy as np
    mu = np.asarray(model['means'])[ids, None] + np.asarray(model['offsets'])[None]
    var = np.asarray(model['variances'])[ids, None]
    mel = -.5*(np.log(2*np.pi*var)[None] + (x[:, None, None, :20]-mu[None])**2/var[None]).sum(3)
    qm, qv = np.asarray(model['qualityMeans']), np.asarray(model['qualityVariances'])
    quality = -.5*(np.log(2*np.pi*qv)[None] + (x[:, None, 20:]-qm[None])**2/qv[None]).sum(2)
    return mel + quality[:, None]


def shared_phone_mean(p, counts, means, tokens):
    """Shrink sparse phones by broad manner, without any word or passage key."""
    import numpy as np
    groups = ['َُِاۦۥ', 'منں۾', 'بتدطكقءڇ', 'ثذفسشصزظخغحعه', 'رلوي']
    chars = next((g for g in groups if p and tokens[p-1] in g), '')
    pool = [q for q in range(1, len(tokens)+1) if tokens[q-1] in chars]
    if not pool:
        pool = list(range(len(tokens)+1))
    weights = counts[pool].sum(1)
    return (means[pool]*weights[:, None]).sum(0)/max(1, weights.sum())


def update(model, stats, tokens, shrinkage):
    import numpy as np
    counts, sums, seconds = stats
    mu, delta = np.asarray(model['means']), np.asarray(model['offsets'])
    # Weighted additive regression: one phone mean and one globally shared
    # phonation shift. No phone x phonation lookup can absorb a consonant edit.
    for _ in range(5):
        targets = np.array([shared_phone_mean(p, counts, mu, tokens) for p in range(len(mu))])
        mu = (sums[:, :, :20].sum(1) - (counts[:, :, None]*delta[None]).sum(1) + shrinkage*targets)/(counts.sum(1)[:, None]+shrinkage)
        delta = (sums[:, :, :20].sum(0)-(counts[:, :, None]*mu[:, None]).sum(0))/(counts.sum(0)[:, None]+shrinkage)
        # Fix the additive gauge, preserving all fitted emission means.
        shift = (np.asarray(model['initial'])[:, None]*delta).sum(0)
        delta -= shift
        mu += shift
    fitted = mu[:, None]+delta[None]
    residual = seconds[:, :, :20] - 2*fitted*sums[:, :, :20] + counts[:, :, None]*fitted**2
    global_var = np.maximum(residual.sum((0, 1))/max(1, counts.sum()), .05)
    variance = (residual.sum(1)+shrinkage*global_var)/(counts.sum(1)[:, None]+shrinkage)
    qc = counts.sum(0)
    qm = sums[:, :, 20:].sum(0)/(qc[:, None]+1e-12)
    qv = np.maximum(seconds[:, :, 20:].sum(0)/(qc[:, None]+1e-12)-qm**2, .05)
    model.update(means=mu.tolist(), offsets=delta.tolist(), variances=np.maximum(variance, .05).tolist(),
                 qualityMeans=qm.tolist(), qualityVariances=qv.tolist())


def fit(rows, tokens, shrinkage, persistence, epochs=10):
    import numpy as np
    raw = np.concatenate([r['x'] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .1)
    rows = [dict(r, x=(r['x']-center)/scale) for r in rows]
    n = len(tokens)+1
    # Phonation seeds use only voice-quality observations. Cluster names are
    # latent proxies, not supervised breathy/regular/creaky physiology labels.
    quality = (raw-center)[:, 20:]/scale[20:]
    qmeans = np.array([quality[np.argmin(quality[:, 1])], np.median(quality, 0), quality[np.argmax(quality[:, 2])]])
    for _ in range(12):
        z = ((quality[:, None]-qmeans[None])**2).sum(2).argmin(1)
        qmeans = np.array([quality[z==q].mean(0) if (z==q).any() else qmeans[q] for q in range(3)])
    def fresh():
        return np.zeros((n, 3)), np.zeros((n, 3, 23)), np.zeros((n, 3, 23))
    counts, sums, seconds = fresh()
    for r in rows:
        ids = [0]+r['ids']+[0]
        state = np.minimum(np.arange(len(r['x']))*len(ids)//len(r['x']), len(ids)-1)
        z = ((r['x'][:, None, 20:]-qmeans[None])**2).sum(2).argmin(1)
        for s, p in enumerate(ids):
            for q in range(3):
                x = r['x'][(state==s)&(z==q)]
                counts[p, q] += len(x)
                sums[p, q] += x.sum(0)
                seconds[p, q] += (x*x).sum(0)
    model = dict(center=center.tolist(), scale=scale.tolist(), means=np.zeros((n, 20)).tolist(), offsets=np.zeros((3, 20)).tolist(),
        variances=np.ones((n, 20)).tolist(), qualityMeans=qmeans.tolist(), qualityVariances=np.ones((3, 3)).tolist(),
        initial=[1/3]*3, transition=((np.ones((3, 3))+persistence*np.eye(3))/(3+persistence)).tolist(), stay=[.5]*n)
    update(model, (counts, sums, seconds), tokens, shrinkage)
    history = []
    for _ in range(epochs):
        counts, sums, seconds = fresh()
        zcounts, initial, values = np.zeros((3, 3)), np.zeros(3), []
        for r in rows:
            ids = [0]+r['ids']+[0]
            value, gamma, transitions = chain(densities(model, r['x'], ids), model['transition'], model['initial'], True)
            if not np.isfinite(value):
                raise ValueError('Nonfinite complete-sequence training likelihood')
            values.append(float(value)/len(r['x']))
            zcounts += transitions
            initial += gamma[0, 0]
            for s, p in enumerate(ids):
                counts[p] += gamma[:, s].sum(0)
                sums[p] += gamma[:, s].T @ r['x']
                seconds[p] += gamma[:, s].T @ (r['x']**2)
        model['initial'] = ((initial+1)/(initial.sum()+3)).tolist()
        zcounts += np.ones((3, 3))+persistence*np.eye(3)
        model['transition'] = (zcounts/zcounts.sum(1)[:, None]).tolist()
        update(model, (counts, sums, seconds), tokens, shrinkage)
        history.append(float(np.mean(values)))
    model['counts'] = counts.sum(1).tolist()
    return model, history


def validation(model, rows):
    import numpy as np
    speakers = {}
    for r in rows:
        x = (r['x']-model['center'])/model['scale']
        value = chain(densities(model, x, [0]+r['ids']+[0]), model['transition'], model['initial'])/len(x)
        speakers.setdefault(r['speaker'], []).append(float(value))
    return float(np.mean([np.mean(v) for v in speakers.values()]))


def prepare(rows, tokens, excluded):
    import numpy as np
    out = []
    for r in rows:
        if r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r['speaker'] in excluded:
            raise ValueError('Only none-labelled, non-excluded help-train training speakers are authorized')
        x = np.asarray(r['features'], dtype=float)
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        if x.ndim != 2 or x.shape[1] != 23 or not np.isfinite(x).all() or not ids or len(x) < len(ids)+2 or len(x)>4096:
            raise ValueError('Invalid SDK acoustic observations or incomplete pronunciation')
        out.append(dict(x=x, ids=ids, speaker=r['speaker']))
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
    for shrinkage in [8, 32, 128]:
        for persistence in [8, 32]:
            values = []
            for fold in range(3):
                model, _ = fit([r for r in rows if folds[r['speaker']] != fold], args.tokens, shrinkage, persistence)
                values.append(validation(model, [r for r in rows if folds[r['speaker']] == fold]))
            result = dict(shrinkage=shrinkage, persistence=persistence, foldLogP=values, meanLogP=sum(values)/3)
            results.append(result)
            print(json.dumps(result), flush=True)
    chosen = max(results, key=lambda r: (r['meanLogP'], r['shrinkage'], r['persistence']))
    model, history = fit(rows, args.tokens, chosen['shrinkage'], chosen['persistence'])
    model.update(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40, observationMs=10, dimensions=23, maxFrames=4096)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'phonation-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'phonation-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        speakerFolds=folds, excludedSpeakers=sorted(excluded), validation=results, selected=chosen, history=history, asset=descriptor)
    (out/'phonation-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

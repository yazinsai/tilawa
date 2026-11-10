"""CPU empirical-Bayes phone emissions on frozen SDK features.

Normal/inverse-Wishart posteriors share prior moments across manner, place and
voicing classes. Forward/backward EM marginalizes training phone boundaries;
posterior uncertainty is conditional on its fractional sufficient statistics.
Input accepts only none-labelled help-train training speakers, never practice.
"""
import argparse
import hashlib
import json
from pathlib import Path

DIMENSIONS = 8
PRIOR_STRENGTHS = (2, 8, 32)
EPOCHS = 12


def phone_classes(p):
    manner = next((name for chars, name in [("َُِاۦۥ", "vowel"), ("منں۾", "nasal"),
        ("بتدطكقءڇ", "stop"), ("ثذفسشصزظخغحعه", "fricative"), ("رلوي", "approximant")]
        if p in chars), "other")
    place = next((name for chars, name in [("بمفو", "labial"), ("ثذظ", "dental"),
        ("تدطسصزشنرل", "coronal"), ("يكقخغ", "dorsal"), ("ءهحع", "guttural")]
        if p in chars), "other")
    return manner, place, "voiced" if p in "بجدذزظعغرلمنوياَُِۦۥں۾" else "unvoiced"


def chain(e, stay, posterior=False):
    """Ordered complete phones, exact boundary marginal, normalized timing prior."""
    import numpy as np
    tmax, states = e.shape
    if tmax < states:
        raise ValueError('Complete pronunciation has too many phones')
    a = np.full_like(e, -np.inf)
    prior = np.full(states, -np.inf)
    a[0, 0], prior[0] = e[0, 0], 0
    selfp, step = np.log(stay), np.log1p(-stay)
    for t in range(1, tmax):
        a[t] = np.logaddexp(a[t-1]+selfp, np.r_[-np.inf, a[t-1, :-1]]+step)+e[t]
        prior = np.logaddexp(prior+selfp, np.r_[-np.inf, prior[:-1]]+step)
    value = a[-1, -1]-prior[-1]
    if not posterior:
        return float(value)
    b = np.full_like(e, -np.inf)
    b[-1, -1] = 0
    for t in range(tmax-2, -1, -1):
        b[t] = np.logaddexp(selfp+e[t+1]+b[t+1], np.r_[step+e[t+1, 1:]+b[t+1, 1:], -np.inf])
    return float(value), np.exp(a+b-a[-1, -1])


def covariance(p):
    import numpy as np
    return np.array(p['psi'])/(p['nu']-DIMENSIONS-1)


def emission(model, x, ids):
    import numpy as np
    columns = []
    for p in ids:
        posterior = model['phones'][p]
        cov = covariance(posterior)
        l = np.linalg.cholesky(cov)
        z = np.linalg.solve(l, (x-posterior['mean']).T)
        columns.append(-.5*(DIMENSIONS*np.log(2*np.pi)+2*np.log(l.diagonal()).sum()+(z*z).sum(0)))
    return np.array(columns).T


def update(stats, tokens, strength):
    """NIW update around a class-shared empirical prior; unseen phones remain unsupported."""
    import numpy as np
    count, sums, cross = stats
    total = max(count.sum(), 1)
    global_mu = sums.sum(0)/total
    global_second = cross.sum(0)/total
    global_cov = global_second-np.outer(global_mu, global_mu)+.05*np.eye(DIMENSIONS)
    classes = [phone_classes(p) for p in '_' + tokens]
    out = []
    for p in range(len(classes)):
        means, seconds = [], []
        for attribute in range(3):
            pool = [q for q in range(1, len(classes)) if q != p and classes[q][attribute] == classes[p][attribute]]
            n = count[pool].sum()
            means.append((sums[pool].sum(0)+strength*global_mu)/(n+strength))
            seconds.append((cross[pool].sum(0)+strength*(global_cov+np.outer(global_mu, global_mu)))/(n+strength))
        mu0 = np.mean(means, 0)
        cov0 = np.mean(seconds, 0)-np.outer(mu0, mu0)
        # Numerical positive-definiteness; the regularization is fixed before validation.
        cov0 = (cov0+cov0.T)/2+.05*np.eye(DIMENSIONS)
        n = count[p]
        empirical = sums[p]/n if n else mu0
        scatter = cross[p]-n*np.outer(empirical, empirical)
        delta = empirical-mu0
        kappa = strength+n
        nu0 = DIMENSIONS+2+strength
        psi = cov0*(nu0-DIMENSIONS-1)+scatter+(strength*n/kappa)*np.outer(delta, delta)
        psi = (psi+psi.T)/2
        np.linalg.cholesky(psi)
        out.append(dict(mean=((strength*mu0+sums[p])/kappa).tolist(), kappa=float(kappa),
            nu=float(nu0+n), psi=psi.tolist(), count=float(n)))
    return out


def fit(rows, tokens, strength, epochs=EPOCHS):
    import numpy as np
    raw = np.concatenate([r['x'] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .1)
    rows = [dict(r, x=(r['x']-center)/scale) for r in rows]
    vocab = len(tokens)+1
    def fresh():
        return np.zeros(vocab), np.zeros((vocab, DIMENSIONS)), np.zeros((vocab, DIMENSIONS, DIMENSIONS))
    def add(stats, p, x, weights):
        stats[0][p] += weights.sum()
        stats[1][p] += weights@x
        stats[2][p] += (x*weights[:, None]).T@x
    stats = fresh()
    # Uniform initialization is not a forced alignment; later updates sum every boundary.
    for r in rows:
        assigned = np.minimum(np.arange(len(r['x']))*len(r['ids'])//len(r['x']), len(r['ids'])-1)
        for s, p in enumerate(r['ids']):
            add(stats, p, r['x'], (assigned == s).astype(float))
    stay = .8
    model = dict(center=center.tolist(), scale=scale.tolist(), phones=update(stats, tokens, strength), stay=stay)
    history = []
    for _ in range(epochs):
        stats, values = fresh(), []
        for r in rows:
            value, gamma = chain(emission(model, r['x'], r['ids']), stay, True)
            values.append(value/len(r['x']))
            for s, p in enumerate(r['ids']):
                add(stats, p, r['x'], gamma[:, s])
        model['phones'] = update(stats, tokens, strength)
        history.append(float(np.mean(values)))
    return model, history


def predictive_validation(model, rows, seed=1729, draws=128):
    """Parameter draws shared throughout each complete pronunciation, as in SDK inference."""
    import numpy as np
    from scipy.stats import invwishart
    by_speaker = {}
    rng = np.random.default_rng(seed)
    for r in rows:
        x = (r['x']-model['center'])/model['scale']
        scores = []
        for _ in range(draws):
            table = {}
            for p in sorted(set(r['ids'])):
                posterior = model['phones'][p]
                cov = invwishart.rvs(df=posterior['nu'], scale=posterior['psi'], random_state=rng)
                mu = rng.multivariate_normal(posterior['mean'], cov/posterior['kappa'])
                table[p] = dict(mean=mu, nu=DIMENSIONS+2, psi=cov)
            temporary = dict(model, phones=[table.get(p, {}) for p in range(len(model['phones']))])
            scores.append(chain(emission(temporary, x, r['ids']), model['stay']))
        score = np.logaddexp.reduce(scores)-np.log(draws)
        by_speaker.setdefault(r['speaker'], []).append(float(score)/len(x))
    return float(np.mean([np.mean(v) for v in by_speaker.values()]))


def prepare(rows, tokens, excluded):
    import numpy as np
    out = []
    for r in rows:
        if r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r.get('source') != 'help-train' or r['speaker'] in excluded:
            raise ValueError('Only none-labelled, non-excluded help-train training speakers may be fitted')
        x = np.asarray(r['features'], dtype=float)
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        if x.ndim != 2 or x.shape[1] != DIMENSIONS or not np.isfinite(x).all() or not ids or len(x) < len(ids):
            raise ValueError('Invalid frozen observations or incomplete pronunciation')
        out.append(dict(x=x, ids=ids, speaker=r['speaker']))
    if len({r['speaker'] for r in out}) < 3:
        raise ValueError('At least three eligible training speakers required')
    return out


def main():
    parser = argparse.ArgumentParser()
    for key in ('input', 'out', 'tokens', 'inventory', 'excluded-speakers'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = prepare([json.loads(line) for line in Path(args.input).open()], args.tokens, excluded)
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    results = []
    for strength in PRIOR_STRENGTHS:
        values = []
        for fold in range(3):
            model, _ = fit([r for r in rows if folds[r['speaker']] != fold], args.tokens, strength)
            values.append(predictive_validation(model, [r for r in rows if folds[r['speaker']] == fold]))
        results.append(dict(strength=strength, foldLogP=values, meanLogP=sum(values)/3))
    selected = max(results, key=lambda r: (r['meanLogP'], r['strength']))
    model, history = fit(rows, args.tokens, selected['strength'])
    model.update(schema=1, inventory=args.inventory, tokens=args.tokens, dimensions=DIMENSIONS, frameMs=40,
        observationMs=10, draws=128, maxFrames=2048)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'predictive-phone-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'predictive-phone-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        excludedSpeakers=sorted(excluded), speakerFolds=folds, validation=results, selected=selected, history=history, asset=descriptor)
    (out/'predictive-phone-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

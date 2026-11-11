"""CPU waveform-regularity / scale-pooled energy comparison.

Consumes SDK observations from none-labelled help-train training speakers only.
All phone boundaries are marginalized by forward/backward EM. Selection sees
speaker-disjoint training folds, never warning windows or practice labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

SCALES = {"fine": [2, 4, 8, 16, 32, 64, 128], "wide": [4, 8, 16, 32, 64, 128, 256]}


def chain(e, stay, posterior=False):
    import numpy as np
    tmax, states = e.shape
    a = np.full_like(e, -np.inf)
    prior = np.full(states, -np.inf)
    a[0, 0], prior[0] = e[0, 0], 0
    ls, la = np.log(stay), np.log1p(-stay)
    for t in range(1, tmax):
        a[t] = np.logaddexp(a[t-1]+ls, np.r_[-np.inf, a[t-1, :-1]]+la)+e[t]
        prior = np.logaddexp(prior+ls, np.r_[-np.inf, prior[:-1]]+la)
    value = a[-1, -1]-prior[-1]
    if not posterior:
        return value
    b = np.full_like(e, -np.inf)
    b[-1, -1] = 0
    for t in range(tmax-2, -1, -1):
        b[t] = np.logaddexp(ls+e[t+1]+b[t+1], np.r_[e[t+1, 1:]+b[t+1, 1:]+la, -np.inf])
    return value, np.exp(a+b-a[-1, -1])


def emissions(model, x, ids):
    import numpy as np
    mu, var = np.asarray(model['means'])[ids], np.asarray(model['variances'])[ids]
    return -.5*(np.log(2*np.pi*var)[None]+(x[:, None]-mu[None])**2/var[None]).sum(2)


def phonetics(p):
    manner = next((name for chars, name in [("َُِاۦۥ", "vowel"), ("منں۾", "nasal"),
        ("بتدطكقءڇ", "stop"), ("ثذفسشصزظخغحعه", "fricative"), ("رلوي", "approximant")]
        if p in chars), "other")
    place = next((name for chars, name in [("بمفو", "labial"), ("ثذظ", "dental"),
        ("تدطسصزشنرل", "coronal"), ("يكقخغ", "dorsal"), ("ءهحع", "guttural")]
        if p in chars), "other")
    return [manner, place]


def update(counts, sums, seconds, tokens, shrinkage):
    import numpy as np
    total = max(counts.sum(), 1)
    global_mu, global_second = sums.sum(0)/total, seconds.sum(0)/total
    attributes = [phonetics(p) for p in "_"+tokens]
    means, variances = [], []
    for p in range(len(attributes)):
        mus, moments = [], []
        for f in range(2):
            pool = [q for q in range(1, len(attributes)) if attributes[q][f] == attributes[p][f]]
            denom = counts[pool].sum()+shrinkage
            mus.append((sums[pool].sum(0)+shrinkage*global_mu)/denom)
            moments.append((seconds[pool].sum(0)+shrinkage*global_second)/denom)
        denominator = counts[p]+shrinkage
        mu = (sums[p]+shrinkage*np.mean(mus, 0))/denominator
        second = (seconds[p]+shrinkage*np.mean(moments, 0))/denominator
        means.append(mu.tolist())
        variances.append(np.maximum(second-mu**2, .05).tolist())
    return means, variances


def fit(rows, tokens, arm, geometry, shrinkage, stay, epochs=8):
    import numpy as np
    raw = np.concatenate([r[geometry][arm] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .1)
    data = [(r['ids'], (r[geometry][arm]-center)/scale) for r in rows]
    n = len(tokens)+1
    def fresh():
        return np.zeros(n), np.zeros((n, 12)), np.zeros((n, 12))
    counts, sums, seconds = fresh()
    for ids, x in data:
        state = np.minimum(np.arange(len(x))*len(ids)//len(x), len(ids)-1)
        for s, p in enumerate(ids):
            part = x[state == s]
            counts[p] += len(part)
            sums[p] += part.sum(0)
            seconds[p] += (part**2).sum(0)
    means, variances = update(counts, sums, seconds, tokens, shrinkage)
    model = dict(center=center.tolist(), scale=scale.tolist(), means=means, variances=variances)
    history = []
    for _ in range(epochs):
        counts, sums, seconds = fresh()
        values = []
        for ids, x in data:
            value, gamma = chain(emissions(model, x, ids), stay, True)
            values.append(float(value)/len(x))
            # Recording identities remain grouped in validation, not treated as speakers per frame.
            weighted_sum, weighted_second = gamma.T @ x, gamma.T @ (x**2)
            np.add.at(counts, ids, gamma.sum(0))
            np.add.at(sums, ids, weighted_sum)
            np.add.at(seconds, ids, weighted_second)
        model['means'], model['variances'] = update(counts, sums, seconds, tokens, shrinkage)
        history.append(float(np.mean(values)))
    model['counts'] = counts.tolist()
    return model, history


def timing(rows):
    # Common normalized geometric duration law, estimated only from whole training recordings.
    return 1-sum(len(r['ids']) for r in rows)/sum(len(r['fine']['energy']) for r in rows)


def validate(model, rows, arm, geometry, stay):
    import numpy as np
    by_speaker = {}
    for r in rows:
        x = (r[geometry][arm]-model['center'])/model['scale']
        value = float(chain(emissions(model, x, r['ids']), stay))/len(x)
        by_speaker.setdefault(r['speaker'], []).append(value)
    return float(np.mean([np.mean(v) for v in by_speaker.values()]))


def prepare(records, tokens, excluded):
    import numpy as np
    rows = []
    for r in records:
        if r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r['speaker'] in excluded:
            raise ValueError('Only non-excluded none-labelled training speakers may be fitted')
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        row = dict(ids=ids, speaker=r['speaker'])
        for geometry in SCALES:
            row[geometry] = {arm: np.asarray([f[arm] for f in r['features'][geometry]], dtype=float)
                for arm in ['regularity', 'energy']}
            for x in row[geometry].values():
                if x.ndim != 2 or x.shape[1] != 12 or not np.isfinite(x).all() or not ids or len(x) <= len(ids):
                    raise ValueError('Invalid SDK observations or impossible complete pronunciation')
        rows.append(row)
    if len({r['speaker'] for r in rows}) < 3:
        raise ValueError('At least three eligible training speakers required')
    return rows


def main():
    p = argparse.ArgumentParser()
    for key in ['input', 'out', 'tokens', 'inventory', 'excluded-speakers']:
        p.add_argument('--'+key, required=True)
    args = p.parse_args()
    source = Path(args.input)
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = prepare([json.loads(line) for line in source.open()], args.tokens, excluded)
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    results = []
    for geometry in SCALES:
        for shrinkage in [16, 64]:
            scores = []
            for fold in range(3):
                train = [r for r in rows if folds[r['speaker']] != fold]
                valid = [r for r in rows if folds[r['speaker']] == fold]
                stay = timing(train)
                scores.append({})
                for arm in ['regularity', 'energy']:
                    model, _ = fit(train, args.tokens, arm, geometry, shrinkage, stay)
                    scores[-1][arm] = validate(model, valid, arm, geometry, stay)
            # Equal-weight predictive likelihood; no warning labels choose scales or regularization.
            results.append(dict(geometry=geometry, shrinkage=shrinkage, folds=scores,
                meanLogP=sum(sum(f.values())/2 for f in scores)/3))
    chosen = max(results, key=lambda r: (r['meanLogP'], r['shrinkage'], r['geometry']))
    stay = timing(rows)
    bundle = dict(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40, observationMs=10,
        dimensions=12, scales=SCALES[chosen['geometry']], stay=stay, maxFrames=2048)
    histories = {}
    for arm in ['regularity', 'energy']:
        bundle[arm], histories[arm] = fit(rows, args.tokens, arm, chosen['geometry'], chosen['shrinkage'], stay)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'regularity-emissions.json'
    asset.write_text(json.dumps(bundle, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'regularity-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), speakerFolds=folds, excludedSpeakers=sorted(excluded),
        inputSha256=hashlib.sha256(source.read_bytes()).hexdigest(), validation=results, selected=chosen,
        history=histories, asset=descriptor)
    (out/'regularity-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

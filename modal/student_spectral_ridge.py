"""CPU ridge-connectivity emissions with latent complete-phone boundaries.

The SDK produces both training and inference observations. Only none-labelled
help-train training speakers may supply transcripts. Geometry and shrinkage are
selected by speaker-grouped training-only likelihood, never engine incidents.
"""
import argparse
import hashlib
import json
from pathlib import Path

DIMENSIONS = 56


def phone_classes(p):
    manner = next((name for chars, name in [("َُِاۦۥ", "vowel"), ("منں۾", "nasal"),
        ("بتدطكقءڇ", "stop"), ("ثذفسشصزظخغحعه", "fricative"), ("رلوي", "approximant")]
        if p in chars), "other")
    place = next((name for chars, name in [("بمفو", "labial"), ("ثذظ", "dental"),
        ("تدطسصزشنرل", "coronal"), ("يكقخغ", "dorsal"), ("ءهحع", "guttural")]
        if p in chars), "other")
    return manner, place, "voiced" if p in "بجدذزظعغرلمنوياَُِۦۥں۾" else "unvoiced"


def chain(e, stay, posterior=False):
    """Sum every complete monotonic segmentation; condition timing on completion."""
    import numpy as np
    tmax, states = e.shape
    if not states or tmax < states:
        raise ValueError('Impossible complete pronunciation')
    a = np.full_like(e, -np.inf)
    prior = np.full(states, -np.inf)
    a[0, 0], prior[0] = e[0, 0], 0
    selfp, step = np.log(stay), np.log1p(-stay)
    for t in range(1, tmax):
        a[t] = np.logaddexp(a[t-1]+selfp, np.r_[-np.inf, a[t-1, :-1]]+step)+e[t]
        prior = np.logaddexp(prior+selfp, np.r_[-np.inf, prior[:-1]]+step)
    value = float(a[-1, -1]-prior[-1])
    if not posterior:
        return value
    b = np.full_like(e, -np.inf)
    b[-1, -1] = 0
    for t in range(tmax-2, -1, -1):
        b[t] = np.logaddexp(selfp+e[t+1]+b[t+1], np.r_[step+e[t+1, 1:]+b[t+1, 1:], -np.inf])
    return value, np.exp(a+b-a[-1, -1])


def emissions(model, x, ids, connected=True):
    import numpy as np
    dims = DIMENSIONS if connected else 32
    mu, var = np.array(model['means'])[ids, :dims], np.array(model['variances'])[ids, :dims]
    return -.5*(np.log(2*np.pi*var)[None]+(x[:, None, :dims]-mu[None])**2/var[None]).sum(2)


def update(stats, tokens, strength):
    import numpy as np
    counts, sums, seconds = stats
    total = max(counts.sum(), 1)
    global_mu, global_second = sums.sum(0)/total, seconds.sum(0)/total
    classes = [phone_classes(p) for p in '_' + tokens]
    means, variances = [], []
    for p in range(len(classes)):
        shared_mu, shared_second = [], []
        for f in range(3):
            pool = [q for q in range(1, len(classes)) if q != p and classes[q][f] == classes[p][f]]
            n = counts[pool].sum()
            shared_mu.append((sums[pool].sum(0)+strength*global_mu)/(n+strength))
            shared_second.append((seconds[pool].sum(0)+strength*global_second)/(n+strength))
        denominator = counts[p]+strength
        mu = (sums[p]+strength*np.mean(shared_mu, 0))/denominator
        second = (seconds[p]+strength*np.mean(shared_second, 0))/denominator
        means.append(mu.tolist())
        variances.append(np.maximum(.05, second-mu**2).tolist())
    return means, variances


def fit(rows, tokens, strength, epochs=12):
    import numpy as np
    raw = np.concatenate([r['x'] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .1)
    rows = [dict(r, x=(r['x']-center)/scale) for r in rows]
    vocab = len(tokens)+1
    def fresh():
        return np.zeros(vocab), np.zeros((vocab, DIMENSIONS)), np.zeros((vocab, DIMENSIONS))
    def add(stats, p, x, weights):
        stats[0][p] += weights.sum()
        stats[1][p] += weights@x
        stats[2][p] += weights@(x*x)
    stats = fresh()
    # Equal-duration initialization is replaced by all-boundary EM, not frozen alignment.
    for r in rows:
        assigned = np.minimum(np.arange(len(r['x']))*len(r['ids'])//len(r['x']), len(r['ids'])-1)
        for s, p in enumerate(r['ids']):
            add(stats, p, r['x'], (assigned == s).astype(float))
    means, variances = update(stats, tokens, strength)
    model = dict(center=center.tolist(), scale=scale.tolist(), means=means, variances=variances, stay=.8)
    history = []
    for _ in range(epochs):
        stats, values = fresh(), []
        for r in rows:
            value, gamma = chain(emissions(model, r['x'], r['ids']), model['stay'], True)
            values.append(value/len(r['x']))
            for s, p in enumerate(r['ids']):
                add(stats, p, r['x'], gamma[:, s])
        model['means'], model['variances'] = update(stats, tokens, strength)
        history.append(float(np.mean(values)))
    model['counts'] = stats[0].tolist()
    return model, history


def validation(model, rows):
    import numpy as np
    by_speaker = {}
    for r in rows:
        x = (r['x']-model['center'])/model['scale']
        score = chain(emissions(model, x, r['ids']), model['stay'])/len(x)
        by_speaker.setdefault(r['speaker'], []).append(score)
    return float(np.mean([np.mean(v) for v in by_speaker.values()]))


def prepare(rows, tokens, excluded, radius):
    import numpy as np
    out = []
    for r in rows:
        if r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r.get('source') != 'help-train' or r['speaker'] in excluded:
            raise ValueError('Only none-labelled non-excluded help-train training speakers may be fitted')
        x = np.asarray(r['featuresByRadius'][str(radius)], dtype=float)
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
    raw = [json.loads(line) for line in Path(args.input).open()]
    data = {radius: prepare(raw, args.tokens, excluded, radius) for radius in (1, 3)}
    speakers = sorted({r['speaker'] for r in data[1]}, key=lambda s: hashlib.sha256(s.encode()).digest())
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    results = []
    for radius, rows in data.items():
        for strength in (8, 32, 128):
            values = []
            for fold in range(3):
                model, _ = fit([r for r in rows if folds[r['speaker']] != fold], args.tokens, strength)
                values.append(validation(model, [r for r in rows if folds[r['speaker']] == fold]))
            results.append(dict(radius=radius, strength=strength, foldLogP=values, meanLogP=sum(values)/3))
    selected = max(results, key=lambda r: (r['meanLogP'], r['strength'], -r['radius']))
    model, history = fit(data[selected['radius']], args.tokens, selected['strength'])
    model.update(schema=1, inventory=args.inventory, tokens=args.tokens, dimensions=DIMENSIONS, controlDimensions=32,
        radius=selected['radius'], frameMs=40, observationMs=10, maxFrames=2048)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'ridge-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'ridge-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(raw), speakers=len(speakers), inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        excludedSpeakers=sorted(excluded), speakerFolds=folds, validation=results, selected=selected, history=history, asset=descriptor)
    (out/'ridge-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

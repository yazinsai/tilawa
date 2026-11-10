"""Compact place / manner / voicing emissions, fitted with shared latent boundaries.

Accept only none-labelled training speakers. SDK extraction supplies native
10 ms observations. Selection uses speaker-disjoint training folds, never
practice windows. The independent-timing control reuses the final emissions.
"""
import argparse
import hashlib
import json
from pathlib import Path


def attributes(p):
    if p == '_':
        return ['background'] * 3
    manner = next((name for chars, name in [
        ('َُِاۦۥ', 'vowel'), ('منں۾', 'nasal'), ('بتدطكقءڇ', 'stop'), ('ج', 'affricate'),
        ('ثذفسشصزظخغحعه', 'fricative'), ('رلوي', 'approximant')]
        if p in chars), 'other:' + p)
    place = next((name for chars, name in [
        ('بمفو', 'labial'), ('ثذظ', 'dental'), ('تدطسصزشنرل', 'coronal'),
        ('يج', 'palatal'), ('كقخغ', 'dorsal'), ('ءهحع', 'guttural'),
        ('َُِاۦۥ', 'vowel'), ('ں۾', 'nasalized')]
        if p in chars), 'other:' + p)
    voice = ('voiced' if p in 'بجدذزظعغرلمنويَُِاۦۥں۾' else 'unvoiced')
    return [place, manner, voice]


def class_map(tokens):
    labels = [attributes(p) for p in '_' + tokens]
    names = [sorted({r[h] for r in labels}) for h in range(3)]
    return [[names[h].index(r[h]) for h in range(3)] for r in labels], names


def graph(ids, stay):
    import numpy as np
    labels = [0] + [q for p in ids for q in [p, 0]]
    transitions = np.full((len(labels), len(labels)), -np.inf)
    for s in range(len(labels)):
        following = [s+1] if s+1 < len(labels) else []
        if s % 2 == 1 and s+2 < len(labels) and labels[s] != labels[s+2]:
            following.append(s+2)
        transitions[s, s] = np.log(stay if following else 1)
        for q in following:
            transitions[s, q] = np.log((1-stay)/len(following))
    return labels, transitions


def chain(emission, ids, stay, posterior=False):
    """Forward/backward on the original-phone graph, normalized on completion."""
    import numpy as np
    from scipy.special import logsumexp
    labels, tr = graph(ids, stay)
    # Only self / next / skip edges; no dense time-by-state-by-state tensor.
    self_edge = np.diag(tr)
    step_edge = np.r_[-np.inf, np.diag(tr, 1)]
    skip_edge = np.r_[-np.inf, -np.inf, np.diag(tr, 2)]
    def forward(v):
        return np.logaddexp(np.logaddexp(v+self_edge, np.r_[-np.inf, v[:-1]]+step_edge),
                            np.r_[-np.inf, -np.inf, v[:-2]]+skip_edge)
    tmax, states = emission.shape
    a = np.full((tmax, states), -np.inf)
    prior = np.full(states, -np.inf)
    a[0, :2], prior[:2] = emission[0, :2]-np.log(2), -np.log(2)
    for t in range(1, tmax):
        a[t] = forward(a[t-1])+emission[t]
        prior = forward(prior)
    z = logsumexp(a[-1, -2:])
    value = z-logsumexp(prior[-2:])
    if not posterior:
        return value
    b = np.full_like(a, -np.inf)
    b[-1, -2:] = 0
    for t in range(tmax-2, -1, -1):
        v = emission[t+1]+b[t+1]
        b[t] = np.logaddexp(np.logaddexp(v+self_edge, np.r_[v[1:]+step_edge[1:], -np.inf]),
                            np.r_[v[2:]+skip_edge[2:], -np.inf, -np.inf])
    return value, np.exp(a+b-z)


def emissions(model, x, labels, heads=(0, 1, 2)):
    import numpy as np
    classes = np.array(model['classes'])[labels]
    out = np.zeros((len(x), len(labels)))
    for h in heads:
        mu = np.array(model['means'][h])[classes[:, h]]
        var = np.array(model['variances'][h])[classes[:, h]]
        out += -.5*(np.log(2*np.pi*var)[None] + (x[:, None]-mu[None])**2/var[None]).sum(2)
    return out


def fit(rows, tokens, shrinkage, stay, epochs=8):
    import numpy as np
    classes, names = class_map(tokens)
    raw = np.concatenate([r['x'] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .1)
    rows = [dict(r, x=(r['x']-center)/scale) for r in rows]
    def fresh():
        return [(np.zeros(len(head)), np.zeros((len(head), 20)), np.zeros((len(head), 20))) for head in names]
    def add(stats, r, gamma, labels):
        for s, p in enumerate(labels):
            weights = gamma[:, s]
            for h in range(3):
                c = classes[p][h]
                stats[h][0][c] += weights.sum()
                stats[h][1][c] += weights @ r['x']
                stats[h][2][c] += weights @ (r['x']**2)
    def update(stats):
        means, variances = [], []
        for count, sums, seconds in stats:
            total = max(count.sum(), 1)
            mean, second = sums.sum(0)/total, seconds.sum(0)/total
            mu = (sums+shrinkage*mean)/(count[:, None]+shrinkage)
            sec = (seconds+shrinkage*second)/(count[:, None]+shrinkage)
            means.append(mu.tolist())
            variances.append(np.maximum(sec-mu**2, .05).tolist())
        return means, variances
    stats = fresh()
    # Uniform phone placement only initializes EM; no forced word boundaries.
    for r in rows:
        labels, _ = graph(r['ids'], stay)
        placement = np.minimum(np.arange(len(r['x']))*len(labels)//len(r['x']), len(labels)-1)
        gamma = np.eye(len(labels))[placement]
        add(stats, r, gamma, labels)
    means, variances = update(stats)
    model = dict(center=center.tolist(), scale=scale.tolist(), classes=classes,
                 means=means, variances=variances, stay=stay)
    history = []
    for _ in range(epochs):
        stats, counts, values = fresh(), np.zeros(len(tokens)+1), []
        for r in rows:
            labels, _ = graph(r['ids'], stay)
            value, gamma = chain(emissions(model, r['x'], labels), r['ids'], stay, True)
            if not np.isfinite(value) or not np.isfinite(gamma).all():
                raise ValueError('Nonfinite complete-pronunciation EM')
            add(stats, r, gamma, labels)
            for s, p in enumerate(labels):
                counts[p] += gamma[:, s].sum()
            values.append(float(value)/len(r['x']))
        model['means'], model['variances'] = update(stats)
        history.append(float(np.mean(values)))
    model['counts'] = counts.tolist()
    return model, history


def validation(model, rows):
    import numpy as np
    by_speaker = {}
    for r in rows:
        x = (r['x']-model['center'])/model['scale']
        labels, _ = graph(r['ids'], model['stay'])
        value = chain(emissions(model, x, labels), r['ids'], model['stay'])/len(x)
        by_speaker.setdefault(r['speaker'], []).append(float(value))
    return float(np.mean([np.mean(v) for v in by_speaker.values()]))


def prepare(rows, tokens, excluded):
    import numpy as np
    out = []
    for r in rows:
        if r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r['speaker'] in excluded:
            raise ValueError('Only non-excluded none-labelled training speakers may be fitted')
        x = np.asarray(r['features'], dtype=float)
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        if x.ndim != 2 or x.shape[1] != 20 or not np.isfinite(x).all() or not ids or len(x) < len(ids):
            raise ValueError('Invalid SDK observations or incomplete pronunciation')
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
        for stay in [.7, .9]:
            values = []
            for fold in range(3):
                model, _ = fit([r for r in rows if folds[r['speaker']] != fold], args.tokens, shrinkage, stay)
                values.append(validation(model, [r for r in rows if folds[r['speaker']] == fold]))
            results.append(dict(shrinkage=shrinkage, stay=stay, foldLogP=values, meanLogP=sum(values)/3))
    chosen = max(results, key=lambda r: (r['meanLogP'], r['shrinkage'], r['stay']))
    model, history = fit(rows, args.tokens, chosen['shrinkage'], chosen['stay'])
    model.update(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40,
                 observationMs=10, dimensions=20, maxFrames=4096)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'cue-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size,
                      sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'cue-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                  speakerFolds=folds, excludedSpeakers=sorted(excluded), validation=results,
                  selected=chosen, history=history, asset=descriptor, featureClasses=class_map(args.tokens)[1])
    (out/'cue-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

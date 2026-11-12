"""Training-only matched envelope/curvature emission models, CPU on Modal.

Only none-labelled help-train training speakers are accepted. Both arms use
identical complete-pronunciation CTC state topology and fixed geometric duration
priors. Soft EM marginalizes all phone boundaries rather than importing engine
word slots or peak labels. Resolution and shrinkage are chosen together from
speaker-grouped training-only contrast validation, then fitted once and frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

RESOLUTIONS = (16, 24, 32)
SHRINKAGES = (8, 32, 128)
EPOCHS = 12
STAY = .8
MAX_FRAMES = 2048


def view(x, curved):
    return x[:, :-2]-2*x[:, 1:-1]+x[:, 2:] if curved else x[:, 1:-1]


def topology(labels):
    states = np.zeros(2*len(labels)+1, dtype=int)
    states[1::2] = labels
    skip = np.zeros(len(states), dtype=bool)
    skip[2:] = (states[2:] != 0) & (states[2:] != states[:-2])
    count = np.ones(len(states), dtype=int)
    count[:-1] += 1
    count[:-2] += skip[2:]
    selfp = np.where(count > 1, np.log(STAY), 0.)
    step = np.full(len(states), -np.inf)
    step[count > 1] = np.log((1-STAY)/(count[count > 1]-1))
    return states, skip, selfp, step


def chain(rows, labels, posterior=False):
    states, skip, selfp, step = topology(labels)
    e = rows[:, states]
    a = np.full_like(e, -np.inf)
    prior = np.full(len(states), -np.inf)
    a[0, :2] = e[0, :2]-np.log(2)
    prior[:2] = -np.log(2)
    for t in range(1, len(e)):
        a[t] = np.logaddexp(a[t-1]+selfp, np.r_[-np.inf, a[t-1, :-1]+step[:-1]])
        z = np.r_[-np.inf, -np.inf, a[t-1, :-2]+step[:-2]]
        a[t] = np.logaddexp(a[t], np.where(skip, z, -np.inf))+e[t]
        p = np.logaddexp(prior+selfp, np.r_[-np.inf, prior[:-1]+step[:-1]])
        prior = np.logaddexp(p, np.where(skip, np.r_[-np.inf, -np.inf, prior[:-2]+step[:-2]], -np.inf))
    mass = np.logaddexp(a[-1, -1], a[-1, -2])
    if not np.isfinite(mass):
        if posterior:
            raise ValueError('Incomplete pronunciation has no accepting path')
        return float('-inf')
    value = mass-np.logaddexp(prior[-1], prior[-2])
    if not posterior:
        return float(value)
    b = np.full_like(e, -np.inf)
    b[-1, -2:] = 0
    for t in range(len(e)-2, -1, -1):
        future = e[t+1]+b[t+1]
        b[t] = np.logaddexp(selfp+future, np.r_[step[:-1]+future[1:], -np.inf])
        z = np.r_[np.where(skip[2:], step[:-2]+future[2:], -np.inf), -np.inf, -np.inf]
        b[t] = np.logaddexp(b[t], z)
    return float(value), states, np.exp(a+b-mass)


def emissions(arm, x):
    mean = np.array([e['mean'] for e in arm['emissions']])
    variance = np.array([e['variance'] for e in arm['emissions']])
    return -.5*(np.log(2*np.pi*variance)[None]+(x[:, None]-mean[None])**2/variance[None]).sum(2)


def estimate(count, sums, squares, shrinkage):
    total = count.sum()
    mu = sums.sum(0)/total
    second = squares.sum(0)/total
    mean = (sums+shrinkage*mu)/(count[:, None]+shrinkage)
    variance = (squares+shrinkage*second)/(count[:, None]+shrinkage)-mean**2
    variance = np.maximum(variance, .05)
    return [dict(mean=m.tolist(), variance=v.tolist()) for m, v in zip(mean, variance)]


def fit(rows, tokens, resolution, shrinkage):
    arms, history = {}, {}
    counts = [sum(r['pronunciation'].count(p) for r in rows) for p in tokens]
    for curved in (False, True):
        values = [view(r['features'][str(resolution)], curved) for r in rows]
        raw = np.concatenate(values)
        center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
        xs = [(x-center)/scale for x in values]
        vocab, d = len(tokens)+1, resolution-2
        def fresh():
            return np.zeros(vocab), np.zeros((vocab, d)), np.zeros((vocab, d))
        def add(stats, x, p, weights):
            stats[0][p] += weights.sum()
            stats[1][p] += weights@x
            stats[2][p] += weights@(x*x)
        stats = fresh()
        # Uniform phonetic placement with silence at utterance edges is only an
        # initialization. Every subsequent update sums all possible boundaries.
        for r, x in zip(rows, xs):
            ids = [0]+r['ids']+[0]
            assigned = np.minimum(np.arange(len(x))*len(ids)//len(x), len(ids)-1)
            for s, p in enumerate(ids):
                add(stats, x, p, (assigned == s).astype(float))
        arm = dict(center=center.tolist(), scale=scale.tolist(), emissions=estimate(*stats, shrinkage))
        losses = []
        for epoch in range(EPOCHS):
            stats, scores = fresh(), []
            for r, x in zip(rows, xs):
                score, states, gamma = chain(emissions(arm, x), r['ids'], True)
                scores.append(score/len(x))
                for s, p in enumerate(states):
                    add(stats, x, p, gamma[:, s])
            arm['emissions'] = estimate(*stats, shrinkage)
            losses.append(float(np.mean(scores)))
        key = 'curvature' if curved else 'control'
        arms[key], history[key] = arm, losses
    return dict(schema=1, tokens=tokens, resolution=resolution, frameMs=40, stay=STAY,
                maxFrames=MAX_FRAMES, phoneCounts=counts, **arms), history


def score(model, r, labels, curved):
    arm = model['curvature' if curved else 'control']
    x = (view(r['features'][str(model['resolution'])], curved)-arm['center'])/arm['scale']
    return chain(emissions(arm, x), labels)


def validation(model, rows):
    """Balanced one-phone contrasts; candidates derive only from training text.
    Contrast log loss is comparable across frequency resolutions, unlike raw
    density in spaces of different dimension. Both arms share the same rivals.
    """
    by_speaker = {}
    supported = [p+1 for p, n in enumerate(model['phoneCounts']) if n]
    for r in rows:
        rng = np.random.default_rng(int(hashlib.sha256((r['speaker']+r['pronunciation']).encode()).hexdigest()[:8], 16))
        alternatives = [r['ids']]
        for _ in range(4):
            labels = r['ids'].copy()
            at = int(rng.integers(len(labels)))
            options = [p for p in supported if p != labels[at]]
            labels[at] = int(rng.choice(options))
            alternatives.append(labels)
        losses = []
        for curved in (False, True):
            scores = np.array([score(model, r, labels, curved) for labels in alternatives])
            losses.append(float(np.logaddexp.reduce(scores)-scores[0]))
        by_speaker.setdefault(r['speaker'], []).append(float(np.mean(losses)))
    return float(np.mean([np.mean(v) for v in by_speaker.values()]))


def prepare(records, tokens, excluded):
    rows = []
    for r in records:
        if r.get('source') != 'help-train' or r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r['speaker'] in excluded:
            raise ValueError('Only none-labelled help-train training speakers accepted')
        p = r['pronunciation']
        if not p or any(c not in tokens for c in p):
            raise ValueError('Invalid complete canonical pronunciation')
        features = {str(k): np.asarray(v, dtype=float) for k, v in r['features'].items()}
        minimum_frames = len(p)+sum(a == b for a, b in zip(p, p[1:]))
        for k in RESOLUTIONS:
            x = features[str(k)]
            if x.ndim != 2 or x.shape[1] != k or not np.isfinite(x).all() or len(x) < minimum_frames or len(x) > MAX_FRAMES:
                raise ValueError('Invalid envelope observations or insufficient duration')
        rows.append(dict(features=features, ids=[tokens.index(c)+1 for c in p], pronunciation=p, speaker=r['speaker']))
    if len({r['speaker'] for r in rows}) < 3:
        raise ValueError('At least three eligible training speakers required')
    return rows


def main():
    parser = argparse.ArgumentParser()
    for key in ('input', 'out', 'excluded-speakers'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    meta = json.loads(Path(args.input+'.inventory.json').read_text())
    tokens, inventory = meta['tokens'], meta['inventory']
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = prepare([json.loads(line) for line in Path(args.input).open()], tokens, excluded)
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    results = []
    for resolution in RESOLUTIONS:
        for shrinkage in SHRINKAGES:
            losses = []
            for fold in range(3):
                train = [r for r in rows if folds[r['speaker']] != fold]
                valid = [r for r in rows if folds[r['speaker']] == fold]
                model, _ = fit(train, tokens, resolution, shrinkage)
                losses.append(validation(model, valid))
            result = dict(resolution=resolution, shrinkage=shrinkage, foldLoss=losses, meanLoss=float(np.mean(losses)))
            results.append(result)
            print(json.dumps(result), flush=True)
    chosen = min(results, key=lambda r: (r['meanLoss'], r['resolution'], -r['shrinkage']))
    model, history = fit(rows, tokens, chosen['resolution'], chosen['shrinkage'])
    model['inventory'] = inventory
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'curvature-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'curvature-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), speakerFolds=folds, excludedSpeakers=sorted(excluded),
                  inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(), validation=results,
                  selected=chosen, history=history, asset=descriptor)
    (out/'curvature-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(selected=chosen, recordings=len(rows), speakers=len(speakers), asset=descriptor)), flush=True)


if __name__ == '__main__':
    main()

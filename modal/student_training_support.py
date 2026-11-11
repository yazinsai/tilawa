"""Compact recording-jackknife pronunciation emissions; CPU work runs on Modal.

Only help-train none-labelled training recordings fit emissions. Scripted swaps
are weak preservation controls, never imposed as performed acoustic transcripts.
Every deletion refits preprocessing, priors and boundary EM from scratch.
"""
import argparse
import hashlib
import json
from pathlib import Path

DIMENSIONS = 8
STRENGTHS = (2, 8, 32)
EPOCHS = 12
STAY = .8
MAX_FRAMES = 2048


def chain(e, posterior=False):
    import numpy as np
    tmax, states = e.shape
    if not states or tmax < states:
        raise ValueError('Incomplete pronunciation')
    a = np.full_like(e, -np.inf)
    prior = np.full(states, -np.inf)
    a[0, 0], prior[0] = e[0, 0], 0
    selfp, step = np.log(STAY), np.log1p(-STAY)
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


def emission(fit, x, ids):
    import numpy as np
    phones = fit['phones']
    means = np.array([phones[p]['mean'] for p in ids])
    variances = np.array([phones[p]['variance'] for p in ids])
    return -.5*(np.log(2*np.pi*variances).sum(1)[None, :]+((x[:, None, :]-means[None, :, :])**2/variances).sum(2))


def manner(p):
    for chars, group in [('َُِاۦۥ', 'vowel'), ('منں۾', 'nasal'), ('بتدطكقءڇ', 'stop'),
                         ('ثذفسشصزظخغحعه', 'fricative'), ('رلوي', 'approximant')]:
        if p in chars:
            return group
    return 'other'


def update(stats, tokens, strength):
    import numpy as np
    counts, sums, seconds = stats
    total = max(counts.sum(), 1)
    mu = sums.sum(0)/total
    second = seconds.sum(0)/total
    groups = [manner(p) for p in '_' + tokens]
    phones = []
    for p in range(len(groups)):
        pool = [q for q in range(1, len(groups)) if q != p and groups[p] == groups[q]]
        n = counts[pool].sum()
        mu0 = (sums[pool].sum(0)+strength*mu)/(n+strength)
        second0 = (seconds[pool].sum(0)+strength*second)/(n+strength)
        mean = (sums[p]+strength*mu0)/(counts[p]+strength)
        variance = (seconds[p]+strength*second0)/(counts[p]+strength)-mean**2
        phones.append(dict(mean=mean.tolist(), variance=np.maximum(variance, .05).tolist(), count=float(counts[p])))
    return phones


def fit(rows, tokens, strength, epochs=EPOCHS):
    import numpy as np
    raw = np.concatenate([r['x'] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .1)
    rows = [dict(r, x=(r['x']-center)/scale) for r in rows]
    vocab = len(tokens)+1
    def fresh():
        return np.zeros(vocab), np.zeros((vocab, DIMENSIONS)), np.zeros((vocab, DIMENSIONS))
    def accumulate(stats, r, weights):
        for state, p in enumerate(r['ids']):
            w = weights[:, state]
            stats[0][p] += w.sum()
            stats[1][p] += w@r['x']
            stats[2][p] += w@(r['x']**2)
    stats = fresh()
    for r in rows:
        assigned = np.minimum(np.arange(len(r['x']))*len(r['ids'])//len(r['x']), len(r['ids'])-1)
        accumulate(stats, r, (assigned[:, None] == np.arange(len(r['ids']))[None, :]).astype(float))
    model = dict(center=center.tolist(), scale=scale.tolist(), phones=update(stats, tokens, strength))
    for _ in range(epochs):
        stats = fresh()
        for r in rows:
            _, gamma = chain(emission(model, r['x'], r['ids']), True)
            accumulate(stats, r, gamma)
        model['phones'] = update(stats, tokens, strength)
    return model


def score(model, raw, ids):
    import numpy as np
    if len(raw) < len(ids) or any(model['phones'][p]['count'] <= 0 for p in ids):
        return None
    x = (raw-model['center'])/model['scale']
    value = chain(emission(model, x, ids))-len(x)*np.log(model['scale']).sum()
    return value if np.isfinite(value) else None


def comparison(full, deletions, raw, written, rival):
    values = [score(full, raw, s) for s in (written, rival)]
    if any(v is None for v in values):
        return None
    preference = values[0]-values[1]
    best = None
    for recording, deleted in enumerate(deletions):
        scores = [score(deleted, raw, s) for s in (written, rival)]
        if any(v is None for v in scores):
            continue
        influence = (scores[0]-scores[1]-preference)/len(raw)
        if best is None or abs(influence) > abs(best['influence']):
            best = dict(fullPreference=preference, removedPreference=scores[0]-scores[1], influence=influence, recording=recording)
    return best


def prepare(raw, tokens, excluded):
    import numpy as np
    rows = []
    seen = set()
    for r in raw:
        if r.get('source') != 'help-train' or r.get('split') != 'train' or r.get('excluded') is not False or not r.get('speaker') or r['speaker'] in excluded:
            raise ValueError('Training source or speaker exclusion violation')
        if r['id'] in seen:
            raise ValueError('Duplicate recording: its frames must form one deletion group')
        seen.add(r['id'])
        x = np.array(r['features'], dtype=float)
        if x.ndim != 2 or x.shape[1] != DIMENSIONS or not np.isfinite(x).all():
            raise ValueError('Invalid acoustic observations')
        if r['kind'] == 'none':
            ids = [tokens.index(p)+1 for p in r['pronunciation']]
            if not ids or len(x) < len(ids):
                raise ValueError('Incomplete training pronunciation')
            rows.append(dict(id=r['id'], speaker=r['speaker'], x=x, ids=ids, kind='none'))
        elif r['kind'] == 'substitution-control':
            for c in r['comparisons']:
                a, b = c['fromFrame']*4, c['toFrame']*4
                if a < 0 or b > len(x) or b <= a:
                    raise ValueError('Control window is unavailable')
                rows.append(dict(id=r['id'], speaker=r['speaker'], x=x[a:b], ids=[tokens.index(p)+1 for p in c['written']],
                    rival=[tokens.index(p)+1 for p in c['rival']], kind='control'))
        else:
            raise ValueError('Unsupported supervision')
    return rows


def jackknife(rows, tokens, strength):
    full = fit(rows, tokens, strength)
    deleted = []
    for i in range(len(rows)):
        deleted.append(fit(rows[:i]+rows[i+1:], tokens, strength))
    return full, deleted


def main():
    p = argparse.ArgumentParser()
    for arg in ('input', 'out', 'tokens', 'inventory', 'excluded-speakers'):
        p.add_argument('--'+arg, required=True)
    args = p.parse_args()
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = prepare([json.loads(s) for s in Path(args.input).read_text().splitlines()], args.tokens, excluded)
    anchors = [r for r in rows if r['kind'] == 'none']
    controls = [r for r in rows if r['kind'] == 'control']
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    if len({r['speaker'] for r in anchors}) < 3 or len({r['speaker'] for r in controls}) < 2:
        raise ValueError('At least three anchor speakers and two control speakers required')
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    selection = []
    for strength in STRENGTHS:
        values = []
        for fold in range(3):
            train = [r for r in anchors if folds[r['speaker']] != fold]
            test = [r for r in anchors if folds[r['speaker']] == fold]
            if not train or not test:
                raise ValueError('Empty speaker fold')
            model = fit(train, args.tokens, strength)
            by_speaker = {}
            for r in test:
                value = score(model, r['x'], r['ids'])
                if value is None:
                    raise ValueError('Held-out canonical phone lacks training coverage')
                by_speaker.setdefault(r['speaker'], []).append(value/len(r['x']))
            values.append(sum(sum(v)/len(v) for v in by_speaker.values())/len(by_speaker))
        selection.append(dict(strength=strength, folds=values, mean=sum(values)/3))
    chosen = max(selection, key=lambda r: (r['mean'], r['strength']))['strength']
    calibration, cutoff = [], 0.0
    for fold in range(3):
        train = [r for r in anchors if folds[r['speaker']] != fold]
        full, deletions = jackknife(train, args.tokens, chosen)
        for r in controls:
            if folds[r['speaker']] != fold:
                continue
            result = comparison(full, deletions, r['x'], r['ids'], r['rival'])
            calibration.append(dict(speaker=r['speaker'], id=r['id'], fold=fold, comparison=result))
            # Strict > cutoff veto preserves every available reversal in these weak controls.
            if result and result['fullPreference'] < 0 < result['removedPreference']:
                cutoff = max(cutoff, result['influence'])
    if any(r['comparison'] is None for r in calibration) or not calibration:
        raise ValueError('Incomplete substitution-preservation calibration')
    full, deletions = jackknife(anchors, args.tokens, chosen)
    model = dict(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40, observationMs=10, dimensions=DIMENSIONS,
        stay=STAY, maxFrames=MAX_FRAMES, cutoff=cutoff, full=full, deletions=deletions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'training-support-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'training-support-asset.json').write_text(json.dumps(descriptor))
    report = dict(inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(), excludedSpeakers=sorted(excluded),
        speakerFolds=folds, selection=selection, strength=chosen, cutoff=cutoff, calibration=calibration,
        recordings=[dict(id=r['id'], speaker=r['speaker']) for r in anchors], asset=descriptor)
    (out/'training-support-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(recordings=len(anchors), controls=len(controls), cutoff=cutoff, strength=chosen)), flush=True)


if __name__ == '__main__':
    main()

"""CPU three-phase acoustic EM; inputs must be SDK 10 ms phaseFeatures.

Only none-labelled help-train training speakers supply transcripts. Phone and
phase boundaries are latent throughout EM and validation. Mistake scripts are
not performed transcripts. No practice input or target selection is accepted.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def phone_features(p):
    """Shared Arabic phonetic attributes for sparse emission regularization."""
    manner = next((name for chars, name in [("َُِاۦۥ", "vowel"), ("منں۾", "nasal"),
        ("بتدطكقءڇ", "stop"), ("ثذفسشصزظخغحعه", "fricative"), ("رلوي", "approximant")]
        if p in chars), "other")
    place = next((name for chars, name in [("بمفو", "labial"), ("ثذظ", "dental"),
        ("تدطسصزشنرل", "coronal"), ("يكقخغ", "dorsal"), ("ءهحع", "guttural")]
        if p in chars), "other")
    voicing = "voiced" if p in "بجدذزظعغرلمنوياَُِۦۥں۾" else "unvoiced"
    return [manner, place, voicing]


def chain(emissions, stay, posterior=False):
    """Exact forward / backward over every ordered state boundary."""
    import numpy as np
    tmax, states = emissions.shape
    a = np.full((tmax, states), -np.inf)
    prior = np.full(states, -np.inf)
    a[0, 0], prior[0] = emissions[0, 0], 0
    logstay = np.log(np.array(stay)[np.arange(states) % 3])
    logstep = np.log1p(-np.array(stay)[(np.arange(states) + 2) % 3])
    for t in range(1, tmax):
        prev = np.r_[-np.inf, a[t-1, :-1]] + logstep
        a[t] = np.logaddexp(a[t-1] + logstay, prev) + emissions[t]
        prior = np.logaddexp(prior + logstay, np.r_[-np.inf, prior[:-1]] + logstep)
    likelihood = a[-1, -1]
    conditional = likelihood - prior[-1]
    if not posterior:
        return conditional
    b = np.full_like(a, -np.inf)
    b[-1, -1] = 0
    for t in range(tmax-2, -1, -1):
        following = np.r_[b[t+1, 1:] + emissions[t+1, 1:] + logstep[1:], -np.inf]
        b[t] = np.logaddexp(logstay + emissions[t+1] + b[t+1], following)
    gamma = np.exp(a + b - likelihood)
    self_counts = np.exp(a[:-1] + logstay + emissions[1:] + b[1:] - likelihood).sum(0)
    step_counts = np.exp(a[:-1, :-1] + logstep[1:] + emissions[1:, 1:] + b[1:, 1:] - likelihood).sum(0)
    return conditional, gamma, self_counts, np.r_[step_counts, 1.0]


def emissions(model, x, ids):
    import numpy as np
    mu, var = np.array(model['means'])[ids].reshape(-1, 20), np.array(model['variances'])[ids].reshape(-1, 20)
    return -.5 * (np.log(2 * np.pi * var)[None] + (x[:, None] - mu[None]) ** 2 / var[None]).sum(2)


def update(stats, tokens, shrinkage):
    import numpy as np
    counts, sums, seconds = stats
    global_count = np.maximum(counts.sum(0), 1)
    global_mean = sums.sum(0) / global_count[:, None]
    global_second = seconds.sum(0) / global_count[:, None]
    features = [phone_features(p) for p in "_" + tokens]
    means, variances = [], []
    for p in range(len(features)):
        shared_mu, shared_second = [], []
        for f in range(3):
            pool = [q for q in range(1, len(features)) if features[q][f] == features[p][f]]
            count = counts[pool].sum(0)
            denom = count[:, None] + shrinkage
            shared_mu.append((sums[pool].sum(0) + shrinkage * global_mean) / denom)
            shared_second.append((seconds[pool].sum(0) + shrinkage * global_second) / denom)
        target_mu, target_second = np.mean(shared_mu, 0), np.mean(shared_second, 0)
        denominator = counts[p, :, None] + shrinkage
        mu = (sums[p] + shrinkage * target_mu) / denominator
        second = (seconds[p] + shrinkage * target_second) / denominator
        means.append(mu.tolist())
        variances.append(np.maximum(second - mu ** 2, .05).tolist())
    return means, variances


def fit(rows, tokens, shrinkage, epochs=12):
    import numpy as np
    raw = np.concatenate([r['x'] for r in rows])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .1)
    normalized = [dict(r, x=(r['x']-center)/scale) for r in rows]
    vocab = len(tokens)+1
    def fresh():
        return np.zeros((vocab, 3)), np.zeros((vocab, 3, 20)), np.zeros((vocab, 3, 20))
    stats = fresh()
    # Uniform initialization only. Subsequent updates integrate all phase / phone boundaries.
    for r in normalized:
        state = np.minimum(np.arange(len(r['x']))*len(r['ids'])*3//len(r['x']), len(r['ids'])*3-1)
        for s in range(len(r['ids'])*3):
            part = r['x'][state == s]
            p, q = r['ids'][s//3], s%3
            stats[0][p, q] += len(part)
            stats[1][p, q] += part.sum(0)
            stats[2][p, q] += (part**2).sum(0)
    means, variances = update(stats, tokens, shrinkage)
    model = dict(center=center.tolist(), scale=scale.tolist(), means=means, variances=variances, stay=[.7, .9, .7])
    history = []
    for _ in range(epochs):
        stats, self_counts, step_counts, values = fresh(), np.zeros(3), np.zeros(3), []
        for r in normalized:
            value, gamma, selfc, stepc = chain(emissions(model, r['x'], r['ids']), model['stay'], True)
            values.append(float(value)/len(r['x']))
            for s in range(len(r['ids'])*3):
                p, q = r['ids'][s//3], s%3
                weights = gamma[:, s]
                stats[0][p, q] += weights.sum()
                stats[1][p, q] += weights @ r['x']
                stats[2][p, q] += weights @ (r['x']**2)
                self_counts[q] += selfc[s]
                step_counts[q] += stepc[s]
        model['means'], model['variances'] = update(stats, tokens, shrinkage)
        model['stay'] = ((self_counts+1)/(self_counts+step_counts+2)).tolist()
        history.append(float(np.mean(values)))
    model['counts'] = stats[0].sum(1).tolist()
    return model, history


def validation(model, rows):
    import numpy as np
    by_speaker = {}
    for r in rows:
        x = (r['x'] - model['center']) / model['scale']
        value = chain(emissions(model, x, r['ids']), model['stay']) / len(x)
        by_speaker.setdefault(r['speaker'], []).append(float(value))
    return float(np.mean([np.mean(v) for v in by_speaker.values()]))


def prepare(rows, tokens, excluded):
    import numpy as np
    out = []
    for r in rows:
        if r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r['speaker'] in excluded:
            raise ValueError('Only none-labelled, non-excluded training-speaker recordings may be fitted')
        x = np.array(r['features'], dtype=float)
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        if x.ndim != 2 or x.shape[1] != 20 or not np.isfinite(x).all() or not ids or len(x) < len(ids)*3:
            raise ValueError('Invalid SDK acoustic observations or impossible complete pronunciation')
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
        values = []
        for fold in range(3):
            model, _ = fit([r for r in rows if folds[r['speaker']] != fold], args.tokens, shrinkage)
            values.append(validation(model, [r for r in rows if folds[r['speaker']] == fold]))
        results.append(dict(shrinkage=shrinkage, foldLogP=values, meanLogP=sum(values)/3))
    chosen = max(results, key=lambda r: (r['meanLogP'], r['shrinkage']))
    model, history = fit(rows, args.tokens, chosen['shrinkage'])
    model.update(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40, observationMs=10, dimensions=20, maxFrames=2048)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'phase-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'phase-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        speakerFolds=folds, excludedSpeakers=sorted(excluded), validation=results, selected=chosen, history=history, asset=descriptor)
    (out/'phase-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

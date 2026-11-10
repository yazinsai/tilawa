"""CPU latent-interval spectral quotient fit. Never accepts practice recordings.

Extracted input carries SDK FFT-bin observations, canonical whole-ayah phones,
none labels and explicit training-speaker provenance. Both arms marginalize all
CV intervals at inference; fitting uses latent MAP intervals (hard EM), starting
from uniform transcript placement. No existing listener/edit backtrace is used.
"""
import argparse
import hashlib
import json
from pathlib import Path

VOWELS = "َُِاۦۥ"
CONSONANTS = "ءبتثجحخدذرزسشصضطظعغفقكلمنهويڇں۾"
DIMENSIONS = 32
MAX_PART = 32
MAX_FRAMES = 1024


def units(pronunciation):
    out, i = [], 0
    while i < len(pronunciation):
        if pronunciation[i] in CONSONANTS and i+1 < len(pronunciation) and pronunciation[i+1] in VOWELS:
            out.append(pronunciation[i:i+2])
            i += 2
        else:
            out.append(pronunciation[i])
            i += 1
    return out


def gaussian(x, emission):
    import numpy as np
    mean, variance = np.array(emission['mean']), np.array(emission['variance'])
    return -.5*(np.log(2*np.pi*variance)+(x-mean)**2/variance).sum(-1)


def lattice(model, raw, pronunciation, relative, viterbi=False):
    """Same segment/cut prior as SDK, with optional MAP boundaries for fitting."""
    import numpy as np
    us, tmax = units(pronunciation), len(raw)
    if not us or tmax > model['maxFrames']:
        return -np.inf, []
    prefix = np.vstack([np.zeros(DIMENSIONS), np.cumsum(raw/np.array(model['scale']), axis=0)])
    score, prior = np.full(tmax+1, -np.inf), np.full(tmax+1, -np.inf)
    score[0] = prior[0] = 0
    back, cache = [], {}
    for u in us:
        pair, minimum = len(u) == 2, len(u)
        limit = model['maxPart'] * len(u)
        nxt, npprior = np.full(tmax+1, -np.inf), np.full(tmax+1, -np.inf)
        pointers = np.full((tmax+1, 2), -1, dtype=int)
        for length in range(minimum, min(limit, tmax)+1):
            start = np.arange(tmax-length+1)
            end = start+length
            key = (u, length)
            if key not in cache:
                if not pair:
                    x = (prefix[end]-prefix[start])/length
                    values, cuts = gaussian(x, model['absolute'][u]), np.zeros(len(start), dtype=int)
                else:
                    splits = np.arange(max(1, length-model['maxPart']), min(length-1, model['maxPart'])+1)
                    mid = start[:, None]+splits[None]
                    left = (prefix[mid]-prefix[start, None])/splits[None, :, None]
                    right = (prefix[end, None]-prefix[mid])/(length-splits)[None, :, None]
                    x = left-right if relative else left
                    g = gaussian(x, model['quotient' if relative else 'absolute'][u])
                    cuts = splits[g.argmax(1)]
                    values = g.max(1) if viterbi else np.logaddexp.reduce(g, axis=1)
                    values -= np.log(len(splits))
                cache[key] = values, cuts
            values, cuts = cache[key]
            duration = -np.log(limit-minimum+1)
            candidates = score[start]+values+duration
            if viterbi:
                better = candidates > nxt[end]
                pointers[end[better], 0] = start[better]
                pointers[end[better], 1] = start[better]+cuts[better]
                nxt[end] = np.maximum(nxt[end], candidates)
            else:
                nxt[end] = np.logaddexp(nxt[end], candidates)
            npprior[end] = np.logaddexp(npprior[end], prior[start]+duration)
        score, prior = nxt, npprior
        back.append(pointers)
    path, end = [], tmax
    if viterbi and np.isfinite(score[end]):
        for u, pointers in reversed(list(zip(us, back))):
            start, cut = pointers[end]
            path.append((u, int(start), int(cut), int(end)))
            end = start
        path.reverse()
    return float(score[tmax]-prior[tmax]), path


def observation(x, unit, start, cut, end, relative):
    import numpy as np
    if len(unit) == 1:
        return x[start:end].mean(0)
    left, right = x[start:cut].mean(0), x[cut:end].mean(0)
    return left-right if relative else left


def attributes(u):
    # Hierarchical phonetic sharing, independent of reader / word / passage.
    c = u[0]
    manner = next((name for chars, name in [("منں۾", "nasal"), ("بتدطكقءڇ", "stop"),
        ("ثذفسشصزظخغحعه", "fricative"), ("رلوي", "approximant"), (VOWELS, "vowel")]
        if c in chars), "other")
    vowel = u[1:] or "single"
    return manner, vowel


def estimate(samples, vocabulary, shrinkage):
    import numpy as np
    out = {}
    for u in vocabulary:
        dimension = DIMENSIONS
        data = samples['values'].get(u, [])
        pool = [v for key, values in samples['values'].items() if len(key) == len(u) and
                (attributes(key) == attributes(u) or key[0] == u[0]) for v in values]
        if not pool:
            pool = [v for key, values in samples['values'].items() if len(key) == len(u) for v in values]
        # No fabricated observations for an entirely absent unit family.
        if not pool:
            continue
        prior = np.array(pool)
        target_mean, target_second = prior.mean(0), (prior**2).mean(0)
        a = np.array(data).reshape(-1, dimension)
        mean = (a.sum(0)+shrinkage*target_mean)/(len(a)+shrinkage)
        second = ((a**2).sum(0)+shrinkage*target_second)/(len(a)+shrinkage)
        out[u] = dict(mean=mean.tolist(), variance=np.maximum(second-mean**2, .05).tolist())
    return out


def fit(rows, tokens, inventory, shrinkage, epochs=5):
    import numpy as np
    scale = np.maximum(np.concatenate([r['x'] for r in rows]).std(0), .1)
    vocabulary = list(tokens)+[c+v for c in tokens if c in CONSONANTS for v in tokens if v in VOWELS]
    counts = [sum(r['pronunciation'].count(p) for r in rows) for p in tokens]
    model = dict(schema=1, inventory=inventory, tokens=tokens, frameMs=40, dimensions=DIMENSIONS,
        scale=scale.tolist(), phoneCounts=counts, maxPart=MAX_PART, maxFrames=MAX_FRAMES)
    for relative in [False, True]:
        sample = dict(relative=relative, values={})
        for r in rows:
            x = r['x']/scale
            us = units(r['pronunciation'])
            # Initially each complete phone receives a uniform share of the utterance.
            lengths = np.array([len(u) for u in us])
            ends = np.r_[0, np.cumsum(lengths)*len(x)//lengths.sum()]
            for i, u in enumerate(us):
                start, end = int(ends[i]), int(ends[i+1])
                cut = (start+end)//2
                sample['values'].setdefault(u, []).append(observation(x, u, start, cut, end, relative))
        model['quotient' if relative else 'absolute'] = estimate(sample, vocabulary, shrinkage)
    history = []
    for epoch in range(epochs):
        values = []
        for relative in [False, True]:
            sample = dict(relative=relative, values={})
            for r in rows:
                value, path = lattice(model, r['x'], r['pronunciation'], relative, True)
                if not path:
                    raise ValueError('Transcript has no complete bounded latent segmentation')
                values.append(value/len(path))
                for u, start, cut, end in path:
                    sample['values'].setdefault(u, []).append(observation(r['x']/scale, u, start, cut, end, relative))
            model['quotient' if relative else 'absolute'] = estimate(sample, vocabulary, shrinkage)
        history.append(dict(epoch=epoch, meanMapLogP=float(np.mean(values))))
    return model, history


def prepare(records, tokens, excluded):
    import numpy as np
    rows = []
    for r in records:
        if r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r['speaker'] in excluded:
            raise ValueError('Fit requires none-labelled, non-excluded training speakers')
        x, p = np.array(r['features'], dtype=float), r['pronunciation']
        if x.ndim != 2 or x.shape[1] != DIMENSIONS or not np.isfinite(x).all() or not p or any(c not in tokens for c in p):
            raise ValueError('Invalid spectral observations or complete pronunciation')
        if len(x) < len(p) or len(x) > MAX_FRAMES or not any(len(u) == 2 for u in units(p)):
            raise ValueError('Insufficient observations, missing CV unit or oversized utterance')
        rows.append(dict(x=x, pronunciation=p, speaker=r['speaker']))
    if len({r['speaker'] for r in rows}) < 3:
        raise ValueError('At least three eligible training speakers required')
    return rows


def main():
    import numpy as np
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
            model, _ = fit([r for r in rows if folds[r['speaker']] != fold], args.tokens, args.inventory, shrinkage)
            by_speaker = {}
            for r in rows:
                if folds[r['speaker']] != fold:
                    continue
                scores = [lattice(model, r['x'], r['pronunciation'], relative)[0]/len(units(r['pronunciation'])) for relative in [False, True]]
                by_speaker.setdefault(r['speaker'], []).append(sum(scores)/2)
            values.append(float(np.mean([np.mean(v) for v in by_speaker.values()])))
        results.append(dict(shrinkage=shrinkage, foldLogP=values, meanLogP=sum(values)/len(values)))
    chosen = max(results, key=lambda r: (r['meanLogP'], r['shrinkage']))
    model, history = fit(rows, args.tokens, args.inventory, chosen['shrinkage'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'quotient-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'quotient-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        speakerFolds=folds, excludedSpeakers=sorted(excluded), validation=results, selected=chosen, history=history, asset=descriptor)
    (out/'quotient-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

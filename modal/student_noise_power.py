"""CPU phone-conditioned clean log-power model with an explicit complex-noise observation law.

Input JSONL contains authorized original Quran training recordings (and optionally
none-labelled help-train training takes), their complete canonical pronunciation,
16 kHz WAV path, and shipped phonemes used only to identify completed pauses.
No practice/release data or scripted mistake transcript is accepted. Fit and select
regularization in disjoint training-speaker folds; export a verified SDK side asset.
"""
import argparse
import hashlib
import json
from pathlib import Path

BINS = [4, 8, 12, 18, 26, 36, 48, 62, 78, 96, 116, 138, 162, 188, 216, 240]
STRENGTHS = (8, 32, 128)
EPOCHS = 8


def quadrature(count):
    import numpy as np
    x, w = np.polynomial.hermite.hermgauss(count)
    return x * np.sqrt(2), w / np.sqrt(np.pi)


def power_features(audio):
    import numpy as np
    windows = np.lib.stride_tricks.sliding_window_view(audio, 400)[::160].astype(float)
    windows -= windows.mean(1, keepdims=True)
    previous = np.concatenate((windows[:, :1], windows[:, :-1]), 1)
    window = (.5-.5*np.cos(2*np.pi*np.arange(400)/399))**.85
    spectrum = np.fft.rfft((windows-.97*previous)*window, n=512)
    return abs(spectrum[:, BINS])**2


def noise_before(powers, phones):
    """Same full-window guards and 8-frame (40 ms clock) pause criterion as SDK."""
    import numpy as np
    spans = []
    for left, right in zip(phones, phones[1:]):
        if right['start']-left['end']-1 >= 8:
            a, b = (left['end']+1)*4+3, right['start']*4-3
            if b > a and b <= len(powers):
                spans.append(powers[a:b])
    if not spans:
        raise ValueError('Original recording has no completed qualifying noise pause')
    n = np.concatenate(spans).mean(0)
    if not np.isfinite(n).all() or (n <= 0).any():
        raise ValueError('Invalid measured noise power')
    return n


def noisy_logp(y, s, n):
    import numpy as np
    from scipy.special import i0e
    return -np.log(n)-(np.sqrt(y)-np.sqrt(s))**2/n+np.log(i0e(2*np.sqrt(y)*np.sqrt(s)/n))


def chain(emissions, posterior=False, background=None):
    import numpy as np
    tmax, phones = emissions.shape
    if background is None:
        background = np.full(tmax, -np.inf)
    e = np.empty((tmax, phones*2+1))
    e[:, ::2] = background[:, None]
    e[:, 1::2] = emissions
    states = e.shape[1]
    a = np.full_like(e, -np.inf)
    a[0, :2] = e[0, :2]
    prior = np.full(states, -np.inf)
    prior[:2] = 0
    skip = np.arange(states) % 2 == 1
    skip[1] = False
    for t in range(1, tmax):
        a[t] = np.logaddexp(np.logaddexp(a[t-1], np.r_[-np.inf, a[t-1, :-1]]),
            np.where(skip, np.r_[-np.inf, -np.inf, a[t-1, :-2]], -np.inf))+e[t]
        prior = np.logaddexp(np.logaddexp(prior, np.r_[-np.inf, prior[:-1]]),
            np.where(skip, np.r_[-np.inf, -np.inf, prior[:-2]], -np.inf))
    total = np.logaddexp(a[-1, -1], a[-1, -2])
    value = float(total-np.logaddexp(prior[-1], prior[-2]))
    if not posterior:
        return value
    b = np.full_like(a, -np.inf)
    b[-1, -2:] = 0
    for t in range(tmax-2, -1, -1):
        nxt = e[t+1]+b[t+1]
        b[t] = np.logaddexp(np.logaddexp(nxt, np.r_[nxt[1:], -np.inf]),
            np.where(np.r_[skip[2:], False, False], np.r_[nxt[2:], -np.inf, -np.inf], -np.inf))
    return value, np.exp(a[:, 1::2]+b[:, 1::2]-total)


def latent(model, row, phone):
    import numpy as np
    from scipy.special import logsumexp
    nodes, weights = quadrature(17)
    logclean = model['means'][phone][None, :, None]+model['deviations'][phone][None, :, None]*nodes
    values = noisy_logp(row['power'][:, :, None], np.exp(logclean), row['noise'][None, :, None])+np.log(weights)
    logp = logsumexp(values, axis=2)
    posterior = np.exp(values-logp[:, :, None])
    return logp.sum(1), (posterior*logclean).sum(2), (posterior*logclean**2).sum(2)


def fit(rows, tokens, strength, epochs=EPOCHS):
    import numpy as np
    vocab, dimensions = len(tokens)+1, len(BINS)
    lograw = np.log(np.concatenate([r['power'] for r in rows]))
    mu0, var0 = lograw.mean(0), np.maximum(lograw.var(0), .01)
    def fresh():
        return np.zeros(vocab), np.zeros((vocab, dimensions)), np.zeros((vocab, dimensions))
    def update(stats):
        count, sums, seconds = stats
        means = (sums+strength*mu0)/(count[:, None]+strength)
        variance = (seconds+strength*(var0+mu0**2))/(count[:, None]+strength)-means**2
        return dict(means=means, deviations=np.sqrt(np.maximum(variance, .01)), counts=count)
    stats = fresh()
    for row in rows:
        x = np.log(row['power'])
        assigned = np.minimum(np.arange(len(x))*len(row['ids'])//len(x), len(row['ids'])-1)
        for s, p in enumerate(row['ids']):
            mask = assigned == s
            stats[0][p] += mask.sum()
            stats[1][p] += x[mask].sum(0)
            stats[2][p] += (x[mask]**2).sum(0)
    model = update(stats)
    history = []
    for epoch in range(epochs):
        stats, values = fresh(), []
        for row in rows:
            table = {p: latent(model, row, p) for p in set(row['ids'])}
            e = np.array([table[p][0] for p in row['ids']]).T
            background = (-np.log(row["noise"])-row["power"]/row["noise"]).sum(1)
            value, gamma = chain(e, True, background)
            values.append(value/len(e))
            for s, p in enumerate(row['ids']):
                w = gamma[:, s]
                stats[0][p] += w.sum()
                stats[1][p] += w@table[p][1]
                stats[2][p] += w@table[p][2]
        model = update(stats)
        history.append(float(np.mean(values)))
    return model, history


def validation(model, rows):
    import numpy as np
    scores = {}
    for row in rows:
        table = {p: latent(model, row, p)[0] for p in set(row['ids'])}
        background = (-np.log(row['noise'])-row['power']/row['noise']).sum(1)
        score = chain(np.array([table[p] for p in row['ids']]).T, background=background)/len(row['power'])
        scores.setdefault(row['speaker'], []).append(score)
    return float(np.mean([np.mean(v) for v in scores.values()]))


def prepare(records, tokens, excluded):
    import numpy as np
    import soundfile as sf
    out = []
    for r in records:
        original = r.get('source') == 'quran-training' and r.get('transcriptVerified') is True
        learner = r.get('source') == 'help-train' and r.get('kind') == 'none'
        if not (original or learner) or r.get('split') != 'train' or r.get('authorized') is not True or r['speaker'] in excluded:
            raise ValueError('Only authorized non-excluded training audio with known canonical content is accepted')
        audio, sr = sf.read(r['wav'], dtype='float32')
        if sr != 16000 or audio.ndim != 1 or not np.isfinite(audio).all() or len(audio) < 400:
            raise ValueError('Training inputs must be finite mono 16 kHz original audio')
        powers = power_features(audio)
        noise = noise_before(powers, r['phonemes'])
        ids = [tokens.index(p)+1 for p in r['pronunciation']]
        # Optional noise states explain pauses without fitting them as speech phones.
        # Only exterior silence is cropped; the full transcript remains required.
        a, b = r['phonemes'][0]['start']*4, (r['phonemes'][-1]['end']+1)*4
        powers = powers[a:min(b, len(powers))]
        if not ids or len(powers)<len(ids) or (powers<=0).any():
            raise ValueError('Invalid complete canonical pronunciation or spectrum')
        out.append(dict(power=powers, noise=noise, ids=ids, speaker=r['speaker'], source=r['source']))
    if not any(r['source']=='quran-training' for r in out):
        raise ValueError('Authorized original Quran training recordings are required by this experiment')
    if len({r['speaker'] for r in out})<3:
        raise ValueError('At least three non-excluded training speakers are required')
    return out


def main():
    p = argparse.ArgumentParser()
    for name in ('input', 'out', 'tokens', 'inventory', 'excluded-speakers'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    records = [json.loads(s) for s in Path(args.input).open()]
    rows = prepare(records, args.tokens, excluded)
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    results = []
    for strength in STRENGTHS:
        values = []
        for fold in range(3):
            model, _ = fit([r for r in rows if folds[r['speaker']] != fold], args.tokens, strength)
            values.append(validation(model, [r for r in rows if folds[r['speaker']] == fold]))
        result = dict(strength=strength, folds=values, mean=sum(values)/3)
        results.append(result)
        print(json.dumps(result), flush=True)
    selected = max(results, key=lambda r: (r['mean'], r['strength']))
    model, history = fit(rows, args.tokens, selected['strength'])
    model = {k: v.tolist() for k, v in model.items()}
    def q(n):
        x, w = quadrature(n)
        return dict(nodes=x.tolist(), weights=w.tolist())
    model.update(schema=1, inventory=args.inventory, tokens=args.tokens, frameMs=40, bins=BINS,
                 quadrature=q(17), refined=q(33), maxFrames=2048)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    file = out/'noise-power-emissions.json'
    file.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=file.name, bytes=file.stat().st_size, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    (out/'noise-power-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), excluded=sorted(excluded), speakerFolds=folds,
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(), validation=results, selected=selected, history=history, asset=descriptor)
    (out/'noise-power-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

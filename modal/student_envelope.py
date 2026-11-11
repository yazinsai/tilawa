"""CPU frequency-domain linear-prediction pronunciation recheck.

Only help-train training speakers' none-labelled single ayahs can enter fitting.
The emission estimator uses forward-backward transcript occupancies, never
listener spikes or an edit backtrace. Speaker-grouped training-only validation
freezes order, LP regularization and Gaussian shrinkage before any target query.
Run on Modal: python3 modal/student_envelope.py --help-dir "$LAB_DATA/help-train"
--out /scratch/listener-temporal-envelope-poles/fit
"""
import argparse
import hashlib
import json
import math
import wave
from pathlib import Path

BANDS = [80, 400, 900, 1600, 2800, 4500, 7600]
SIZE, HOP, CENTER = 2048, 640, 200


def prediction(coefficients, order, ridge):
    """Levinson recursion along DCT frequency coefficients; matches the SDK."""
    correlation = [sum(x * coefficients[i] for i, x in enumerate(coefficients[k:]))
                   for k in range(order + 1)]
    error, a, reflection = correlation[0] * (1 + ridge) + 1e-12, [], []
    for k in range(order):
        numerator = correlation[k + 1] + sum(v * correlation[k - j] for j, v in enumerate(a))
        r = max(-.999999, min(.999999, -numerator / error))
        a = [v + r * a[k - 1 - j] for j, v in enumerate(a)] + [r]
        error *= 1 - r * r
        reflection.append(math.atanh(r))
    return reflection


def features(audio, order, ridge):
    import numpy as np
    from scipy.fft import dct, rfft
    n = max(0, 1 + (len(audio) - 400) // HOP)
    envelope, control = [], []
    padded = np.pad(audio, (SIZE, SIZE))
    for f in range(n):
        start = f * HOP + CENTER - SIZE // 2 + SIZE
        x = padded[start:start + SIZE].astype(float)
        x -= x.mean()
        energy = np.mean(x * x) + 1e-12
        spectrum = dct(x, type=2) / 2
        pole = [math.log(energy)]
        for b in range(6):
            lo, hi = math.ceil(BANDS[b] * 2 * SIZE / 16000), math.floor(BANDS[b + 1] * 2 * SIZE / 16000)
            c = spectrum[lo:hi] * np.hamming(hi - lo)
            power = np.sum(c * c) / ((hi - lo) * SIZE * energy)
            pole.extend([math.log(power + 1e-12)] + prediction(c.tolist(), order, ridge))
        snapshots = []
        for t in range(order + 1):
            off = math.floor(t * (SIZE - 400) / order + .5)
            z = rfft(x[off:off + 400] * np.hanning(400), n=512)
            row = []
            for b in range(6):
                lo, hi = math.ceil(BANDS[b] * 512 / 16000), math.floor(BANDS[b + 1] * 512 / 16000)
                row.append(math.log(np.sum(np.abs(z[lo:hi]) ** 2) / ((hi - lo) * 400 * energy) + 1e-12))
            snapshots.append(row)
        envelope.append(pole)
        control.append([math.log(energy)] + [row[b] for b in range(6) for row in snapshots])
    return dict(envelope=np.array(envelope), control=np.array(control))


def transitions(length):
    import numpy as np
    a = np.full((length, length), -np.inf)
    for s in range(length):
        if s == length - 1:
            a[s, s] = 0
        elif s % 2:
            a[s, s], a[s, s + 1] = math.log(.6), math.log(.2 if s + 2 < length else .4)
            if s + 2 < length:
                a[s, s + 2] = math.log(.2)
        else:
            a[s, s], a[s, s + 1] = math.log(.6), math.log(.4)
    return a


def forward_backward(emissions, phones, posterior=False):
    import numpy as np
    from scipy.special import logsumexp
    states = np.array([0] + [v for p in phones for v in (p, 0)])
    tmax, size = len(emissions), len(states)
    if not tmax or not phones:
        return -np.inf, None
    e = emissions[:, states]
    a = transitions(size)
    alpha = np.full((tmax, size), -np.inf)
    alpha[0, :2] = math.log(.5) + e[0, :2]
    prior = np.full(size, -np.inf)
    prior[:2] = math.log(.5)
    for t in range(1, tmax):
        # Only self, next and next-phone arcs exist. Avoid a dense state matrix.
        alpha[t] = alpha[t - 1] + np.diag(a) + e[t]
        alpha[t, 1:] = np.logaddexp(alpha[t, 1:], alpha[t - 1, :-1] + np.diag(a, 1) + e[t, 1:])
        alpha[t, 2:] = np.logaddexp(alpha[t, 2:], alpha[t - 1, :-2] + np.diag(a, 2) + e[t, 2:])
        nxt = prior + np.diag(a)
        nxt[1:] = np.logaddexp(nxt[1:], prior[:-1] + np.diag(a, 1))
        nxt[2:] = np.logaddexp(nxt[2:], prior[:-2] + np.diag(a, 2))
        prior = nxt
    partition = logsumexp(alpha[-1, -2:])
    value = float(partition - logsumexp(prior[-2:]))
    if not posterior or not np.isfinite(partition):
        return value, None
    beta = np.full_like(alpha, -np.inf)
    beta[-1, -2:] = 0
    for t in range(tmax - 2, -1, -1):
        row = beta[t + 1] + e[t + 1]
        beta[t] = row + np.diag(a)
        beta[t, :-1] = np.logaddexp(beta[t, :-1], row[1:] + np.diag(a, 1))
        beta[t, :-2] = np.logaddexp(beta[t, :-2], row[2:] + np.diag(a, 2))
    gamma = np.exp(alpha + beta - partition)
    occupancy = np.zeros((tmax, emissions.shape[1]))
    for s, p in enumerate(states):
        occupancy[:, p] += gamma[:, s]
    return value, occupancy


def gaussian(x, emissions):
    import numpy as np
    mean = np.array([e['mean'] for e in emissions])
    variance = np.array([e['variance'] for e in emissions])
    return -.5 * (np.log(2 * np.pi * variance)[None] + (x[:, None] - mean[None]) ** 2 / variance[None]).sum(2)


def manner(p):
    return next((name for chars, name in [('منں۾', 'nasal'), ('بتدطكقءڇ', 'stop'),
        ('ثذفسشصزظخغحعه', 'fricative'), ('رلوي', 'approximant'), ('َُِاۦۥ', 'vowel')]
        if p in chars), 'other')


def estimate(stats, tokens, shrinkage):
    import numpy as np
    weight, total, second = stats
    out = []
    for p in range(len(tokens) + 1):
        pool = [i for i in range(len(tokens) + 1) if i == p or
                (p and i and manner(tokens[i - 1]) == manner(tokens[p - 1]))]
        mass = weight[pool].sum()
        if mass <= 0:
            pool = list(range(len(tokens) + 1))
            mass = weight.sum()
        prior_mean = total[pool].sum(0) / mass
        prior_second = second[pool].sum(0) / mass
        mean = (total[p] + shrinkage * prior_mean) / (weight[p] + shrinkage)
        raw_second = (second[p] + shrinkage * prior_second) / (weight[p] + shrinkage)
        out.append(dict(mean=mean.tolist(), variance=np.maximum(raw_second - mean ** 2, .05).tolist()))
    return out


def fit(rows, tokens, inventory, order, ridge, shrinkage, epochs=4):
    import numpy as np
    model = dict(schema=1, inventory=inventory, tokens=tokens, frameMs=40, order=order,
        ridge=ridge, maxFrames=4096, phoneCounts=[sum(r['pronunciation'].count(p) for r in rows) for p in tokens])
    speaker_counts = {s: sum(r['speaker'] == s for r in rows) for s in {r['speaker'] for r in rows}}
    history = {}
    for arm in ['envelope', 'control']:
        d = rows[0][arm].shape[1]
        scale = np.maximum(np.concatenate([r[arm] for r in rows]).std(0), .1)
        model[arm + 'Scale'] = scale.tolist()
        emission, logs = None, []
        for epoch in range(epochs + 1):
            stats = (np.zeros(len(tokens) + 1), np.zeros((len(tokens) + 1, d)), np.zeros((len(tokens) + 1, d)))
            likelihoods = []
            for r in rows:
                x = r[arm] / scale
                phones = [tokens.index(p) + 1 for p in r['pronunciation']]
                if emission is None:
                    occupancy = np.zeros((len(x), len(tokens) + 1))
                    # Only initialization uses proportional placement. Subsequent
                    # updates integrate every transcript-consistent boundary.
                    for t in range(len(x)):
                        q = (t / len(x) - .1) / .8
                        p = 0 if q < 0 or q >= 1 else phones[min(len(phones) - 1, int(q * len(phones)))]
                        occupancy[t, p] = 1
                else:
                    value, occupancy = forward_backward(gaussian(x, emission), phones, True)
                    if occupancy is None:
                        raise ValueError('No complete transcript placement in training recording')
                    likelihoods.append(value / len(x))
                w = occupancy * (200 / len(x) / speaker_counts[r['speaker']])
                stats[0][:] += w.sum(0)
                stats[1][:] += w.T @ x
                stats[2][:] += w.T @ (x * x)
            emission = estimate(stats, tokens, shrinkage)
            logs.append(dict(epoch=epoch, meanFrameLogP=float(np.mean(likelihoods)) if likelihoods else None))
        model[arm] = emission
        history[arm] = logs
    return model, history


def score(model, x, pronunciation, arm):
    import numpy as np
    if not pronunciation or len(x) > model['maxFrames'] or any(p not in model['tokens'] or
            model['phoneCounts'][model['tokens'].index(p)] <= 0 for p in pronunciation):
        return None
    phones = [model['tokens'].index(p) + 1 for p in pronunciation]
    value, _ = forward_backward(gaussian(x / np.array(model[arm + 'Scale']), model[arm]), phones)
    return value if np.isfinite(value) else None


def read_training(root, reference, extra_excluded):
    """Reject contradictory labels and speakers with any nontraining split."""
    import numpy as np
    from scipy.signal import resample_poly
    labels = [json.loads(line) for line in (root / 'labels.jsonl').read_text().splitlines() if line.strip()]
    if any(l.get('split') not in ('train', 'eval', 'dev', 'test', 'heldout', 'validation') or
           not isinstance(l.get('speaker'), str) or not l['speaker'] for l in labels):
        raise ValueError('Explicit speaker and train/held-out split metadata required')
    excluded = extra_excluded | {l['speaker'] for l in labels if l['split'] != 'train'}
    ayahs = {f"{a['s']}:{a['a']}": a for a in reference['ayahs']}
    rows = []
    for l in labels:
        if (l.get('kind') != 'none' or l['split'] != 'train' or l['speaker'] in excluded or
                l.get('excluded') or l.get('scored') is False):
            continue
        if l.get('events') or l.get('non_errors') or l.get('unsupported') or len(l.get('ayahs', [])) != 1:
            raise ValueError('None take has contradictory labels or lacks one canonical ayah')
        path = (root / l['wav']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Training audio escapes help-train root')
        with wave.open(str(path)) as w:
            if w.getsampwidth() != 2:
                raise ValueError('Training extractor requires PCM16 WAV')
            rate, channels = w.getframerate(), w.getnchannels()
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').reshape(-1, channels).mean(1) / 32768
        if rate != 16000:
            divisor = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor)
        audio = audio.astype(np.float32)
        if len(audio) < 400 or 1 + (len(audio) - 400) // HOP > 4096:
            raise ValueError('Training take has no observations or exceeds the frozen bound')
        a = ayahs[l['ayahs'][0]]
        pronunciation = a['p'][:a['o'][-1]] + a['iso'][-1]
        if not pronunciation or any(p not in reference['inventory']['tokens'] for p in pronunciation):
            raise ValueError('Training transcript outside canonical inventory')
        rows.append(dict(speaker=l['speaker'], pronunciation=pronunciation, audio=audio,
            audioSha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if len({r['speaker'] for r in rows}) < 3:
        raise ValueError('At least three eligible none-labelled training speakers required')
    return rows, excluded


def main():
    import numpy as np
    import scipy
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--help-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--excluded-speakers', type=Path)
    args = parser.parse_args()
    if args.help_dir.resolve().name != 'help-train':
        raise ValueError('This fit accepts only the authorized help-train root')
    reference = json.loads((Path(__file__).resolve().parents[1] / 'packages/core/assets/ref-v2.json').read_text())
    extra = set(json.loads(args.excluded_speakers.read_text())) if args.excluded_speakers else set()
    rows, excluded = read_training(args.help_dir, reference, extra)
    tokens = reference['inventory']['tokens']
    inventory = reference['inventory']['id']
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    candidates = []
    # Selection uses generic one-phone contrasts on held-in training speakers,
    # not incomparable likelihoods from different feature dimensions.
    for order in [2, 4]:
        for ridge in [.001, .01]:
            observed = [{k: v for k, v in r.items() if k != 'audio'} | features(r['audio'], order, ridge) for r in rows]
            for shrinkage in [8, 32]:
                losses = []
                for fold in range(3):
                    training = [r for r in observed if folds[r['speaker']] != fold]
                    m, _ = fit(training, tokens, inventory, order, ridge, shrinkage)
                    by_speaker = {}
                    for r in observed:
                        if folds[r['speaker']] != fold:
                            continue
                        p = r['pronunciation']
                        available = [c for c in tokens if m['phoneCounts'][tokens.index(c)] > 0]
                        h = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
                        pos = h % len(p)
                        alternatives = [c for c in available if c != p[pos]]
                        if not alternatives:
                            raise ValueError('No covered competing phones for training-only validation')
                        rival = p[:pos] + alternatives[h % len(alternatives)] + p[pos + 1:]
                        for arm in ['envelope', 'control']:
                            good, bad = score(m, r[arm], p, arm), score(m, r[arm], rival, arm)
                            if good is None or bad is None:
                                continue
                            by_speaker.setdefault(r['speaker'], []).append(float(np.logaddexp(0, (bad - good) / len(r[arm]))))
                    if not by_speaker:
                        raise ValueError('No covered training-only validation contrasts')
                    losses.append(float(np.mean([np.mean(v) for v in by_speaker.values()])))
                result = dict(order=order, ridge=ridge, shrinkage=shrinkage, foldLoss=losses, loss=float(np.mean(losses)))
                candidates.append(result)
                print(json.dumps(result), flush=True)
    chosen = min(candidates, key=lambda r: (r['loss'], r['order'], -r['shrinkage'], -r['ridge']))
    observed = [{k: v for k, v in r.items() if k != 'audio'} | features(r['audio'], chosen['order'], chosen['ridge']) for r in rows]
    model, history = fit(observed, tokens, inventory, chosen['order'], chosen['ridge'], chosen['shrinkage'])
    args.out.mkdir(parents=True, exist_ok=True)
    asset = args.out / 'envelope-emissions.json'
    asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (args.out / 'envelope-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), excludedSpeakers=sorted(excluded),
        runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        speakerFolds=folds, selected=chosen, validation=candidates, history=history, asset=descriptor,
        recordingsProvenance=[dict(speaker=r['speaker'], audioSha256=r['audioSha256']) for r in rows])
    (args.out / 'envelope-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(recordings=len(rows), speakers=len(speakers), selected=chosen, asset=descriptor)), flush=True)


if __name__ == '__main__':
    main()

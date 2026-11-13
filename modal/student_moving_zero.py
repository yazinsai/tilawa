"""CPU-only moving pole-zero pronunciation model; no practice inputs in fitting.

Three numerator root pairs are unordered observations. One zero has a single
window-shared center; two articulatory zeros have marginalized correspondence.
The moving arm is a phone-conditioned AR(1) density. The static control is a
two-component diagonal mixture with precisely the same parameter count. Both
use the same pole emissions, nuisance centers, CTC timing and accepting states.
"""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

PERMS = np.array(list(itertools.permutations(range(3))))
SHRINKAGES = (16, 64)
EPOCHS = 8
STAY = .8
MAX_FRAMES = 2048
ZERO_NOISE = np.array([.06, .10]) ** 2
FLOOR = .0025


def root_pairs(coefficients):
    """Minimum-phase representatives; real roots are paired on the real axis."""
    roots = np.roots(coefficients)
    if not np.isfinite(roots).all():
        raise ValueError('Nonfinite rational roots')
    pairs = [z for z in roots if z.imag > 1e-6]
    real = sorted([z for z in roots if abs(z.imag) <= 1e-6], key=lambda z: z.real)
    for a, b in zip(real[::2], real[1::2]):
        pairs.append(complex((a.real+b.real)/2, 0))
    if len(pairs) != (len(coefficients)-1)//2:
        raise ValueError('Unresolved conjugate roots')
    values = []
    for z in pairs:
        r = np.clip(min(abs(z), 1/max(abs(z), 1e-8)), .01, .999)
        values.append([abs(np.angle(z))/np.pi, -np.log(r)/np.pi])
    return np.array(sorted(values))


def spectrum_roots(power):
    """Fit A(z)H(z)=B(z) to a liftered minimum-phase spectrum.

    Fixed degrees 8/6, 16 cepstral coefficients, three weighted linear solves.
    This estimates rational roots, rather than picking independent local minima.
    FFT samples and all regularization match the SDK extractor.
    """
    power = np.asarray(power, dtype=float)
    # Relative floor preserves multiplicative recording gain before smoothing.
    logmag = .5*np.log(np.maximum(power/max(power.mean(), 1e-30), 1e-12))
    logmag -= logmag.mean()
    cep = np.fft.ifft(logmag).real
    minimum = np.zeros(512)
    minimum[0], minimum[1:17] = cep[0], 2*cep[1:17]
    h = np.exp(np.fft.fft(minimum))[np.arange(1, 256, 4)]
    z = np.exp(-2j*np.pi*np.arange(1, 256, 4)/512)
    a = np.ones(9)
    a[1:] = 0
    basis_a = z[:, None]**np.arange(1, 9)
    basis_b = z[:, None]**np.arange(7)
    design = np.concatenate([h[:, None]*basis_a, -basis_b], axis=1)
    for _ in range(3):
        weight = 1/np.maximum(abs(h), .1)/np.maximum(abs(1+basis_a@a[1:]), .1)
        x = np.concatenate([(design*weight[:, None]).real, (design*weight[:, None]).imag])
        y = np.r_[(-h*weight).real, (-h*weight).imag]
        c = np.linalg.solve(x.T@x+.01*np.eye(15), x.T@y)
        a = np.r_[1., c[:8]]
    return root_pairs(a), root_pairs(c[8:])


def observations(audio):
    """16 kHz, snipped 25 ms Povey/preemphasis windows every 40 ms."""
    audio = np.asarray(audio, dtype=float)
    window = (.5-.5*np.cos(2*np.pi*np.arange(400)/399))**.85
    rows = []
    for off in range(0, len(audio)-399, 640):
        x = audio[off:off+400].copy()
        x -= x.mean()
        x -= .97*np.r_[x[0], x[:-1]]
        power = abs(np.fft.fft(x*window, 512))**2
        energy = float(np.log(max(power.sum()/512, 1e-12)))
        try:
            poles, zeros = spectrum_roots(power)
            f, b = poles.T
            # Symmetric pole summaries avoid a second arbitrary pole ordering.
            features = [float(np.mean(f**j)) for j in range(1, 5)]
            features += [float(b.mean()), float((b*b).mean()), float((f*b).mean()), energy]
            valid = bool(power.sum() > 1e-9)
        except (ValueError, np.linalg.LinAlgError):
            features, zeros, valid = [0.]*7+[energy], np.zeros((3, 2)), False
        rows.append(dict(poles=features, zeros=zeros.tolist(), valid=valid))
    return rows


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


def normal(x, mean, variance):
    return -.5*(np.log(2*np.pi*variance)+(x-mean)**2/variance).sum(-1)


def prepare_window(rows):
    poles = np.array([r['poles'] for r in rows])
    zeros = np.array([r['zeros'] for r in rows])
    valid = np.array([r['valid'] for r in rows], dtype=bool)
    x = zeros[:, PERMS[:, 1:]].reshape(len(rows), 6, 4)
    stationary = zeros[:, PERMS[:, 0]]
    # Three deterministic quadrature centers; latent choice is shared by ALL
    # frames and both pronunciations, never estimated using candidate text.
    ordered = np.sort(zeros[valid, :, 0], axis=1)
    ranked = zeros[valid][np.arange(valid.sum())[:, None], np.argsort(zeros[valid, :, 0], axis=1)]
    centers = ranked.mean(0) if len(ordered) else np.zeros((3, 2))
    nuisance = normal(stationary[:, None], centers[None, :, None], ZERO_NOISE)
    nuisance[~valid] = 0
    return dict(poles=poles, x=x, valid=valid, nuisance=nuisance)


def likelihoods(model, w, moving):
    arm = model['moving' if moving else 'control']
    mean, var = np.array(arm['mean']), np.array(arm['variance'])
    poles = normal(w['poles'][:, None], np.array(model['poles']['mean']), np.array(model['poles']['variance']))
    # t, phone, previous permutation, current permutation. Root probabilities
    # sum over all six assignments, including which root is the channel zero.
    initial = normal(w['x'][:, None], mean[None, :, 0, None], var[None, :, 0, None])
    if moving:
        alpha = np.array(arm['alpha'])
        noise = np.array(arm['noise'])
        predicted = mean[None, :, 0, None]+alpha[None, :, None]*(w['x'][:-1, None]-mean[None, :, 0, None])
        transitions = normal(w['x'][1:, None, None], predicted[:, :, :, None], noise[None, :, None, None])
    else:
        mix = logsumexp(normal(w['x'][:, None, None], mean[None, :, :, None], var[None, :, :, None]), axis=2)-np.log(2)
        initial = mix
        transitions = np.repeat(mix[1:, :, None], 6, axis=2)
    initial[~w['valid']] = 0
    transitions[~w['valid'][1:]] = 0
    reset = ~w['valid'][:-1] & w['valid'][1:]
    transitions[reset] = np.repeat(initial[1:][reset, :, None], 6, axis=2)
    return initial, transitions, poles


def chain(model, w, labels, moving, posterior=False):
    states, skip, selfp, step = topology(labels)
    initial, trans, pole = likelihoods(model, w, moving)
    n, tmax = len(states), len(w['x'])
    a = np.full((tmax, 3, n, 6), -np.inf)
    a[0, :, :2] = initial[0, states[:2]][None]+pole[0, states[:2]][None, :, None]+w['nuisance'][0, :, None]-np.log(36)
    prior = np.full(n, -np.inf)
    prior[:2] = -np.log(2)
    def incoming(row):
        v = row+selfp[None, :, None]
        v[:, 1:] = np.logaddexp(v[:, 1:], row[:, :-1]+step[None, :-1, None])
        v[:, 2:] = np.logaddexp(v[:, 2:], np.where(skip[None, 2:, None], row[:, :-2]+step[None, :-2, None], -np.inf))
        return v
    for t in range(1, tmax):
        v = incoming(a[t-1])
        a[t] = logsumexp(v[:, :, :, None]+trans[t-1, states][None], axis=2)-np.log(6)
        a[t] += pole[t, states][None, :, None]+w['nuisance'][t, :, None]
        p = prior+selfp
        p[1:] = np.logaddexp(p[1:], prior[:-1]+step[:-1])
        p[2:] = np.logaddexp(p[2:], np.where(skip[2:], prior[:-2]+step[:-2], -np.inf))
        prior = p
    mass = logsumexp(a[-1, :, -2:])
    if not np.isfinite(mass):
        if posterior:
            raise ValueError('Incomplete pronunciation has no accepting path')
        return float('-inf')
    score = float(mass-np.logaddexp(prior[-1], prior[-2]))
    if not posterior:
        return score
    if not np.isfinite(score):
        raise ValueError('Incomplete pronunciation has no accepting path')
    b = np.full_like(a, -np.inf)
    b[-1, :, -2:] = 0
    for t in range(tmax-2, -1, -1):
        future = pole[t+1, states][None, :, None]+w['nuisance'][t+1, :, None]+b[t+1]
        v = logsumexp(trans[t, states][None]+future[:, :, None, :], axis=3)-np.log(6)
        b[t] = v+selfp[None, :, None]
        b[t, :, :-1] = np.logaddexp(b[t, :, :-1], v[:, 1:]+step[None, :-1, None])
        b[t, :, :-2] = np.logaddexp(b[t, :, :-2], np.where(skip[None, 2:, None], v[:, 2:]+step[None, :-2, None], -np.inf))
    gamma = np.exp(a+b-mass).sum(1)
    pairs = np.zeros((tmax-1, len(model['tokens'])+1, 6, 6))
    if moving:
        for t in range(1, tmax):
            joint = np.exp(incoming(a[t-1])[:, :, :, None]+trans[t-1, states][None]-np.log(6)+
                pole[t, states][None, :, None, None]+w['nuisance'][t, :, None, None, :]+b[t, :, :, None, :]-mass).sum(0)
            for s, p in enumerate(states):
                pairs[t-1, p] += joint[s]
    occupancies = np.zeros((tmax, len(model['tokens'])+1, 6))
    for s, p in enumerate(states):
        occupancies[:, p] += gamma[:, s]
    return score, occupancies, pairs


def shrink(count, sums, squares, strength, floor=FLOOR):
    pooled_mean = sums.sum(0)/max(count.sum(), 1e-12)
    pooled_second = squares.sum(0)/max(count.sum(), 1e-12)
    mu = (sums+strength*pooled_mean)/(count[:, None]+strength)
    var = np.maximum((squares+strength*pooled_second)/(count[:, None]+strength)-mu*mu, floor)
    return mu, var


def fit(records, tokens, strength, deadline=None):
    vocab = len(tokens)+1
    raw = np.concatenate([r['window']['poles'] for r in records])
    center, scale = raw.mean(0), np.maximum(raw.std(0), .01)
    rows = [dict(r, window={**r['window'], 'poles': (r['window']['poles']-center)/scale}) for r in records]
    zero = np.concatenate([r['window']['x'][r['window']['valid']].reshape(-1, 4) for r in rows])
    mu, var = zero.mean(0), np.maximum(zero.var(0), FLOOR)
    control = dict(mean=np.tile(mu, (vocab, 2, 1)).tolist(), variance=np.tile(var, (vocab, 2, 1)).tolist())
    # Break mixture symmetry without adding phone/word-specific initialization.
    for p in range(vocab):
        control['mean'][p][0] = (mu-.1*np.sqrt(var)).tolist()
        control['mean'][p][1] = (mu+.1*np.sqrt(var)).tolist()
    model = dict(schema=1, tokens=tokens, frameMs=40, stay=STAY, maxFrames=MAX_FRAMES, center=center.tolist(), scale=scale.tolist(),
        stationaryVariance=ZERO_NOISE.tolist(), phoneCounts=[sum(r['pronunciation'].count(p) for r in rows) for p in tokens],
        poles=dict(mean=np.zeros((vocab, 8)).tolist(), variance=np.ones((vocab, 8)).tolist()), control=control)
    history = {}
    for moving in (False, True):
        if moving:
            cm = np.array(model['control']['mean'])
            cv = np.array(model['control']['variance'])
            m = cm.mean(1)
            v = np.maximum((cv+cm*cm).mean(1)-m*m, FLOOR)
            model['moving'] = dict(mean=m[:, None].tolist(), variance=v[:, None].tolist(),
                alpha=np.full((vocab, 4), .8).tolist(), noise=np.maximum(v*.36, FLOOR).tolist())
        losses = []
        for epoch in range(EPOCHS):
            if deadline is not None and time.process_time() >= deadline:
                raise TimeoutError('Training exceeded its reserved CPU budget')
            pc, ps, pq = np.zeros(vocab), np.zeros((vocab, 8)), np.zeros((vocab, 8))
            zc, zs, zq = np.zeros((vocab, 2)), np.zeros((vocab, 2, 4)), np.zeros((vocab, 2, 4))
            cc, cx, cy, cxx, cxy, cyy = [np.zeros((vocab, 4)) for _ in range(6)]
            scores = []
            for r in rows:
                if deadline is not None and time.process_time() >= deadline:
                    raise TimeoutError('Training exceeded its reserved CPU budget')
                w = r['window']
                score, gamma, pairs = chain(model, w, r['ids'], moving, True)
                scores.append(score/len(w['x']))
                gp = gamma.sum(2)
                pc += gp.sum(0); ps += gp.T@w['poles']; pq += gp.T@(w['poles']**2)
                if not moving:
                    arm = model['control']
                    responsibility = normal(w['x'][:, None, None], np.array(arm['mean'])[None, :, :, None], np.array(arm['variance'])[None, :, :, None])
                    responsibility = np.exp(responsibility-logsumexp(responsibility, axis=2, keepdims=True))
                    weights = gamma[:, :, None]*responsibility*w['valid'][:, None, None, None]
                    zc += weights.sum((0, 3))
                    zs += np.einsum('tpmj,tjd->pmd', weights, w['x'])
                    zq += np.einsum('tpmj,tjd->pmd', weights, w['x']**2)
                else:
                    weights = gamma*w['valid'][:, None, None]
                    zc[:, 0] += weights.sum((0, 2))
                    zs[:, 0] += np.einsum('tpj,tjd->pd', weights, w['x'])
                    zq[:, 0] += np.einsum('tpj,tjd->pd', weights, w['x']**2)
                    q = pairs*(w['valid'][1:] & w['valid'][:-1])[:, None, None, None]
                    cc += q.sum((0, 2, 3))[:, None]
                    cx += np.einsum('tpij,tid->pd', q, w['x'][:-1]); cy += np.einsum('tpij,tjd->pd', q, w['x'][1:])
                    cxx += np.einsum('tpij,tid->pd', q, w['x'][:-1]**2); cyy += np.einsum('tpij,tjd->pd', q, w['x'][1:]**2)
                    cxy += np.einsum('tpij,tid,tjd->pd', q, w['x'][:-1], w['x'][1:])
            if not moving:
                pm, pv = shrink(pc, ps, pq, strength, .05)
                model['poles'] = dict(mean=pm.tolist(), variance=pv.tolist())
                zm, zv = shrink(zc.reshape(-1), zs.reshape(-1, 4), zq.reshape(-1, 4), strength)
                model['control'] = dict(mean=zm.reshape(vocab, 2, 4).tolist(), variance=zv.reshape(vocab, 2, 4).tolist())
            else:
                m, v = shrink(zc[:, 0], zs[:, 0], zq[:, 0], strength)
                xx = cxx-2*m*cx+cc*m*m
                xy = cxy-m*(cx+cy)+cc*m*m
                yy = cyy-2*m*cy+cc*m*m
                alpha = np.clip((xy+strength*.8*v)/(xx+strength*v), -.98, .98)
                residual = np.maximum((yy-2*alpha*xy+alpha*alpha*xx+strength*v*(1-.8**2))/(cc+strength), FLOOR)
                model['moving'] = dict(mean=m[:, None].tolist(), variance=v[:, None].tolist(), alpha=alpha.tolist(), noise=residual.tolist())
            losses.append(float(np.mean(scores)))
        history['moving' if moving else 'control'] = losses
    return model, history


def score(model, row, labels, moving):
    w = {**row['window'], 'poles': (row['window']['poles']-model['center'])/model['scale']}
    return chain(model, w, labels, moving)


def validation(model, rows, deadline=None):
    groups = {}
    phones = [i+1 for i, n in enumerate(model['phoneCounts']) if n]
    for r in rows:
        if deadline is not None and time.process_time() >= deadline:
            raise TimeoutError('Validation exceeded its reserved CPU budget')
        rng = np.random.default_rng(int(hashlib.sha256((r['speaker']+r['pronunciation']).encode()).hexdigest()[:8], 16))
        alternatives = [r['ids']]
        for _ in range(3):
            labels = r['ids'].copy(); at = int(rng.integers(len(labels)))
            labels[at] = int(rng.choice([p for p in phones if p != labels[at]]))
            alternatives.append(labels)
        losses = []
        for moving in (False, True):
            scores = np.array([score(model, r, ids, moving) for ids in alternatives])
            losses.append(float(logsumexp(scores)-scores[0]))
        groups.setdefault(r['speaker'], []).append(np.mean(losses))
    return float(np.mean([np.mean(v) for v in groups.values()]))


def prepare(records, tokens, excluded):
    rows = []
    for r in records:
        if r.get('source') != 'help-train' or r.get('kind') != 'none' or r.get('split') != 'train' or r.get('excluded') is not False or r['speaker'] in excluded:
            raise ValueError('Only none-labelled help-train training speakers accepted')
        p = r['pronunciation']
        if not p or any(c not in tokens for c in p):
            raise ValueError('Invalid complete canonical pronunciation')
        obs = r['observations']
        if len(obs) < len(p)+sum(a == b for a, b in zip(p, p[1:])) or len(obs) > MAX_FRAMES:
            raise ValueError('Invalid duration')
        for f in obs:
            if len(f['poles']) != 8 or np.array(f['zeros']).shape != (3, 2) or not isinstance(f['valid'], bool) or not np.isfinite(f['poles']+np.array(f['zeros']).flatten().tolist()).all():
                raise ValueError('Invalid root observations')
        if not any(f['valid'] for f in obs):
            raise ValueError('No resolved spectral roots')
        rows.append(dict(window=prepare_window(obs), ids=[tokens.index(c)+1 for c in p], pronunciation=p, speaker=r['speaker']))
    if len({r['speaker'] for r in rows}) < 3:
        raise ValueError('At least three eligible speakers required')
    return rows


def main():
    # Extraction reserves 30 CPU minutes; fitting gets the rest of one CPU day.
    deadline = time.process_time()+84600
    parser = argparse.ArgumentParser()
    for key in ('input', 'out', 'excluded-speakers'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    meta = json.loads(Path(args.input+'.inventory.json').read_text())
    tokens = meta['tokens']
    excluded = set(json.loads(Path(args.excluded_speakers).read_text()))
    rows = prepare([json.loads(line) for line in Path(args.input).open()], tokens, excluded)
    speakers = sorted({r['speaker'] for r in rows}, key=lambda s: hashlib.sha256(s.encode()).digest())
    folds = {s: i % 3 for i, s in enumerate(speakers)}
    results = []
    for strength in SHRINKAGES:
        losses = []
        for fold in range(3):
            model, _ = fit([r for r in rows if folds[r['speaker']] != fold], tokens, strength, deadline)
            losses.append(validation(model, [r for r in rows if folds[r['speaker']] == fold], deadline))
        result = dict(shrinkage=strength, foldLoss=losses, meanLoss=float(np.mean(losses)))
        results.append(result); print(json.dumps(result), flush=True)
    selected = min(results, key=lambda r: (r['meanLoss'], -r['shrinkage']))
    model, history = fit(rows, tokens, selected['shrinkage'], deadline)
    model['inventory'] = meta['inventory']
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    asset = out/'moving-zero-emissions.json'; asset.write_text(json.dumps(model, allow_nan=False))
    descriptor = dict(schema=1, file=asset.name, bytes=asset.stat().st_size, sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    (out/'moving-zero-asset.json').write_text(json.dumps(descriptor))
    report = dict(recordings=len(rows), speakers=len(speakers), speakerFolds=folds, excludedSpeakers=sorted(excluded),
        inputSha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(), validation=results, selected=selected, history=history, asset=descriptor)
    (out/'moving-zero-training-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(selected=selected, recordings=len(rows), speakers=len(speakers), asset=descriptor)), flush=True)


if __name__ == '__main__':
    main()

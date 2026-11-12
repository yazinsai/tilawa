"""Pronunciation-conditioned reconstruction using passive digital waveguides.

This side judge never changes the CTC listener. Its ten physical coordinates are
six oral areas, velum opening, nasal area scale, tract length and wall loss.
Every tube is a bidirectional, lossy fractional delay; pressure continuity and
volume-flow conservation couple the oral tract and nasal side branch. Separate
glottal and constriction sources share recording nuisances estimated *before*
the entire comparison interval. We integrate source phase in a Gaussian
waveform covariance likelihood rather than aligning glottal cycles to text.

The equal-coordinate-count control is a stable, unconstrained ARMA source-filter
model. It is an experimental control, never a production certificate condition.
No Modal imports or local-path operations occur at module import time.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

RATE, FFT, HOP, DIM = 8000, 256, 128, 10
# Bounds express geometry and passivity, not an acoustic decision threshold.
LOW = np.array([.025]*6 + [.001, .45, 13., .002])
HIGH = np.array([8.]*6 + [2., 1.8, 21., .09])


def geometry(u):
    lo = torch.as_tensor(LOW, device=u.device, dtype=u.dtype)
    hi = torch.as_tensor(HIGH, device=u.device, dtype=u.dtype)
    # Log areas span closures through open vowels without negative impedances.
    return torch.exp(torch.log(lo) + torch.sigmoid(u)*(torch.log(hi)-torch.log(lo)))


def tube(area, length, loss, z):
    """Two-port admittance of a stable first-order allpass delay line.

    q is one-way propagation. The two admittances are exactly the sum/difference
    obtained by eliminating forward/backward travelling pressure waves.
    """
    delay = length*RATE/35000.
    a = (1-delay)/(1+delay)
    q = torch.exp(-loss*length)[..., None]*(a[..., None]+z)/(1+a[..., None]*z)
    denominator = 1-q*q
    return area[..., None]*(1+q*q)/denominator, -2*area[..., None]*q/denominator


def transfer(u):
    """Return glottal and constriction-to-radiated-flow transfer functions.

    The branch shares node 2 with the oral path. Positive tube areas, attenuated
    delay lines and positive-real radiation/glottal loads ensure passivity.
    An explicit nodal solve enforces conservation, including nasal antiresonance.
    """
    p = geometry(u)
    frequency = torch.arange(FFT//2+1, device=u.device, dtype=u.dtype)
    z = torch.exp(-2j*math.pi*frequency/FFT)
    shape = (*u.shape[:-1], len(frequency), 11, 11)
    y = torch.zeros(shape, device=u.device, dtype=torch.complex64)
    edges = [(i, i+1, p[..., i], p[..., 8]/6) for i in range(6)]
    edges += [(2, 7, p[..., 6], p[..., 8]/12)]
    edges += [(7+i, 8+i, p[..., 7]*(1.2, 1.8, 1.4)[i], p[..., 8]/4) for i in range(3)]
    for left, right, area, length in edges:
        diagonal, off = tube(area, length, p[..., 9], z)
        y[..., left, left] += diagonal
        y[..., right, right] += diagonal
        y[..., left, right] += off
        y[..., right, left] += off
    oral_load = 1/(.25+.12*(1-z))
    nasal_load = 1/(.5+.18*(1-z))
    y[..., 0, 0] += .2
    y[..., 6, 6] += oral_load
    y[..., 10, 10] += nasal_load
    source = torch.zeros((*shape[:-1], 2), device=u.device, dtype=torch.complex64)
    source[..., 0, 0] = 1
    # Turbulence concentrates at actual oral constrictions, independently of
    # glottal excitation. A smooth placement keeps the solver differentiable.
    constriction = torch.softmax(-p[..., :6]/.3, -1)
    source[..., 1:7, 1] = constriction[..., None, :]
    pressure = torch.linalg.solve(y, source)
    radiation = pressure[..., 6, :]*oral_load[..., None]+pressure[..., 10, :]*nasal_load[..., None]
    return radiation[..., 0], radiation[..., 1]


def control_transfer(u):
    """Matched ten-coordinate stable lattice plus minimum-phase FIR zeros."""
    frequency = torch.arange(FFT//2+1, device=u.device, dtype=u.dtype)
    z = torch.exp(-2j*math.pi*frequency/FFT)
    ar = torch.ones((*u.shape[:-1], 1), device=u.device, dtype=u.dtype)
    for k in .92*torch.tanh(u[..., :5]).unbind(-1):
        old = ar
        ar = torch.cat([old, torch.zeros_like(old[..., :1])], -1)
        ar = ar + k[..., None]*torch.cat([torch.zeros_like(old[..., :1]), old.flip(-1)], -1)
    az = sum(ar[..., j, None]*z**j for j in range(6))
    numerator = torch.ones_like(az)
    for b in .92*torch.tanh(u[..., 5:]).unbind(-1):
        numerator = numerator*(1+b[..., None]*z)
    h = numerator/az
    return h, h*(1-z)


def gates(phone):
    """Articulation-independent physiological source permissions, shared arms."""
    if phone in '_ؙۜ':
        return 0., 0.
    if phone in 'ثفسشصخحه':
        return .05, 1.
    if phone in 'بتدطكقءڇ':
        return .2, .7
    return 1., .08


def spectra(audio):
    audio = torch.as_tensor(audio, dtype=torch.float32)
    if len(audio) < FFT:
        return torch.empty((0, FFT//2+1))
    window = torch.hann_window(FFT)
    frames = audio.unfold(0, FFT, HOP)*window
    return torch.fft.rfft(frames).abs().square()/window.square().sum()


def read_audio(path):
    rate, audio = wavfile.read(path)
    if audio.dtype.kind in 'iu':
        bits = np.iinfo(audio.dtype).bits
        midpoint = float(2**(bits-1))
        audio = (audio.astype(np.float32)-(midpoint if audio.dtype.kind == 'u' else 0))/midpoint
    if audio.ndim > 1:
        audio = audio.mean(1)
    gcd = math.gcd(rate, RATE)
    return resample_poly(audio, RATE//gcd, rate//gcd).astype(np.float32)


def nuisance(audio, before_sample):
    """One causal recording estimate, with no target samples or target text.

    Robust smooth channel and source power/tilt are intentionally inseparable;
    either pronunciation receives precisely the same combined nuisance. The
    disputed frame's level is never re-normalized separately for a hypothesis.
    """
    outside = np.asarray(audio)[max(0, before_sample-30*RATE):before_sample]
    x = spectra(outside)
    if len(x) < 32:
        return None
    power = x.numpy()
    level = power.sum(1)
    speech = power[level >= np.quantile(level, .5)]
    quiet = power[level <= np.quantile(level, .2)]
    raw = np.maximum(np.median(speech, 0), 1e-9)
    # Fixed broad smoothing cannot fit phone-specific resonances.
    smooth = np.exp(np.convolve(np.pad(np.log(raw), (16, 16), mode='edge'), np.ones(33)/33, mode='valid'))
    noise = np.maximum(np.median(quiet, 0), 1e-10)
    # Phase-independent voiced comb from the outside waveform's autocorrelation.
    blocks = outside[:len(outside)//FFT*FFT].reshape(-1, FFT)
    ac = np.fft.irfft(np.abs(np.fft.rfft(blocks*np.hanning(FFT)))**2)
    lag = 32+np.argmax(np.mean(ac[:, 32:101], axis=0))
    f0 = RATE/lag
    f = np.arange(FFT//2+1)*RATE/FFT
    harmonic_distance = np.abs(f-f0*np.round(f/f0))
    voiced = .08+np.exp(-.5*(harmonic_distance/35)**2)
    return {"channel": smooth.tolist(), "noise": noise.tolist(), "voiced": voiced.tolist(),
            "beforeSample": int(before_sample), "outsideSamples": int(len(outside))}


def covariance(u, phones, shared, physical):
    hg, ht = transfer(u) if physical else control_transfer(u)
    dev = u.device
    channel = torch.as_tensor(shared['channel'], device=dev)
    noise = torch.as_tensor(shared['noise'], device=dev)
    voiced = torch.as_tensor(shared['voiced'], device=dev)
    permissions = torch.as_tensor([gates(p) for p in phones], device=dev)
    # Shapes: phone x sample x frequency. A fixed reference response gives both
    # arms equal source units; no candidate-specific channel or gain is fitted.
    source = permissions[..., 0, None, None]*voiced*hg.abs().square()
    source += permissions[..., 1, None, None]*ht.abs().square()
    return (channel*source+noise).clamp_min(1e-10)


def emissions(x, u, phones, shared, physical):
    variance = covariance(u, phones, shared, physical)
    # Circular complex Gaussian FFT likelihood is a time-domain covariance
    # reconstruction score under a unitary transform; phase is marginalized.
    # Ignore DC/Nyquist, whose coefficients are real and have other weights.
    return -(torch.log(variance[None, ..., 1:-1])+x[:, None, None, 1:-1]/variance[None, ..., 1:-1]).sum(-1)


def sequence_score(e, attack=None):
    """All complete-phone placements; one latent realization per phone run.

    Each state has two entry frames and a sustain state. Either entry state can
    advance, so short phones are possible. Staying cannot reselect a tract to
    explain each frame independently. Normalize the same timing/realization
    prior by running the identical recurrence with zero acoustic scores.
    """
    tmax, n, samples = e.shape
    if not n or tmax < n:
        return e.new_tensor(-float('inf'))
    if attack is None:
        attack = e[..., None].expand(-1, -1, -1, 2)
    def forward(scores, entry):
        previous = scores.new_full((n, samples, 3), -float('inf'))
        # Avoid in-place edits to autograd views.
        first = scores.new_full((n, samples), -float('inf'))
        first = torch.cat([entry[0, :1, :, 0]-math.log(samples), first[1:]], 0)
        previous = torch.stack([first, torch.full_like(first, -float('inf')), torch.full_like(first, -float('inf'))], -1)
        for t in range(1, tmax):
            complete = torch.logsumexp(previous, (-1, -2))
            incoming = torch.cat([complete.new_full((1,), -float('inf')), complete[:-1]])[:, None]
            start = incoming+entry[t, :, :, 0]-math.log(samples)
            middle = previous[..., 0]+entry[t, :, :, 1]
            sustain = torch.logaddexp(previous[..., 1], previous[..., 2])+scores[t]
            previous = torch.stack([start, middle, sustain], -1)
        return torch.logsumexp(previous[-1].reshape(-1), 0)
    return forward(e, attack)-forward(torch.zeros_like(e), torch.zeros_like(attack))


def occupancies(e):
    """Calibration-only complete-pronunciation soft alignment, no CTC labels."""
    x = e.detach().cpu().numpy()
    tmax, n = x.shape
    a = np.full_like(x, -np.inf); a[0, 0] = x[0, 0]
    for t in range(1, tmax):
        a[t] = a[t-1]
        a[t, 1:] = np.logaddexp(a[t, 1:], a[t-1, :-1])
        a[t] += x[t]
    b = np.full_like(x, -np.inf); b[-1, -1] = 0
    for t in range(tmax-2, -1, -1):
        following = b[t+1]+x[t+1]
        b[t] = following
        b[t, :-1] = np.logaddexp(b[t, :-1], following[1:])
    return np.exp(a+b-a[-1, -1])


class Initializer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(FFT//2-1, 64), torch.nn.Tanh(), torch.nn.Linear(64, DIM))

    def forward(self, power):
        z = power.clamp_min(1e-10).log()[..., 1:-1]
        return self.layers(z-z.mean(-1, keepdim=True))


def train_initializer(physical, device, steps=500):
    """Learn inverse initialization solely from generated model waveforms."""
    net = Initializer().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=.003)
    ones = torch.ones(FFT//2+1, device=device)
    for step in range(steps):
        u = torch.randn(32, DIM, device=device)*1.2
        with torch.no_grad():
            hg, ht = transfer(u) if physical else control_transfer(u)
            power = hg.abs().square()+.08*ht.abs().square()+1e-8*ones
            # Generate a waveform with separate independent phase-marginalized
            # sources, then present its periodogram to the inverse network.
            source = torch.complex(torch.randn_like(power), torch.randn_like(power))*torch.sqrt(power/2)
            wave = torch.fft.irfft(source, n=FFT)
            observed = torch.fft.rfft(wave).abs().square()
        predicted = net(observed)
        loss = (predicted-u).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 100 == 0:
            print(json.dumps({'stage':'synthetic-inverse', 'physical':physical, 'step':step, 'loss':float(loss)}), flush=True)
    with torch.no_grad():
        u = torch.randn(128, DIM, device=device)*1.2
        hg, ht = transfer(u) if physical else control_transfer(u)
        expected = hg.abs().square()+.08*ht.abs().square()+1e-8
        source = torch.complex(torch.randn_like(expected), torch.randn_like(expected))*torch.sqrt(expected/2)
        observed = torch.fft.rfft(torch.fft.irfft(source, n=FFT)).abs().square()
        inferred = net(observed)
        hgi, hti = transfer(inferred) if physical else control_transfer(inferred)
        hg0, ht0 = transfer(torch.zeros_like(u)) if physical else control_transfer(torch.zeros_like(u))
        actual = hgi.abs().square()+.08*hti.abs().square()+1e-8
        baseline = hg0.abs().square()+.08*ht0.abs().square()+1e-8
        validation = {'parameterMse':float((inferred-u).square().mean()),
            'reconstructionLogMse':float((actual.log()-expected.log()).square().mean()),
            'zeroInitializerLogMse':float((baseline.log()-expected.log()).square().mean())}
        print(json.dumps({'stage':'held-synthetic-validation','physical':physical,**validation}), flush=True)
    return net


def training_rows(root, exclusions, reference):
    root = Path(root).resolve()
    labels = [json.loads(l) for l in (root/'labels.jsonl').read_text().splitlines() if l.strip()]
    excluded = set(exclusions)
    excluded.update(l['speaker'] for l in labels if l.get('split') != 'train')
    ayahs = {f"{a['s']}:{a['a']}":a for a in reference['ayahs']}
    rows = []
    for l in labels:
        if l.get('kind') != 'none' or l.get('split') != 'train' or l['speaker'] in excluded:
            continue
        if l.get('events') or l.get('non_errors') or len(l.get('ayahs', [])) != 1:
            raise ValueError('Contradictory none-labelled single-ayah take')
        path = (root/l['wav']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Audio escapes authorized training root')
        a = ayahs[l['ayahs'][0]]
        phones = '_'+a['p'][:a['o'][-1]]+a['iso'][-1]+'_'
        audio = read_audio(path)
        x = spectra(audio)
        # For training takes only, use completed initial audio outside the first
        # transcript phone. Never use an accused window for nuisance estimation.
        before = min(len(audio)//3, 3*RATE)
        shared = nuisance(audio, before)
        if shared is None or len(x) < len(phones):
            continue
        rows.append({'speaker':l['speaker'], 'phones':phones, 'x':x, 'shared':shared})
    if len(set(r['speaker'] for r in rows)) < 3:
        raise ValueError('Too few authorized speakers for grouped training validation')
    return rows, sorted(excluded)


def fit(rows, tokens, inverse, physical, shrinkage, device, rounds=4):
    observations = {p:[] for p in '_'+tokens}
    for r in rows:
        with torch.no_grad():
            init = inverse(r['x'].to(device)).cpu().numpy()
        # Only initialization uses equal-duration placement; all subsequent
        # estimates integrate complete transcript placements.
        ids = np.minimum(np.arange(len(init))*len(r['phones'])//len(init), len(r['phones'])-1)
        for j, p in enumerate(r['phones']):
            observations[p].extend(init[ids == j])
    pool = np.zeros((len(tokens)+1, DIM), np.float32)
    for j, p in enumerate('_'+tokens):
        if observations[p]:
            pool[j] = np.median(observations[p], axis=0)
    target = torch.nn.Parameter(torch.tensor(pool, device=device))
    optimizer = torch.optim.Adam([target], lr=.02)
    lookup = {p:i for i,p in enumerate('_'+tokens)}
    for epoch in range(rounds):
        for r in rows:
            ids = [lookup[p] for p in r['phones']]
            x = r['x'].to(device)
            e = emissions(x, target[ids, None], r['phones'], r['shared'], physical)[..., 0]
            weight = torch.tensor(occupancies(e), device=device)
            loss = -(e*weight).sum()/(len(x)*(FFT//2-1))+shrinkage*(target-torch.tensor(pool, device=device)).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(json.dumps({'stage':'calibration', 'physical':physical, 'epoch':epoch, 'loss':float(loss)}), flush=True)
    # Pronunciation-conditioned ranges from posterior-weighted inferred tracts,
    # shrunk to shared spread; low-rank sigma points share each whole phone run.
    all_points = []; all_weights = [[] for _ in range(len(tokens)+1)]
    for r in rows:
        ids = [lookup[p] for p in r['phones']]
        with torch.no_grad():
            e = emissions(r['x'].to(device), target[ids, None], r['phones'], r['shared'], physical)[..., 0]
            weight = occupancies(e)
            inferred = inverse(r['x'].to(device)).cpu().numpy()
        all_points.extend(inferred)
        for j, p in enumerate(ids):
            all_weights[p].append((inferred, weight[:, j]))
    center = target.detach().cpu().numpy()
    global_cov = np.cov(np.array(all_points).T)+np.eye(DIM)*.02
    directions = []
    for p in range(len(tokens)+1):
        cov = global_cov.copy()*10; mass = 10.
        for inferred, weights in all_weights[p]:
            delta = inferred-center[p]
            cov += (delta*weights[:, None]).T@delta; mass += weights.sum()
        values, vectors = np.linalg.eigh(cov/mass)
        directions.append((vectors[:, -2:]*np.sqrt(np.maximum(values[-2:], .02))).T.tolist())
    return {'center':center.tolist(), 'directions':directions, 'shrinkage':shrinkage}


def realization(model, ids, device):
    center = torch.tensor(model['center'], device=device)[ids]
    directions = torch.tensor(model['directions'], device=device)[ids]
    return torch.stack([center, center+directions[:, 0], center-directions[:, 0],
                        center+directions[:, 1], center-directions[:, 1]], 1)


def compare(model, x, pronunciation, shared, physical, device):
    lookup = {p:i for i,p in enumerate('_'+model['tokens'])}
    if not pronunciation or any(p not in lookup for p in pronunciation):
        return None
    ids = [lookup[p] for p in pronunciation]
    arm = model['physical' if physical else 'control']
    u = realization(arm, ids, device)
    e = emissions(x.to(device), u, pronunciation, shared, physical)
    center = torch.tensor(arm['center'], device=device)[ids]
    before = torch.cat([center[:1], center[:-1]])[:, None, :]
    # Identical smooth entry trajectories in both arms; geometry transforms
    # maintain positive areas and passive delays at every intermediate point.
    attacks = torch.stack([emissions(x.to(device), before+(u-before)*fraction,
        pronunciation, shared, physical) for fraction in (.5, .85)], -1)
    score = sequence_score(e, attacks)
    return float(score.detach()) if torch.isfinite(score) else None


def train(args):
    torch.manual_seed(2026); np.random.seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reference = json.loads(Path(args.reference).read_text())
    tokens = reference['inventory']['tokens']
    if isinstance(tokens, list):
        tokens = ''.join(tokens)
    rows, exclusions = training_rows(args.help_dir, json.loads(Path(args.excluded).read_text()), reference)
    speakers = sorted(set(r['speaker'] for r in rows))
    validation_speakers = set(speakers[::4])
    training = [r for r in rows if r['speaker'] not in validation_speakers]
    validation = [r for r in rows if r['speaker'] in validation_speakers]
    result = {'version':1, 'tokens':tokens, 'inventory':reference['inventory']['id'], 'rate':RATE,
              'fft':FFT, 'hop':HOP, 'exclusions':exclusions, 'trainingSpeakers':speakers,
              'validationSpeakers':sorted(validation_speakers), 'recordings':len(rows)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for physical in (True, False):
        name = 'physical' if physical else 'control'
        inverse = train_initializer(physical, device, args.synthetic_steps)
        torch.save(inverse.state_dict(), out/f'{name}-inverse.pt')
        choices = []
        for shrinkage in (.2, 1.):
            arm = fit(training, tokens, inverse, physical, shrinkage, device, args.rounds)
            trial = dict(result, **{name:arm})
            with torch.no_grad():
                scores = [compare(trial, r['x'], r['phones'], r['shared'], physical, device)/len(r['x']) for r in validation]
            score = float(np.mean(scores))
            choices.append((score, shrinkage))
            print(json.dumps({'stage':'speaker-validation', 'arm':name, 'shrinkage':shrinkage, 'score':score}), flush=True)
        _, selected = max(choices)
        result[name] = fit(rows, tokens, inverse, physical, selected, device, args.rounds)
        result[name]['selection'] = choices
        (out/'waveguide.json').write_text(json.dumps(result))
    result['assetHashes'] = {name:hashlib.sha256((out/name).read_bytes()).hexdigest()
                            for name in ('physical-inverse.pt', 'control-inverse.pt')}
    (out/'waveguide.json').write_text(json.dumps(result))
    print(json.dumps({'finished':str(out), 'recordings':len(rows), 'speakers':len(speakers)}), flush=True)


def probe(args):
    model = json.loads(Path(args.model).read_text())
    audio = read_audio(args.audio)
    candidates = json.loads(Path(args.candidates).read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    with torch.no_grad():
        for candidate in candidates:
            start = int(candidate['fromFrame']*candidate['frameMs']*RATE/1000)
            end = int(candidate['toFrame']*candidate['frameMs']*RATE/1000)
            shared = nuisance(audio, start)
            scores = {}
            if shared is not None and 0 <= start < end <= len(audio):
                x = spectra(audio[start:end])
                for physical in (True, False):
                    name = 'physical' if physical else 'control'
                    scores[name] = {hypothesis:compare(model, x, candidate[hypothesis], shared, physical, device)
                                    for hypothesis in ('written', 'rival')}
            physical = scores.get('physical', {})
            veto = all(physical.get(h) is not None for h in ('written', 'rival')) and physical['written'] > physical['rival']
            control = scores.get('control', {})
            reversal = veto and all(control.get(h) is not None for h in ('written', 'rival')) and control['written'] < control['rival']
            results.append({'candidate':candidate, 'scores':scores, 'veto':veto, 'controlReversal':reversal,
                            'nuisance':None if shared is None else {k:shared[k] for k in ('beforeSample','outsideSamples')}})
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps({'comparisons':len(results), 'vetoes':sum(r['veto'] for r in results),
                      'controlReversals':sum(r['controlReversal'] for r in results)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    for flag in ('help-dir', 'excluded', 'reference', 'out'):
        t.add_argument('--'+flag, required=True)
    t.add_argument('--synthetic-steps', type=int, default=500)
    t.add_argument('--rounds', type=int, default=4)
    p = sub.add_parser('probe')
    for flag in ('model', 'audio', 'candidates', 'out'):
        p.add_argument('--'+flag, required=True)
    args = parser.parse_args()
    train(args) if args.command == 'train' else probe(args)

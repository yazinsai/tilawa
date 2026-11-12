"""Cross recorded speech identity with independently recorded background identity.

Run prepare, fit, export through the supervisor. Only prepare reads help-train;
fit and export require its immutable cache. No practice input selects parameters.
"""
import argparse
import copy
from collections import Counter
import json
import math
import os
from pathlib import Path

from student_noise_features import (ROOT, clean_target, digest, export,
    feature_runtime, stable, stream_features, write_json)

RECIPE = dict(seed=173, max_seconds=40., pause_seconds=.24, pause_ratio=.01,
    fft=512, hop=128, epochs=24, lr=.0003, anchor=4., temperature=.1,
    contrast_weights=[.1, .3], batch_pairs=4, max_occurrences=128)
EXPERIMENT = 'listener-local-noise-contrast-training'


def eligible(label):
    return (label.get('kind') == 'none' and label.get('scored') is not False
            and not any(label.get(k) for k in ('events', 'non_errors', 'unsupported')))


def partitions(labels):
    """Exclude whole release-held-out speakers, then reserve training speakers."""
    banned = {str(r['speaker']) for r in labels if r.get('split') != 'train'}
    speakers = sorted({str(r['speaker']) for r in labels
                       if r.get('split') == 'train' and str(r['speaker']) not in banned}, key=stable)
    if len(speakers) < 4:
        raise ValueError('At least four authorized training speakers required')
    held = set(speakers[:max(2, len(speakers)//5)])
    return {s: 'validation' if s in held else 'train' for s in speakers}, sorted(banned)


def recorded_background(wave, sr=16000):
    """Require a contiguous quiet recorded interval, not scattered blank frames.

    A raw-energy criterion rejects speech and digital silence. This cannot prove
    absence of a quiet breath; provenance and measured levels are retained.
    """
    import numpy as np
    block = sr//100
    n = len(wave)//block
    if n < 2:
        return None
    energy = np.mean(wave[:n*block].reshape(n, block).astype('float64')**2, axis=1)
    speech = float(np.quantile(energy, .8))
    quiet = (energy <= speech*RECIPE['pause_ratio']) & (energy > 1e-12)
    length = math.ceil(RECIPE['pause_seconds']*100)
    starts = [i for i in range(n-length+1) if quiet[i:i+length].all()]
    if not starts:
        return None
    at = min(starts, key=lambda i: float(energy[i:i+length].mean()))
    noise = wave[at*block:(at+length)*block].copy()
    return noise, dict(start_sample=at*block, samples=len(noise),
                       energy=float(np.mean(noise.astype('float64')**2)), speech_energy=speech)


def crossed_view(wave, own_noise, donor_noise):
    """One shared speech estimate; exchange the estimated recorded background.

    Wiener decomposition x=s+n gives an exactly unchanged-background control
    x. The swapped view s+n' uses recorded donor samples at n's original energy.
    Speech is identical between the two decomposed views, but s is estimated:
    imperfect separation can remove quiet speech and is an explicit limitation.
    """
    import numpy as np
    from scipy.signal import stft, istft
    fft, hop = RECIPE['fft'], RECIPE['hop']
    kwargs = dict(fs=16000, nperseg=fft, noverlap=fft-hop, boundary='zeros')
    _, _, observed = stft(wave, **kwargs)
    _, _, quiet = stft(own_noise, **kwargs)
    noise_power = np.mean(np.abs(quiet)**2, axis=1, keepdims=True)
    mask = np.minimum(noise_power / np.maximum(np.abs(observed)**2, 1e-20), 1.)
    _, estimated = istft(observed*mask, fs=16000, nperseg=fft, noverlap=fft-hop)
    estimated = estimated[:len(wave)].astype('float32')
    speech = wave-estimated
    # Tile a real contiguous recording at the same phase for both paired speeches.
    donor = np.resize(donor_noise, len(wave)).astype('float32')
    energy = float(np.mean(estimated.astype('float64')**2))
    donor_energy = float(np.mean(donor.astype('float64')**2))
    if donor_energy <= 0 or energy <= 0:
        raise ValueError('Nonzero recorded background is required')
    donor *= np.sqrt(energy/donor_energy)
    swapped = speech+donor
    if not np.isfinite(swapped).all():
        raise ValueError('Nonfinite crossed waveform')
    return swapped, speech, estimated, donor


def pairing(records):
    """Two distinct speeches share one donor; donors never cross partitions."""
    result = []
    for partition in ('train', 'validation'):
        rows = sorted([r for r in records if r['partition'] == partition], key=lambda r: stable(r['key']))
        if len({r['speaker'] for r in rows}) < 2:
            raise ValueError('Both partitions need independent background speakers')
        for i in range(0, len(rows)-1, 2):
            a, b = rows[i:i+2]
            donors = [r for r in rows if r['speaker'] not in (a['speaker'], b['speaker'])]
            if not donors:
                # Same-partition, other-recording donors still cross identities;
                # require a speaker different from the first speech.
                donors = [r for r in rows if r['speaker'] != a['speaker'] and r['key'] != b['key']]
            if not donors:
                continue
            donor = min(donors, key=lambda r: stable(f"{a['key']}:{b['key']}:{r['key']}"))
            result.append(dict(speech=[a['key'], b['key']], donor=donor['key'], partition=partition))
    if not all(any(p['partition'] == part for p in result) for part in ('train', 'validation')):
        raise ValueError('No crossed speech/background pairs in a required partition')
    return result


def occurrence_occupancy(log_probs, target):
    """CTC forward-backward for every transcript occurrence, including repeats.

    Occupancies are detached frozen alignment distributions, not checker visits
    or argmax boundaries. Each occurrence integrates over all valid placements.
    """
    import numpy as np
    from scipy.special import logsumexp
    target = np.asarray(target, dtype='int64')
    if (np.ndim(log_probs) != 2 or not len(log_probs) or not len(target)
            or np.any(target <= 0) or np.any(target >= np.shape(log_probs)[1])):
        raise ValueError('A nonempty valid phone transcript and frame matrix are required')
    states = np.zeros(2*len(target)+1, dtype='int64')
    states[1::2] = target
    t, s = len(log_probs), len(states)
    emission = np.asarray(log_probs, dtype='float64')[:, states]
    alpha = np.full((t, s), -np.inf)
    alpha[0, :min(2, s)] = emission[0, :min(2, s)]
    skip = np.arange(s) >= 2
    skip[2:] &= (states[2:] != 0) & (states[2:] != states[:-2])
    for i in range(1, t):
        previous = alpha[i-1]
        alpha[i] = emission[i]+logsumexp(np.stack([previous,
            np.r_[-np.inf, previous[:-1]],
            np.where(skip, np.r_[-np.inf, -np.inf, previous[:-2]], -np.inf)]), axis=0)
    z = logsumexp(alpha[-1, -2:])
    if not np.isfinite(z):
        raise ValueError('Transcript has no complete CTC alignment')
    beta = np.full((t, s), -np.inf)
    beta[-1, -2:] = 0.
    for i in range(t-2, -1, -1):
        following = beta[i+1]+emission[i+1]
        beta[i] = logsumexp(np.stack([following,
            np.r_[following[1:], -np.inf],
            np.r_[np.where(skip[2:], following[2:], -np.inf), -np.inf, -np.inf]]), axis=0)
    occupancy = np.exp((alpha+beta-z)[:, 1::2]).T
    return (occupancy/np.maximum(occupancy.sum(1, keepdims=True), 1e-20)).astype('float32')


def prepare(model, out):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import torchaudio
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise ValueError('Pairing is already frozen; use a new output directory')
    root = Path(os.environ['LAB_DATA'])/'help-train'
    labels = [json.loads(s) for s in (root/'labels.jsonl').read_text().splitlines() if s.strip()]
    mapping, banned = partitions(labels)
    ref = json.loads((ROOT/'packages/core/assets/ref-v2.json').read_text())
    ayahs = {f"{a['s']}:{a['a']}": a['p'] for a in ref['ayahs']}
    manifest, head, runtime = feature_runtime(model, out)
    np.savez(out/'initial-head.npz', weight=head['weight'].T, bias=head['bias'])
    records, waves, noises = [], {}, {}
    audit = Counter()
    for label in labels:
        speaker = str(label['speaker'])
        if speaker not in mapping or label.get('exclude') or label.get('excluded'):
            audit['excluded_recordings'] += 1
            continue
        path = (root/label['wav']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Missing or unauthorized training waveform')
        if not 0 < sf.info(path).duration <= RECIPE['max_seconds']:
            audit['duration_rejected'] += 1
            continue
        wave, sr = sf.read(path, dtype='float32')
        if wave.ndim == 2:
            wave = wave.mean(1)
        if sr != 16000:
            divisor = math.gcd(sr, 16000)
            wave = resample_poly(wave, 16000//divisor, sr//divisor).astype('float32')
        found = recorded_background(wave)
        if found is None:
            audit['no_recorded_background'] += 1
            continue
        noise, interval = found
        key = label['id']
        if key in waves:
            raise ValueError('Duplicate training recording identity')
        target = clean_target(''.join(ayahs[a] for a in label['ayahs']), manifest['tokens']) if eligible(label) else []
        records.append(dict(key=key, speaker=speaker, partition=mapping[speaker],
                            target=target, noise=interval, audio_sha256=digest(path)))
        waves[key], noises[key] = wave, noise
    supervised = [r for r in records if r['target']]
    audit.update(recordings=len(records), supervised=len(supervised))
    print(json.dumps(dict(background_audit=dict(audit),
        supervised_speakers={part: len({r['speaker'] for r in supervised if r['partition'] == part})
                             for part in ('train', 'validation')})), flush=True)
    pairs = pairing(supervised)
    cache = out/'features'
    cache.mkdir(exist_ok=True)
    lookup = {r['key']: r for r in records}
    views = []
    torch.set_num_threads(2)
    def capture(key, wave, view, donor=None):
        feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(wave).unsqueeze(0), num_mel_bins=80,
            sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400).numpy()
        features, baseline = stream_features(runtime, manifest, feats, head['feature'])
        target = lookup[key]['target']
        occupancy = occurrence_occupancy(baseline, target) if target else np.empty((0, len(baseline)), dtype='float32')
        path = cache/(stable(f'{key}:{view}:{donor}')+'.npz')
        np.savez_compressed(path, features=features, baseline=baseline, target=np.asarray(target, dtype='int64'), occupancy=occupancy)
        entry = dict(file=str(path), sha256=digest(path), recording=key, speaker=lookup[key]['speaker'],
                     partition=lookup[key]['partition'], view=view, donor=donor, supervised=bool(target))
        views.append(entry)
        return entry
    controls = {r['key']: capture(r['key'], waves[r['key']], 'control') for r in records}
    for pair in pairs:
        pair['views'] = []
        for key in pair['speech']:
            swapped, _, estimated, donor = crossed_view(waves[key], noises[key], noises[pair['donor']])
            pair['views'].append([controls[key]['file'], capture(key, swapped, 'swap', pair['donor'])['file']])
            pair.setdefault('energy', []).append(dict(original=float(np.mean(estimated.astype('float64')**2)),
                                                       substituted=float(np.mean(donor.astype('float64')**2))))
        print(json.dumps(dict(cached=len(views), pairs=len(pair['views']))), flush=True)
    write_json(out/'plan.json', dict(schema=1, id=EXPERIMENT, recipe=RECIPE, records=records,
        views=views, pairs=pairs, excluded_speakers=banned, labels_sha256=digest(root/'labels.jsonl'),
        background_audit=dict(audit),
        primary_sha256=manifest['sha256'], initial_head_sha256=digest(out/'initial-head.npz'),
        feature_graph_sha256=digest(out/'encoder-features.onnx')))
    print(json.dumps(dict(recordings=len(records), supervised=len(supervised), crossed_pairs=len(pairs),
                         speakers=len(mapping), excluded=len(banned))), flush=True)


def read_plan(out):
    plan = json.loads((out/'plan.json').read_text())
    if plan['id'] != EXPERIMENT or plan['recipe'] != RECIPE:
        raise ValueError('Frozen training recipe changed')
    for file, key in [('initial-head.npz', 'initial_head_sha256'), ('encoder-features.onnx', 'feature_graph_sha256')]:
        if digest(out/file) != plan[key]:
            raise ValueError('Frozen feature or head parameters changed')
    mapping, files = {}, {}
    for view in plan['views']:
        if digest(view['file']) != view['sha256']:
            raise ValueError('Frozen cached view changed')
        if mapping.setdefault(view['speaker'], view['partition']) != view['partition'] or view['speaker'] in plan['excluded_speakers']:
            raise ValueError('Speaker partition leakage')
        if view['file'] in files:
            raise ValueError('Duplicate cached view')
        files[view['file']] = view
    records = {r['key']: r for r in plan['records']}
    for pair in plan['pairs']:
        if len(pair['speech']) != 2 or len(set(pair['speech']+[pair['donor']])) != 3 or len(pair['views']) != 2:
            raise ValueError('Distinct speech and background recordings are required')
        if any(records[k]['partition'] != pair['partition'] for k in pair['speech']+[pair['donor']]):
            raise ValueError('Background donor crosses partition')
        for key, (control, swapped) in zip(pair['speech'], pair['views']):
            original, crossed = files[control], files[swapped]
            if (original['recording'] != key or crossed['recording'] != key
                    or original['view'] != 'control' or original['donor'] is not None
                    or crossed['view'] != 'swap' or crossed['donor'] != pair['donor']
                    or original['partition'] != pair['partition'] or crossed['partition'] != pair['partition']
                    or not original['supervised'] or not crossed['supervised']):
                raise ValueError('Crossed speech/background identity changed')
    return plan


def ordinary_loss(head, data):
    import torch
    import torch.nn.functional as F
    lp = head(data['features']).log_softmax(-1)
    loss = RECIPE['anchor']*F.kl_div(lp, data['baseline'].exp(), reduction='sum')/len(lp)
    if len(data['target']):
        loss = loss+F.ctc_loss(lp[:, None], data['target'], torch.tensor([len(lp)]),
            torch.tensor([len(data['target'])]), reduction='sum')/len(lp)
    return loss


def crossed_contrast(head, pair):
    """Same occurrence agrees across backgrounds; shared-noise consonants differ."""
    import torch
    import torch.nn.functional as F
    pooled, labels = [], []
    for control, swapped in pair:
        # Marginal occupancies are specific to each acoustic view; identity and
        # transcript occurrence correspondence are fixed, boundaries are not.
        target = control['target']
        if not torch.equal(target, swapped['target']):
            raise ValueError('Paired speech transcripts differ')
        keep = torch.where((target > 0) & (target <= 28))[0][:RECIPE['max_occurrences']]
        labels.append(target[keep])
        pooled.append([F.normalize(d['occupancy'][keep] @ head(d['features']).softmax(-1), dim=-1)
                       for d in (control, swapped)])
    losses = []
    for i in range(2):
        original, swapped = pooled[i]
        other = pooled[1-i][1]
        if not len(original) or not len(other):
            continue
        # Explicit agreement remains active even where no different consonant
        # exists. InfoNCE rules out a collapsed all-phone representation.
        agreement = (1-(original*swapped).sum(-1)).mean()
        positive = (swapped*original).sum(-1, keepdim=True)
        negative = swapped @ other.T
        mask = labels[i][:, None] != labels[1-i][None, :]
        logits = torch.cat([positive, negative.masked_fill(~mask, -torch.inf)], dim=1)/RECIPE['temperature']
        losses.append(agreement+F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long)))
    return sum(losses)/len(losses) if losses else head.weight.sum()*0


def fit(out):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(RECIPE['seed'])
    plan = read_plan(out)
    data = {}
    for v in plan['views']:
        with np.load(v['file']) as a:
            data[v['file']] = {k: torch.from_numpy(a[k].copy()) for k in a.files}
    initial = np.load(out/'initial-head.npz')
    base = torch.nn.Linear(initial['weight'].shape[1], initial['weight'].shape[0])
    with torch.no_grad():
        base.weight.copy_(torch.from_numpy(initial['weight']))
        base.bias.copy_(torch.from_numpy(initial['bias']))
    weights = {'ctc': 0., **{f'contrast-{w}': w for w in RECIPE['contrast_weights']}}
    heads = {name: copy.deepcopy(base) for name in weights}
    optimizers = {name: torch.optim.Adam(h.parameters(), lr=RECIPE['lr']) for name, h in heads.items()}
    train = [p for p in plan['pairs'] if p['partition'] == 'train']
    protected = [v for v in plan['views'] if v['partition'] == 'train' and not v['supervised']]
    validation = [v for v in plan['views'] if v['partition'] == 'validation']
    if not protected or not validation:
        raise ValueError('Preservation and reserved-speaker validation are required')
    best, history = {}, []
    def evaluate(head):
        by_speaker = {}
        with torch.no_grad():
            for view in validation:
                value = float(ordinary_loss(head, data[view['file']]))
                by_speaker.setdefault(view['speaker'], []).append(value)
        return float(np.mean([np.mean(v) for v in by_speaker.values()]))
    for epoch in range(RECIPE['epochs']):
        order = np.random.default_rng(RECIPE['seed']+epoch).permutation(len(train))
        for start in range(0, len(order), RECIPE['batch_pairs']):
            pairs = [train[i] for i in order[start:start+RECIPE['batch_pairs']]]
            views = [data[f] for p in pairs for pair in p['views'] for f in pair]
            bank = data[protected[(epoch*len(train)+start) % len(protected)]['file']]
            for name, head in heads.items():
                optimizers[name].zero_grad(set_to_none=True)
                loss = sum(ordinary_loss(head, d) for d in views)/len(views)+ordinary_loss(head, bank)
                if weights[name]:
                    loss += weights[name]*sum(crossed_contrast(head, [[data[f] for f in pair] for pair in p['views']])
                                              for p in pairs)/len(pairs)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                optimizers[name].step()
        scores = {name: evaluate(h) for name, h in heads.items()}
        for name, score in scores.items():
            if not math.isfinite(score):
                raise ValueError('Nonfinite validation objective')
            if name not in best or score < best[name]['loss']:
                path = out/(name+'.npz')
                np.savez(path, weight=heads[name].weight.detach().numpy(), bias=heads[name].bias.detach().numpy())
                best[name] = dict(loss=score, epoch=epoch+1, path=str(path), sha256=digest(path))
        history.append(dict(epoch=epoch+1, validation=scores))
        print(json.dumps(history[-1]), flush=True)
    selected = min([n for n in weights if weights[n]], key=lambda n: (best[n]['loss'], weights[n]))
    write_json(out/'selection.json', dict(schema=1, plan_sha256=digest(out/'plan.json'),
        selected=selected, arms={'ctc': best['ctc'], 'crossed': best[selected]}, candidates=best, history=history,
        identical_views=True, identical_updates=True, selection='speaker-mean CTC and preservation on reserved training speakers'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'fit', 'export'))
    parser.add_argument('--model', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command != 'fit' and args.model is None:
        parser.error('prepare/export require --model')
    if args.command == 'prepare':
        prepare(args.model, args.out)
    elif args.command == 'fit':
        fit(args.out)
    else:
        export(args.model, args.out, read_plan, RECIPE['seed'])

"""Recording-level transcript contamination refit of a frozen streaming phone head.

prepare, fit and export operate only on authorized training audio. The fixed
normalized transcript background is independent of the recording. Calibration
speakers estimate the mixture weight; separate validation speakers select a
common checkpoint for the mixture and matched ordinary-CTC refits.
"""
import argparse
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path

from student_head_refit import (ROOT, authorized_records, clean_target, digest,
    export, feature_runtime, has_clean_transcript, stable, stream_features,
    training_helpers, write_json)

RECIPE = dict(seed=831, original_limit=512, views=3, epochs=12, batch=8,
    preservation_batch=4, lr=.0001, anchor=2., max_seconds=30.)


def speaker_partitions(records):
    """Fit/calibration/selection are separate whole-speaker training subsets."""
    mapping = {}
    for source in ('original', 'help'):
        speakers = sorted({r['speaker'] for r in records if r['source'] == source}, key=stable)
        if len(speakers) < 3:
            raise ValueError('At least three training speakers required per source')
        reserved = max(1, len(speakers)//5)
        for i, speaker in enumerate(speakers):
            mapping[speaker] = ('validation' if i < reserved else
                                'calibration' if i < 2*reserved else 'train')
    return mapping


def transcript_background(targets, vocab):
    """Proper finite-string Markov distribution with explicit EOS (class zero).

    Add-one smoothing gives every nonblank identity and EOS positive mass.
    Every row sums to one; the positive minimum EOS probability guarantees
    eventual termination, so total probability over all finite strings is one.
    Counts use each fit recording once, never its augmentation copies.
    """
    import numpy as np

    if vocab < 2:
        raise ValueError('A nonblank inventory is required')
    counts = np.ones((vocab, vocab), dtype='float64')
    for target in targets:
        previous = 0
        for phone in target:
            if not 0 < phone < vocab:
                raise ValueError('Transcript contains blank or unknown identity')
            counts[previous, phone] += 1
            previous = phone
        counts[previous, 0] += 1
    return counts/counts.sum(1, keepdims=True)


def transcript_log_probability(background, target):
    previous, value = 0, 0.
    for phone in target:
        if not 0 < phone < len(background):
            raise ValueError('Invalid transcript identity')
        value += math.log(background[previous, phone])
        previous = phone
    return value+math.log(background[previous, 0])


def acoustic_responsibility(log_acoustic, log_background, contamination):
    """Probability the complete supplied transcript has the acoustic source."""
    import numpy as np

    if not 0 <= contamination <= 1:
        raise ValueError('Invalid contamination probability')
    a, q = np.asarray(log_acoustic), np.asarray(log_background)
    if np.isnan(a).any() or not np.isfinite(q).all():
        raise ValueError('Invalid recording likelihood')
    if contamination == 0:
        return np.ones_like(a, dtype='float64')
    if contamination == 1:
        return np.zeros_like(a, dtype='float64')
    clean = math.log1p(-contamination)+a
    bad = math.log(contamination)+q
    return np.exp(clean-np.logaddexp(clean, bad))


def estimate_contamination(log_acoustic, log_background):
    """MAP EM on independent calibration recordings, with Beta(2,2) prior.

    One pseudorecording for each source prevents a degenerate all-background
    solution. The head gets no calibration gradient and no replacement labels.
    """
    import numpy as np

    if not len(log_acoustic) or len(log_acoustic) != len(log_background):
        raise ValueError('Missing matched calibration likelihoods')
    pi = .5
    for _ in range(1000):
        r = acoustic_responsibility(log_acoustic, log_background, pi)
        updated = float((1+(1-r).sum())/(2+len(r)))
        if abs(updated-pi) < 1e-10:
            return updated
        pi = updated
    raise ValueError('Contamination estimation did not converge')


def recording_nll(parameters, data):
    import torch
    import torch.nn.functional as F

    lp = F.linear(data['features'], *parameters).log_softmax(-1)
    return F.ctc_loss(lp[:, None, :], data['target'], torch.tensor([len(lp)]),
        torch.tensor([len(data['target'])]), reduction='sum', zero_infinity=False)


def mixture_nll(acoustic_nll, log_background, contamination):
    """Mix full normalized sequence probabilities before length reduction."""
    import torch

    if not 0 <= contamination < 1 or not math.isfinite(log_background):
        raise ValueError('Invalid fixed transcript mixture')
    if contamination == 0:
        return acoustic_nll
    a = math.log1p(-contamination)-acoustic_nll
    b = torch.as_tensor(math.log(contamination)+log_background,
                        dtype=acoustic_nll.dtype, device=acoustic_nll.device)
    return -torch.logaddexp(a, b)

def help_preservation(max_seconds):
    """Scripts identify controls, never supply a performed phone transcript."""
    import soundfile as sf

    root = Path(os.environ['LAB_DATA'])/'help-train'
    labels = [json.loads(s) for s in (root/'labels.jsonl').read_text().splitlines() if s.strip()]
    banned = {str(l['speaker']) for l in labels if l.get('split') != 'train' or
              l.get('exclude') or l.get('excluded')}
    rows = []
    for l in labels:
        if l.get('split') != 'train' or str(l['speaker']) in banned or has_clean_transcript(l):
            continue
        path = (root/l['wav']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Preservation audio is missing or escapes help-train')
        if 0 < sf.info(path).duration <= max_seconds:
            rows.append(dict(key='help/'+l['id'], source='preservation', file=str(path),
                             speaker='help/'+str(l['speaker']), target=[]))
    if not rows:
        raise ValueError('No mistake/repeat recordings available for output preservation')
    return rows

def prepare(model, out):
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import torchaudio
    from student import TOKENS, mix_noise

    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise ValueError('Training plan already frozen; use a new output directory')
    records, excluded, labels_sha = authorized_records(RECIPE['original_limit'], RECIPE['max_seconds'])
    preservation = help_preservation(RECIPE['max_seconds'])
    partitions = speaker_partitions(records)
    for r in preservation:
        # A speaker without a clean take contributes transcript-free fit
        # preservation only; no claimed independent calibration/validation.
        r['partition'] = partitions.get(r['speaker'], 'train')
    for r in records:
        r['partition'] = partitions[r['speaker']]
    records += preservation
    manifest, head, runtime = feature_runtime(model, out, RECIPE['seed'])
    if (manifest['tokens'] != TOKENS or manifest['blank'] != 0 or manifest['frameMs'] != 40 or
            manifest['sampleRate'] != 16000 or manifest['fbank']['mels'] != 80):
        raise ValueError('Training does not match shipped inventory, clock or frontend')
    np.savez(out/'initial-head.npz', weight=head['weight'].T, bias=head['bias'])
    ref = json.loads((ROOT/'packages/core/assets/ref-v2.json').read_text())
    ayahs = {f"{a['s']}:{a['a']}": a['p'] for a in ref['ayahs']}
    waves = {}
    for path in sorted({r['file'] for r in records if r['source'] == 'original'}):
        selected = [r for r in records if r['file'] == path]
        table = pq.read_table(path, columns=['flac', 'ref']).take([r['index'] for r in selected]).to_pylist()
        for r, value in zip(selected, table):
            wave, sr = sf.read(io.BytesIO(value['flac']), dtype='float32')
            if sr != 16000 or wave.ndim != 1:
                raise ValueError('Original prepared audio is not mono 16 kHz')
            waves[r['key']] = wave
            # Retain the supplied reference, including its edges. Contamination
            # changes its acoustic responsibility, never its phone identities.
            r['target'] = clean_target(value['ref'], manifest['tokens'])
    for r in records:
        if r['source'] == 'original':
            continue
        wave, sr = sf.read(r['file'], dtype='float32')
        if wave.ndim == 2:
            wave = wave.mean(1)
        if sr != 16000:
            divisor = math.gcd(sr, 16000)
            wave = resample_poly(wave, 16000//divisor, sr//divisor).astype('float32')
        waves[r['key']] = wave
        if r['source'] == 'help':
            r['target'] = clean_target(''.join(ayahs[a] for a in r['ayahs']), manifest['tokens'])
    if any(not r['target'] for r in records if r['source'] != 'preservation'):
        raise ValueError('Transcribed recording has no authorized phone target')
    # Exact waveform copies across speaker partitions would invalidate
    # independence. Audit content before producing any feature cache.
    seen = {}
    for r in records:
        sha = hashlib.sha256(np.ascontiguousarray(waves[r['key']]).tobytes()).hexdigest()
        if sha in seen and seen[sha] != r['speaker']:
            raise ValueError('Identical performance appears under different speaker identities')
        seen[sha] = r['speaker']
        r['waveform_sha256'] = sha
    helpers = training_helpers()
    aug = {k: np.load(Path('/student/aug')/(k+'.npz')) for k in ('noise', 'rir')}
    cache = out/'features'
    cache.mkdir(exist_ok=True)
    views = []
    torch.set_num_threads(2)
    for r in records:
        for view in range(RECIPE['views']):
            rng = np.random.default_rng(int(stable(f"{r['key']}:{view}:{RECIPE['seed']}")[:16], 16))
            wave = waves[r['key']].copy()
            if view:
                if rng.random() < .2:
                    wave = helpers['reverb'](wave, helpers['pick'](aug['rir'], rng))
                if rng.random() < .5:
                    wave = mix_noise(wave, helpers['pick'](aug['noise'], rng), rng.uniform(5., 25.), rng)
                wave *= 10**(rng.uniform(-10, 6)/20)
            feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(wave).unsqueeze(0), num_mel_bins=80,
                sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400)
            if view:
                feats = helpers['spec_augment'](feats, rng)
            x, lp = stream_features(runtime, manifest, feats.numpy(), head['feature'])
            logits = x@head['weight']+head['bias']
            peak = logits.max(1, keepdims=True)
            reconstructed = logits-peak-np.log(np.exp(logits-peak).sum(1, keepdims=True))
            if not np.allclose(reconstructed, lp, atol=.002, rtol=0):
                raise ValueError('Cached features do not reproduce shipped phone-head probabilities')
            file = cache/(stable(f"{r['key']}:{view}")+'.npz')
            np.savez_compressed(file, features=x, baseline=lp, target=np.asarray(r['target'], dtype=np.int64))
            views.append(dict(recording=r['key'], view=view, speaker=r['speaker'], partition=r['partition'],
                              supervised=r['source'] != 'preservation', file=str(file), sha256=digest(file)))
        print(json.dumps(dict(cached=r['key'], partition=r['partition'], supervised=r['source'] != 'preservation')), flush=True)
    background = transcript_background([r['target'] for r in records
        if r['source'] != 'preservation' and r['partition'] == 'train'], len(manifest['tokens'])+1)
    write_json(out/'plan.json', dict(schema=1, recipe=RECIPE, records=records, views=views,
        background=background.tolist(),
        excluded_speakers=excluded, help_labels_sha256=labels_sha, primary_sha256=manifest['sha256'],
        feature_graph_sha256=digest(out/'encoder-features.onnx'), initial_head_sha256=digest(out/'initial-head.npz')))

def preservation_loss(parameters, data):
    import torch.nn.functional as F

    lp = F.linear(data['features'], *parameters).log_softmax(-1)
    return F.kl_div(lp, data['baseline'].exp(), reduction='sum')/len(lp)


def read_plan(out):
    import numpy as np

    plan = json.loads((out/'plan.json').read_text())
    if plan['recipe'] != RECIPE:
        raise ValueError('Frozen training settings changed')
    for file, sha in [('initial-head.npz', plan['initial_head_sha256']),
                      ('encoder-features.onnx', plan['feature_graph_sha256'])]:
        if digest(out/file) != sha:
            raise ValueError('Frozen head or encoder graph changed')
    background = np.asarray(plan['background'])
    if (background.ndim != 2 or background.shape[0] != background.shape[1] or
            not np.isfinite(background).all() or not (background > 0).all() or
            not np.allclose(background.sum(1), 1., atol=1e-12, rtol=0)):
        raise ValueError('Transcript background is not normalized')
    groups = {}
    for view in plan['views']:
        if digest(view['file']) != view['sha256']:
            raise ValueError('Frozen feature cache changed')
        if view['partition'] not in ('train', 'calibration', 'validation'):
            raise ValueError('Unknown training partition')
        previous = groups.setdefault(view['speaker'], view['partition'])
        if previous != view['partition'] or view['speaker'] in plan['excluded_speakers']:
            raise ValueError('Whole-speaker separation violated')
    return plan


def fit(out):
    import copy
    import numpy as np
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(RECIPE['seed'])
    plan = read_plan(out)
    initial = np.load(out/'initial-head.npz')
    head = torch.nn.Linear(initial['weight'].shape[1], initial['weight'].shape[0])
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(initial['weight']))
        head.bias.copy_(torch.from_numpy(initial['bias']))
    arms = {'ordinary': head, 'mixture': copy.deepcopy(head)}
    train = [v for v in plan['views'] if v['partition'] == 'train' and v['supervised']]
    bank = [v for v in plan['views'] if v['partition'] == 'train' and not v['supervised']]
    calibration = [v for v in plan['views'] if v['partition'] == 'calibration'
                   and v['supervised'] and v['view'] == 0]
    validation = [v for v in plan['views'] if v['partition'] == 'validation']
    if (not train or not bank or not calibration or not validation or
            not any(v['supervised'] for v in validation)):
        raise ValueError('Required training/calibration/selection inputs unavailable')

    @lru_cache(maxsize=32)
    def cached(path):
        with np.load(path) as a:
            return {k: torch.from_numpy(a[k].copy()) for k in ('features', 'baseline', 'target')}

    background = np.asarray(plan['background'])
    q = {r['key']: transcript_log_probability(background, r['target'])
         for r in plan['records'] if r['source'] != 'preservation'}

    def calibrate(h):
        with torch.no_grad():
            a = [-float(recording_nll(tuple(h.parameters()), cached(v['file']))) for v in calibration]
        return estimate_contamination(a, [q[v['recording']] for v in calibration])

    def evaluate(h):
        by_speaker = {}
        with torch.no_grad():
            for v in validation:
                d = cached(v['file'])
                loss = RECIPE['anchor']*preservation_loss(tuple(h.parameters()), d)
                if v['supervised']:
                    loss = loss+recording_nll(tuple(h.parameters()), d)/len(d['features'])
                by_speaker.setdefault(v['speaker'], []).append(float(loss))
        result = float(np.mean([np.mean(values) for values in by_speaker.values()]))
        if not math.isfinite(result):
            raise ValueError('Invalid reserved-speaker validation loss')
        return result

    baseline = evaluate(head)
    history, updates = [], 0
    for epoch in range(RECIPE['epochs']):
        # EM uses only ordinary, unmasked calibration recordings, so artificial
        # SpecAugment occlusion cannot determine the recording contamination rate.
        pi = calibrate(arms['mixture'])
        order = np.random.default_rng(RECIPE['seed']+epoch).permutation(len(train))
        responsibility = []
        for start in range(0, len(order), RECIPE['batch']):
            indices = order[start:start+RECIPE['batch']]
            support = [cached(train[i]['file']) for i in indices]
            protected = [cached(bank[(updates*RECIPE['preservation_batch']+j) % len(bank)]['file'])
                         for j in range(RECIPE['preservation_batch'])]
            for name, h in arms.items():
                h.zero_grad(set_to_none=True)
                params = tuple(h.parameters())
                terms = []
                for i, data in zip(indices, support):
                    nll = recording_nll(params, data)
                    if not torch.isfinite(nll):
                        raise ValueError('Impossible or invalid complete acoustic transcript')
                    if name == 'mixture':
                        log_q = q[train[i]['recording']]
                        responsibility.append(dict(recording=train[i]['recording'], view=train[i]['view'],
                            acoustic=float(acoustic_responsibility(-float(nll.detach()), log_q, pi))))
                        nll = mixture_nll(nll, log_q, pi)
                    if not torch.isfinite(nll):
                        raise ValueError('Non-finite full transcript likelihood')
                    terms.append(nll/len(data['features']))
                loss = sum(terms)/len(terms)
                loss = loss+RECIPE['anchor']*sum(preservation_loss(params, d)
                    for d in support+protected)/len(support+protected)
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in h.parameters()):
                    raise ValueError('Invalid head gradient')
                with torch.no_grad():
                    for p in h.parameters():
                        p.add_(p.grad, alpha=-RECIPE['lr'])
            updates += 1
        losses = {name: evaluate(h) for name, h in arms.items()}
        for name, h in arms.items():
            np.savez(out/f'{name}-{epoch+1}.npz', weight=h.weight.detach().numpy(), bias=h.bias.detach().numpy())
        write_json(out/f'responsibilities-{epoch+1}.json', responsibility)
        history.append(dict(epoch=epoch+1, updates=updates, validation=losses, contamination=pi,
            mean_acoustic_responsibility=float(np.mean([r['acoustic'] for r in responsibility]))))
        print(json.dumps(history[-1]), flush=True)
    selected = min(history, key=lambda r: (r['validation']['mixture'], r['epoch']))
    paths = {name: out/f"{name}-{selected['epoch']}.npz" for name in arms}
    write_json(out/'selection.json', dict(schema=1, plan_sha256=digest(out/'plan.json'), baseline_validation=baseline,
        selected=selected, history=history, arms={name: dict(path=str(path), sha256=digest(path)) for name, path in paths.items()},
        identical_updates=True, identical_minibatches=True, identical_preservation=True,
        checkpoint_selection='reserved training speakers: ordinary CTC plus shipped-output preservation'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'fit', 'export'))
    p.add_argument('--model', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.command != 'fit' and args.model is None:
        p.error('prepare/export require --model')
    if args.command == 'prepare':
        prepare(args.model, args.out)
    elif args.command == 'fit':
        fit(args.out)
    else:
        export(args.model, args.out, read_plan, RECIPE['seed'])

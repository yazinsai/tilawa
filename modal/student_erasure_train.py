"""Matched ordinary, random-emission and strongest-consonant-emission refits.

Run prepare, fit, export through the supervisor's training image. Enrollment and
checkpoint selection use authorized training speakers.
No practice audio, incident or scoring interface is read by this runner.
"""

import argparse
import copy
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from student_head_refit import (ROOT, authorized_records, clean_target, digest,
                               export, feature_runtime, inherited_assets, stable, stream_features,
                               training_helpers, write_json)
from student_erasure import ARMS, head_objective, sequence_nll, viterbi_states

sys.path.insert(0, str(ROOT / 'tools'))

# Frozen before any target query. All three arms share every setting and sample.
RECIPE = dict(seed=831, original_limit=512, views=3, epochs=12, batch=8,
              lr=.0001, anchor=2., max_seconds=30., sequence_evaluations=2)


def prepare(model, out):
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import torchaudio
    from student import TOKENS, mix_noise

    out.mkdir(parents=True, exist_ok=True)
    if (out / 'plan.json').exists():
        plan = read_plan(out)
        if (digest(model/'manifest.json') != plan['manifest_sha256'] or
                {file: sha for file, _, sha in inherited_assets(model)} != plan['auxiliary_sha256']):
            raise ValueError('Source bundle changed after preparation')
        print('Using the already frozen training cache', flush=True)
        return
    records, excluded, labels_sha = authorized_records(RECIPE['original_limit'], RECIPE['max_seconds'])
    auxiliary = inherited_assets(model)
    manifest, head, runtime = feature_runtime(model, out, RECIPE['seed'])
    if (manifest['tokens'] != TOKENS or manifest['blank'] != 0 or manifest['frameMs'] != 40 or
            manifest['sampleRate'] != 16000 or manifest['fbank']['mels'] != 80):
        raise ValueError('Inventory, clock or frontend differs from shipped listener')
    np.savez(out / 'initial-head.npz', weight=head['weight'].T, bias=head['bias'])
    ref = json.loads((ROOT / 'packages/core/assets/ref-v2.json').read_text())
    ayahs = {f"{a['s']}:{a['a']}": a['p'] for a in ref['ayahs']}
    waves = {}
    for path in sorted({r['file'] for r in records if r['source'] == 'original'}):
        selected = [r for r in records if r['file'] == path]
        rows = pq.read_table(path, columns=['flac', 'ref']).take(
            [r['index'] for r in selected]).to_pylist()
        for r, row in zip(selected, rows):
            wave, sr = sf.read(io.BytesIO(row['flac']), dtype='float32')
            if sr != 16000 or wave.ndim != 1:
                raise ValueError('Original prepared audio is not mono 16 kHz')
            waves[r['key']] = wave
            # Authorized transcript phone identity, never teacher argmax text.
            r['target'] = clean_target(row['ref'], manifest['tokens'])
    for r in records:
        if r['source'] == 'original':
            continue
        wave, sr = sf.read(r['file'], dtype='float32')
        if wave.ndim == 2:
            wave = wave.mean(1)
        if sr != 16000:
            divisor = math.gcd(sr, 16000)
            wave = resample_poly(wave, 16000 // divisor, sr // divisor).astype('float32')
        waves[r['key']] = wave
        r['target'] = clean_target(''.join(ayahs[a] for a in r['ayahs']), manifest['tokens'])
    seen = {}
    source_hashes = {path: digest(path) for path in {r['file'] for r in records}}
    for r in records:
        if not r['target']:
            raise ValueError('Authorized transcript is empty')
        sha = hashlib.sha256(np.ascontiguousarray(waves[r['key']]).tobytes()).hexdigest()
        if sha in seen and seen[sha] != r['speaker']:
            raise ValueError('Identical audio appears under different training speakers')
        seen[sha] = r['speaker']
        r['waveform_sha256'] = sha
        r['source_sha256'] = source_hashes[r['file']]
    helpers = training_helpers()
    aug = {k: np.load(Path('/student/aug') / (k + '.npz')) for k in ('noise', 'rir')}
    cache = out / 'features'
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
                wave *= 10 ** (rng.uniform(-10, 6) / 20)
            feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(wave).unsqueeze(0), num_mel_bins=80,
                sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400)
            if view:
                feats = helpers['spec_augment'](feats, rng)
            x, lp = stream_features(runtime, manifest, feats.numpy(), head['feature'])
            logits = x @ head['weight'] + head['bias']
            peak = logits.max(1, keepdims=True)
            reconstructed = logits - peak - np.log(np.exp(logits - peak).sum(1, keepdims=True))
            if not np.allclose(reconstructed, lp, atol=.002, rtol=0):
                raise ValueError('Frozen features do not reconstruct the shipped head')
            if viterbi_states(lp, np.asarray(r['target'])) is None:
                raise ValueError('Complete authorized transcript cannot align')
            file = cache / (stable(f"{r['key']}:{view}") + '.npz')
            np.savez_compressed(file, features=x, baseline=lp, fbank=feats.numpy(),
                target=np.asarray(r['target'], dtype=np.int64),
                waveform_sha256=r['waveform_sha256'])
            v = dict(recording=r['key'], speaker=r['speaker'], partition=r['partition'], view=view, file=str(file))
            views.append(v)
        print(json.dumps(dict(cached=r['key'], partition=r['partition'])), flush=True)
    for v in views:
        v['sha256'] = digest(v['file'])
    write_json(out / 'plan.json', dict(schema=1, recipe=RECIPE, records=records, views=views,
        excluded_speakers=excluded, help_labels_sha256=labels_sha, primary_sha256=manifest['sha256'],
        manifest_sha256=digest(model/'manifest.json'),
        auxiliary_sha256={file: sha for file, _, sha in auxiliary},
        augmentation_sha256={k: digest(Path('/student/aug') / (k + '.npz')) for k in ('noise', 'rir')},
        feature_graph_sha256=digest(out / 'encoder-features.onnx'),
        initial_head_sha256=digest(out / 'initial-head.npz'), quantized=head['quantized'],
        per_channel=head['scale_shape'] == (len(TOKENS) + 1,),
        exclusion_semantics='uniform row, transcript Viterbi occurrence, no waveform or time-axis masking'))


def read_plan(out):
    plan = json.loads((out / 'plan.json').read_text())
    if plan['recipe'] != RECIPE:
        raise ValueError('Frozen training settings changed')
    for file, key in [('initial-head.npz', 'initial_head_sha256'), ('encoder-features.onnx', 'feature_graph_sha256')]:
        if digest(out / file) != plan[key]:
            raise ValueError('Frozen head or encoder feature graph changed')
    for v in plan['views']:
        if digest(v['file']) != v['sha256']:
            raise ValueError('Frozen features changed')
    records = {r['key']: r for r in plan['records']}
    for v in plan['views']:
        r = records[v['recording']]
        if r['speaker'] != v['speaker'] or r['partition'] != v['partition']:
            raise ValueError('Cached view lost its recording provenance')
    train = {v['speaker'] for v in plan['views'] if v['partition'] == 'train'}
    validation = {v['speaker'] for v in plan['views'] if v['partition'] == 'validation'}
    if (not train or not validation or train & validation or
            (train | validation) & set(plan['excluded_speakers'])):
        raise ValueError('Training-speaker separation violated')
    return plan


def quantized_parameters(head, plan):
    """Reproduce bundle weight rounding for training-only checkpoint selection."""
    import torch
    w = head.weight.detach()
    if plan['quantized']:
        maximum = w.abs().amax(1, keepdim=True) if plan['per_channel'] else w.abs().max()
        scale = (maximum / 127).clamp_min(1e-8)
        w = (w / scale).round().clamp(-127, 127) * scale
    return w, head.bias.detach()


def choose_checkpoint(history):
    """Select strongest-erasure arm by ordinary held-speaker loss; pair all steps."""
    if not history or any(not math.isfinite(h['validation']['strongest']) for h in history):
        raise ValueError('Missing or invalid training-only validation loss')
    return min(history, key=lambda h: (h['validation']['strongest'], h['epoch']))


def fit(out):
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(2)
    torch.manual_seed(RECIPE['seed'])
    plan = read_plan(out)
    train = [v for v in plan['views'] if v['partition'] == 'train']
    validation = [v for v in plan['views'] if v['partition'] == 'validation' and v['view'] == 0]

    @lru_cache(maxsize=64)
    def cached(path):
        with np.load(path) as a:
            return {k: torch.from_numpy(a[k].copy()) for k in ('features', 'baseline', 'target')}

    with np.load(out / 'initial-head.npz') as a:
        head = torch.nn.Linear(a['weight'].shape[1], a['weight'].shape[0])
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(a['weight']))
            head.bias.copy_(torch.from_numpy(a['bias']))
    arms = {name: copy.deepcopy(head) for name in ARMS}

    from student import TOKENS

    def objective(parameters, d, name, draw):
        lp = F.linear(d['features'], *parameters).log_softmax(-1)
        return head_objective(lp, d['target'], d['baseline'], TOKENS,
                              np.random.default_rng(draw), name, RECIPE['anchor'])

    def evaluate(h):
        speakers = {}
        with torch.no_grad():
            for v in validation:
                d = cached(v['file'])
                lp = F.linear(d['features'], *quantized_parameters(h, plan)).log_softmax(-1)
                loss = 2 * sequence_nll(lp, d['target']) + RECIPE['anchor'] * F.kl_div(
                    lp, d['baseline'].exp(), reduction='sum') / len(lp)
                speakers.setdefault(v['speaker'], []).append(float(loss))
        value = float(np.mean([np.mean(x) for x in speakers.values()]))
        if not math.isfinite(value):
            raise ValueError('Invalid held-training-speaker loss')
        return value

    history, updates = [], 0
    audit = {name: dict(evaluations=0, erased=0, skipped=0) for name in ARMS}
    for epoch in range(RECIPE['epochs'] + 1):
        if epoch:
            order = np.random.default_rng(RECIPE['seed'] + epoch).permutation(len(train))
            for at in range(0, len(order), RECIPE['batch']):
                batch = [train[i] for i in order[at:at + RECIPE['batch']]]
                for name, h in arms.items():
                    h.zero_grad(set_to_none=True)
                    losses = []
                    for v in batch:
                        draw = int(stable(f"{RECIPE['seed']}:{epoch}:{v['recording']}:{v['view']}")[:16], 16)
                        value, report = objective(tuple(h.parameters()), cached(v['file']), name, draw)
                        audit[name]['evaluations'] += 2
                        audit[name]['erased'] += int(name != 'ordinary' and report['skipped'] is None)
                        audit[name]['skipped'] += int(report['skipped'] is not None)
                        losses.append(value)
                    loss = sum(losses) / len(batch)
                    loss.backward()
                    if not math.isfinite(float(loss.detach())) or any(not torch.isfinite(p.grad).all() for p in h.parameters()):
                        raise ValueError('Nonfinite matched phone-head update')
                    torch.nn.utils.clip_grad_norm_(h.parameters(), 5.)
                    with torch.no_grad():
                        for p in h.parameters():
                            p.add_(p.grad, alpha=-RECIPE['lr'])
                updates += 1
        losses = {name: evaluate(h) for name, h in arms.items()}
        for name, h in arms.items():
            np.savez(out / f'{name}-{epoch}.npz', weight=h.weight.detach().numpy(), bias=h.bias.detach().numpy())
        history.append(dict(epoch=epoch, updates=updates, validation=losses))
        print(json.dumps(history[-1]), flush=True)
    selected = choose_checkpoint(history)
    paths = {name: out / f"{name}-{selected['epoch']}.npz" for name in arms}
    write_json(out / 'selection.json', dict(schema=1, plan_sha256=digest(out / 'plan.json'),
        selected=selected, history=history, audit=audit,
        arms={name: dict(path=str(path), sha256=digest(path)) for name, path in paths.items()},
        identical_updates=True, identical_batches=True,
        checkpoint_selection='whole held training speakers; ordinary CTC plus full-recording shipped-output preservation; exported weight rounding'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'fit', 'export'))
    parser.add_argument('--model', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if 'LAB_DATA' not in os.environ:
        parser.error('Training/export must run through the authorized supervisor')
    scratch = Path('/scratch/listener-emission-erasure-training').resolve()
    if not args.out.resolve().is_relative_to(scratch):
        parser.error('Output must stay inside this experiment scratch directory')
    if args.command != 'fit' and args.model is None:
        parser.error('prepare/export require --model')
    if args.command == 'prepare':
        prepare(args.model, args.out)
    elif args.command == 'fit':
        fit(args.out)
    else:
        export(args.model, args.out, read_plan, RECIPE['seed'])

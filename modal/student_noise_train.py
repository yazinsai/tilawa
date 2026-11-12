"""Matched frozen-head refits with ordinary or forward-corrected teacher KD.

Run prepare, fit, export through the supervisor's training image. All enrollment,
channel selection and checkpoint selection use authorized training speakers.
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

import numpy as np

from student_head_refit import (ROOT, authorized_records, clean_target, digest,
                               export, feature_runtime, inherited_assets, stable, stream_features,
                               training_helpers, write_json)
from student_noise import Observation, fit_channel, head_loss, observation_nll

# Frozen before any target query. Both arms share every setting and sample.
RECIPE = dict(seed=831, original_limit=512, views=3, epochs=12, batch=8,
              lr=.0001, anchor=2., ctc=1., kd=1., pause=1., max_seconds=30.,
              channel_prior_fractions=[.01, .1, 1.], channel_iterations=12)


def digital_pause(wave, frames):
    """Independent, conservative real-silence labels on the existing 40 ms clock.

    Only 120 ms neighborhoods entirely below one 16-bit PCM quantization step
    count as silence. Teacher blank, CTC alignment and utterance padding never
    define this mask. Other real pauses still receive ordinary-output anchors.
    """
    mask = np.zeros(frames, dtype=bool)
    for t in range(frames):
        center = (t + 1) * 640
        lo, hi = center - 960, center + 960
        if lo >= 0 and hi <= len(wave):
            mask[t] = np.max(np.abs(wave[lo:hi])) <= 1 / 32768
    return mask


def full_topk(ids, probabilities, vocabulary, frames):
    """The existing cached top-five targets, normalized without hard labels."""
    ids = np.frombuffer(ids, dtype=np.uint8).reshape(-1, 5)
    p = np.frombuffer(probabilities, dtype=np.float16).astype('float32').reshape(-1, 5)
    if ids.shape != p.shape or not len(p) or (ids >= vocabulary).any() or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid cached teacher soft targets')
    mass = p.sum(-1, keepdims=True)
    if (mass <= 0).any():
        raise ValueError('Cached teacher target has no probability mass')
    p /= mass
    q = np.zeros((len(p), vocabulary), dtype='float32')
    for k in range(5):
        np.add.at(q, (np.arange(len(p)), ids[:, k]), p[:, k])
    # Preserve the shipped zero-delay 25 Hz mapping, including final clamping.
    return q[np.minimum(np.arange(frames), len(q) - 1)]


def teacher_probabilities(runtime, wave, frames):
    """The shipped inference-mode teacher and its SeamlessM4T frontend."""
    import torch
    import torchaudio

    f = torchaudio.compliance.kaldi.fbank(torch.from_numpy(wave * 32768).unsqueeze(0),
        num_mel_bins=80, sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=8000)
    if len(f) < 2:
        raise ValueError('Not enough teacher features')
    f = ((f - f.mean(0)) / (f.var(0, unbiased=True) + 1e-7).sqrt()).numpy()
    n = math.ceil(len(f) / 2)
    stacked = np.ones((n * 2, 80), dtype='float32')
    stacked[:len(f)] = f
    mask = np.ones((1, n), dtype=np.int64)
    if len(f) % 2:
        mask[0, -1] = 0
    lp = runtime.run(['log_probs'], dict(input_features=stacked.reshape(1, n, 160), attention_mask=mask))[0][0]
    if not np.isfinite(lp).all() or not len(lp):
        raise ValueError('Teacher returned invalid soft targets')
    q = np.exp(lp)
    q /= q.sum(-1, keepdims=True)
    return q[np.minimum(np.arange(frames), len(q) - 1)]


def prepare(model, out):
    import onnxruntime as ort
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import torchaudio
    from student import TOKENS, mix_noise

    out.mkdir(parents=True, exist_ok=True)
    if (out / 'plan.json').exists():
        raise ValueError('Training plan already frozen; use a new directory')
    records, excluded, labels_sha = authorized_records(RECIPE['original_limit'], RECIPE['max_seconds'])
    auxiliary = inherited_assets(model)
    manifest, head, runtime = feature_runtime(model, out, RECIPE['seed'])
    if (manifest['tokens'] != TOKENS or manifest['blank'] != 0 or manifest['frameMs'] != 40 or
            manifest['sampleRate'] != 16000 or manifest['fbank']['mels'] != 80):
        raise ValueError('Inventory, clock or frontend differs from shipped listener')
    np.savez(out / 'initial-head.npz', weight=head['weight'].T, bias=head['bias'])
    ref = json.loads((ROOT / 'packages/core/assets/ref-v2.json').read_text())
    ayahs = {f"{a['s']}:{a['a']}": a['p'] for a in ref['ayahs']}
    waves, original_targets = {}, {}
    for path in sorted({r['file'] for r in records if r['source'] == 'original'}):
        selected = [r for r in records if r['file'] == path]
        rows = pq.read_table(path, columns=['flac', 'ref', 'top5_ids', 'top5_p']).take(
            [r['index'] for r in selected]).to_pylist()
        for r, row in zip(selected, rows):
            wave, sr = sf.read(io.BytesIO(row['flac']), dtype='float32')
            if sr != 16000 or wave.ndim != 1:
                raise ValueError('Original prepared audio is not mono 16 kHz')
            waves[r['key']] = wave
            # Authorized transcript phone identity, never teacher argmax text.
            r['target'] = clean_target(row['ref'], manifest['tokens'])
            original_targets[r['key']] = row['top5_ids'], row['top5_p']
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
    views, help_teacher_requests = [], []
    torch.set_num_threads(2)
    for r in records:
        q = None
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
            if r['source'] == 'original' and q is None:
                q = full_topk(*original_targets[r['key']], len(TOKENS) + 1, len(x))
            file = cache / (stable(f"{r['key']}:{view}") + '.npz')
            np.savez_compressed(file, features=x, baseline=lp, target=np.asarray(r['target'], dtype=np.int64),
                pause=digital_pause(waves[r['key']], len(x)), teacher=q if q is not None else np.empty((0, len(TOKENS)+1)))
            v = dict(recording=r['key'], speaker=r['speaker'], partition=r['partition'], view=view, file=str(file))
            views.append(v)
            if r['source'] == 'help':
                help_teacher_requests.append((r, v))
        print(json.dumps(dict(cached=r['key'], partition=r['partition'])), flush=True)
    # Release encoder memory before the large teacher; it runs only on clean
    # original takes. Both arms inherit the same cached augmentation mapping.
    del runtime
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    teacher_path = model / 'teacher.onnx'
    if not teacher_path.is_file():
        raise FileNotFoundError('Unchanged insertion/KD teacher is missing')
    teacher = ort.InferenceSession(str(teacher_path), options, providers=['CPUExecutionProvider'])
    for r in records:
        if r['source'] != 'help':
            continue
        selected = [v for record, v in help_teacher_requests if record is r]
        with np.load(selected[0]['file']) as a:
            frames = len(a['features'])
        q = teacher_probabilities(teacher, waves[r['key']], frames)
        if q.shape != (frames, len(TOKENS) + 1):
            raise ValueError('Teacher alphabet differs from the shipped listener')
        for v in selected:
            with np.load(v['file']) as a:
                contents = {k: a[k].copy() for k in a.files}
            contents['teacher'] = q
            np.savez_compressed(v['file'], **contents)
    for v in views:
        v['sha256'] = digest(v['file'])
    write_json(out / 'plan.json', dict(schema=1, recipe=RECIPE, records=records, views=views,
        excluded_speakers=excluded, help_labels_sha256=labels_sha, primary_sha256=manifest['sha256'],
        teacher_sha256=digest(teacher_path), auxiliary_sha256={file: sha for file, _, sha in auxiliary},
        augmentation_sha256={k: digest(Path('/student/aug') / (k + '.npz')) for k in ('noise', 'rir')},
        feature_graph_sha256=digest(out / 'encoder-features.onnx'),
        initial_head_sha256=digest(out / 'initial-head.npz'), quantized=head['quantized'],
        per_channel=head['scale_shape'] == (len(TOKENS) + 1,),
        teacher_semantics='cached original top5; inference ONNX help; zero-delay clamped 25Hz; no argmax labels'))


def read_plan(out):
    plan = json.loads((out / 'plan.json').read_text())
    if plan['recipe'] != RECIPE:
        raise ValueError('Frozen training settings changed')
    for file, key in [('initial-head.npz', 'initial_head_sha256'), ('encoder-features.onnx', 'feature_graph_sha256')]:
        if digest(out / file) != plan[key]:
            raise ValueError('Frozen head or encoder feature graph changed')
    for v in plan['views']:
        if digest(v['file']) != v['sha256']:
            raise ValueError('Frozen features or teacher observations changed')
    train = {v['speaker'] for v in plan['views'] if v['partition'] == 'train'}
    validation = {v['speaker'] for v in plan['views'] if v['partition'] == 'validation'}
    if not train or not validation or train & validation or (train | validation) & set(plan['excluded_speakers']):
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
    """Select the corrected arm by ordinary held-speaker loss, pair same step."""
    if not history or any(not math.isfinite(h['validation']['corrected']) for h in history):
        raise ValueError('Missing or invalid training-only validation loss')
    return min(history, key=lambda h: (h['validation']['corrected'], h['epoch']))


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
            return {k: torch.from_numpy(a[k].copy()) for k in a.files}

    def observation(v):
        d = cached(v['file'])
        return Observation(v['speaker'], tuple(d['target'].tolist()), d['teacher'].numpy(), d['pause'].numpy())

    channel_train = [observation(v) for v in train if v['view'] == 0]
    channel_validation = [observation(v) for v in validation]
    if not channel_train or not channel_validation:
        raise ValueError('Missing speaker-grouped channel training or validation')
    V = channel_train[0].teacher.shape[1]
    frame_unit = sum(len(x.teacher) for x in channel_train) / V
    candidates = []
    for fraction in RECIPE['channel_prior_fractions']:
        c = fit_channel(channel_train, fraction * frame_unit, RECIPE['channel_iterations'])
        score = observation_nll(c, channel_validation)
        candidates.append((score, fraction, c))
        print(json.dumps(dict(channel_prior_fraction=fraction, validation_observation_nll=score)), flush=True)
    score, fraction, channel = min(candidates, key=lambda x: (x[0], -x[1]))
    np.save(out / 'channel.npy', channel)
    channel_sha = digest(out / 'channel.npy')
    channels = {'ordinary': torch.eye(V), 'corrected': torch.from_numpy(channel.copy()).float()}
    with np.load(out / 'initial-head.npz') as a:
        head = torch.nn.Linear(a['weight'].shape[1], a['weight'].shape[0])
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(a['weight']))
            head.bias.copy_(torch.from_numpy(a['bias']))
    arms = {'ordinary': head, 'corrected': copy.deepcopy(head)}

    def objective(parameters, d, c, selection=False):
        lp = F.linear(d['features'], *parameters).log_softmax(-1)[None]
        T = lp.shape[1]
        loss, _ = head_loss(lp, d['teacher'][None], d['target'], torch.tensor([T]),
            torch.tensor([len(d['target'])]), d['pause'][None], c,
            ctc_weight=RECIPE['ctc'], kd_weight=0. if selection else RECIPE['kd'], pause_weight=RECIPE['pause'])
        anchor = F.kl_div(lp[0], d['baseline'].exp(), reduction='sum') / T
        return loss + RECIPE['anchor'] * anchor

    def evaluate(h):
        speakers = {}
        with torch.no_grad():
            for v in validation:
                loss = objective(quantized_parameters(h, plan), cached(v['file']), torch.eye(V), selection=True)
                speakers.setdefault(v['speaker'], []).append(float(loss))
        value = float(np.mean([np.mean(x) for x in speakers.values()]))
        if not math.isfinite(value):
            raise ValueError('Invalid held-training-speaker loss')
        return value

    history, updates = [], 0
    for epoch in range(RECIPE['epochs'] + 1):
        if epoch:
            order = np.random.default_rng(RECIPE['seed'] + epoch).permutation(len(train))
            for at in range(0, len(order), RECIPE['batch']):
                batch = [cached(train[i]['file']) for i in order[at:at + RECIPE['batch']]]
                for name, h in arms.items():
                    h.zero_grad(set_to_none=True)
                    loss = sum(objective(tuple(h.parameters()), d, channels[name]) for d in batch) / len(batch)
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
    if digest(out / 'channel.npy') != channel_sha:
        raise ValueError('Channel changed during head refitting')
    selected = choose_checkpoint(history)
    paths = {name: out / f"{name}-{selected['epoch']}.npz" for name in arms}
    write_json(out / 'selection.json', dict(schema=1, plan_sha256=digest(out / 'plan.json'),
        selected=selected, history=history, channel_sha256=channel_sha, channel_prior_fraction=fraction,
        channel_validation_nll=score, channel_candidates=[dict(nll=s, fraction=f) for s, f, _ in candidates],
        arms={name: dict(path=str(path), sha256=digest(path)) for name, path in paths.items()},
        identical_updates=True, identical_batches=True,
        checkpoint_selection='whole held training speakers; ordinary CTC plus pause and shipped-output preservation; exported weight rounding'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'fit', 'export'))
    parser.add_argument('--model', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if 'LAB_DATA' not in os.environ:
        parser.error('Training/export must run through the authorized supervisor')
    scratch = Path('/scratch/listener-noise-aware-teacher-loss').resolve()
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

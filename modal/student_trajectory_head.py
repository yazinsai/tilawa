"""Average one converged frozen-encoder phone-head optimization trajectory.

prepare/fit/export never read practice recordings or scores. All schedule and
checkpoint decisions use whole-speaker partitions of authorized training data.
The exported primary is one linear head, not an inference ensemble.
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

from student_head_refit import (ROOT, authorized_records, clean_target, digest,
    export, feature_runtime, has_clean_transcript, stable, stream_features,
    training_helpers, write_json)

RECIPE = dict(seed=831, original_limit=512, views=3, batch=8,
    preservation_batch=4, lr=.001, momentum=.9, anchor=2., max_seconds=30.,
    convergence_relative=.002, convergence_patience=4, warmup_max_epochs=48,
    tail_epochs=8, average_last_epochs=[8, 4, 2])
EXPERIMENT = 'listener-trajectory-averaged-readout'


def speaker_partitions(records):
    """Reserve entire training speakers separately within each source."""
    mapping = {}
    for source in ('original', 'help'):
        speakers = sorted({r['speaker'] for r in records if r['source'] == source}, key=stable)
        if len(speakers) < 3:
            raise ValueError('At least three authorized training speakers required per source')
        reserved = max(1, len(speakers)//5)
        for i, speaker in enumerate(speakers):
            mapping[speaker] = 'validation' if i < reserved else 'train'
    return mapping


def help_preservation(max_seconds):
    """Scripted/repeat takes preserve shipped output; scripts are not transcripts."""
    import soundfile as sf

    root = Path(os.environ['LAB_DATA'])/'help-train'
    labels = [json.loads(s) for s in (root/'labels.jsonl').read_text().splitlines() if s.strip()]
    banned = {str(l['speaker']) for l in labels if l.get('split') != 'train' or
              l.get('exclude') or l.get('excluded')}
    rows = []
    for label in labels:
        if (label.get('split') != 'train' or str(label['speaker']) in banned or
                has_clean_transcript(label)):
            continue
        path = (root/label['wav']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Missing or unauthorized preservation waveform')
        if 0 < sf.info(path).duration <= max_seconds:
            rows.append(dict(key='help/'+label['id'], source='preservation', file=str(path),
                             speaker='help/'+str(label['speaker']), target=[]))
    if not rows:
        raise ValueError('Mistake/repeat preservation recordings are required')
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
    partitions = speaker_partitions(records)
    preservation = help_preservation(RECIPE['max_seconds'])
    for r in records+preservation:
        # Speakers with no supervised take enter transcript-free fit preservation
        # only. They are never also claimed as independent validation speakers.
        r['partition'] = partitions.get(r['speaker'], 'train')
    records += preservation
    manifest, head, runtime = feature_runtime(model, out, RECIPE['seed'])
    if (manifest['tokens'] != TOKENS or manifest['blank'] != 0 or manifest['frameMs'] != 40 or
            manifest['sampleRate'] != 16000 or manifest['fbank']['mels'] != 80):
        raise ValueError('Feature capture must retain the shipped inventory and clock')
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
                raise ValueError('Original prepared audio must be mono 16 kHz')
            waves[r['key']] = wave
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
        raise ValueError('Authorized supervised take has no transcript')
    seen = {}
    for r in records:
        sha = hashlib.sha256(np.ascontiguousarray(waves[r['key']]).tobytes()).hexdigest()
        if sha in seen and seen[sha] != r['speaker']:
            raise ValueError('Identical performance has conflicting speaker identities')
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
            features, baseline = stream_features(runtime, manifest, feats.numpy(), head['feature'])
            if not np.isfinite(features).all() or not np.isfinite(baseline).all():
                raise ValueError('Invalid frozen streaming features')
            path = cache/(stable(r['key'])+f'-{view}.npz')
            np.savez_compressed(path, features=features.astype('float32'), baseline=baseline.astype('float32'),
                                target=np.asarray(r['target'], dtype='int64'))
            views.append(dict(file=str(path), sha256=digest(path), recording=r['key'], speaker=r['speaker'],
                partition=r['partition'], supervised=r['source'] != 'preservation', view=view))
        print(json.dumps(dict(cached=len(views), recordings=len(records))), flush=True)
    write_json(out/'plan.json', dict(schema=1, id=EXPERIMENT, recipe=RECIPE, records=records, views=views,
        excluded_speakers=excluded, labels_sha256=labels_sha, primary_sha256=manifest['sha256'],
        initial_head_sha256=digest(out/'initial-head.npz'), feature_graph_sha256=digest(out/'encoder-features.onnx')))


def read_plan(out):
    plan = json.loads((out/'plan.json').read_text())
    if plan['recipe'] != RECIPE or plan['id'] != EXPERIMENT:
        raise ValueError('Frozen experiment or recipe changed')
    for file, sha in [('initial-head.npz', plan['initial_head_sha256']),
                      ('encoder-features.onnx', plan['feature_graph_sha256'])]:
        if digest(out/file) != sha:
            raise ValueError('Frozen head or feature graph changed')
    groups = {}
    for view in plan['views']:
        if digest(view['file']) != view['sha256']:
            raise ValueError('Frozen feature cache changed')
        if view['partition'] not in ('train', 'validation'):
            raise ValueError('Unknown training partition')
        previous = groups.setdefault(view['speaker'], view['partition'])
        if previous != view['partition'] or view['speaker'] in plan['excluded_speakers']:
            raise ValueError('Whole-speaker separation violated')
    return plan


class ParameterAverage:
    """Equal-weight parameter mean; no optimizer state or output voting."""
    def __init__(self):
        self.count = 0
        self.mean = None

    def add(self, head):
        state = head.state_dict()
        if self.mean is None:
            self.mean = {k: v.detach().clone().double() for k, v in state.items()}
        else:
            if state.keys() != self.mean.keys():
                raise ValueError('Trajectory parameter geometry changed')
            for key, value in state.items():
                if value.shape != self.mean[key].shape:
                    raise ValueError('Trajectory parameter geometry changed')
                self.mean[key].add_((value.detach().double()-self.mean[key])/(self.count+1))
        self.count += 1

    def head(self, template):
        if not self.count:
            raise ValueError('Cannot export an empty parameter average')
        result = copy.deepcopy(template)
        result.load_state_dict({k: v.to(template.state_dict()[k].dtype) for k, v in self.mean.items()})
        return result


def objective(head, data):
    import torch
    import torch.nn.functional as F

    lp = head(data['features']).log_softmax(-1)
    loss = RECIPE['anchor']*F.kl_div(lp, data['baseline'].exp(), reduction='sum')/len(lp)
    if len(data['target']):
        loss = loss+F.ctc_loss(lp[:, None], data['target'], torch.tensor([len(lp)]),
            torch.tensor([len(data['target'])]), reduction='sum', zero_infinity=False)/len(lp)
    if not torch.isfinite(loss):
        raise ValueError('Impossible transcript or nonfinite preservation objective')
    return loss


def converged(history):
    """Convergence depends only on the deterministic fit-bank objective."""
    patience = RECIPE['convergence_patience']
    if len(history) <= patience:
        return False
    recent = history[-patience-1:]
    if not all(math.isfinite(v) for v in recent):
        raise ValueError('Invalid convergence measurements')
    return all(abs(a-b)/max(abs(a), 1e-12) <= RECIPE['convergence_relative']
               for a, b in zip(recent, recent[1:]))


def continuation_lr(epoch):
    """Ordinary annealed continuation; matched data, optimizer and update count."""
    return RECIPE['lr']*.5*(1+math.cos(math.pi*epoch/RECIPE['tail_epochs']))


def fit(out):
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
    train = [v for v in plan['views'] if v['partition'] == 'train' and v['supervised']]
    bank = [v for v in plan['views'] if v['partition'] == 'train' and not v['supervised']]
    validation = [v for v in plan['views'] if v['partition'] == 'validation']
    if not train or not bank or not validation or not any(v['supervised'] for v in validation):
        raise ValueError('Supervision, preservation and validation banks are required')

    @lru_cache(maxsize=32)
    def cached(path):
        with np.load(path) as a:
            return {k: torch.from_numpy(a[k].copy()) for k in ('features', 'baseline', 'target')}

    def evaluate(h, views):
        by_speaker = {}
        with torch.no_grad():
            for v in views:
                by_speaker.setdefault(v['speaker'], []).append(float(objective(h, cached(v['file']))))
        return float(np.mean([np.mean(values) for values in by_speaker.values()]))

    def optimizer(h):
        return torch.optim.SGD(h.parameters(), lr=RECIPE['lr'], momentum=RECIPE['momentum'])

    updates = 0
    def epoch(arms, epoch_number, averages=None):
        nonlocal updates
        order = np.random.default_rng(RECIPE['seed']+epoch_number).permutation(len(train))
        for start in range(0, len(order), RECIPE['batch']):
            support = [cached(train[i]['file']) for i in order[start:start+RECIPE['batch']]]
            protected = [cached(bank[(updates*RECIPE['preservation_batch']+j) % len(bank)]['file'])
                         for j in range(RECIPE['preservation_batch'])]
            for h, opt in arms:
                opt.zero_grad(set_to_none=True)
                # Each supervised recording receives its own preservation loss;
                # mistake/repeat recordings receive preservation alone.
                loss = sum(objective(h, d) for d in support)/len(support)
                loss = loss+sum(objective(h, d) for d in protected)/len(protected)
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in h.parameters()):
                    raise ValueError('Nonfinite head gradient')
                opt.step()
            if averages:
                for average in averages:
                    average.add(arms[0][0])
            updates += 1

    opt = optimizer(head)
    fit_views = train+bank
    warmup = [evaluate(head, fit_views)]
    baseline_validation = evaluate(head, validation)
    for i in range(RECIPE['warmup_max_epochs']):
        epoch([(head, opt)], i)
        warmup.append(evaluate(head, fit_views))
        print(json.dumps(dict(warmup_epoch=i+1, fit_loss=warmup[-1], updates=updates)), flush=True)
        if converged(warmup):
            break
    if not converged(warmup):
        write_json(out/'convergence.json', dict(converged=False, fit_losses=warmup, updates=updates))
        raise RuntimeError('Training-only convergence not reached; no tail or candidate is declared')
    warmup_epochs = len(warmup)-1
    write_json(out/'convergence.json', dict(converged=True, fit_losses=warmup, epochs=warmup_epochs, updates=updates))
    ordinary = copy.deepcopy(head)
    ordinary_opt = optimizer(ordinary)
    ordinary_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    averages = {n: ParameterAverage() for n in RECIPE['average_last_epochs']}
    history = []
    for i in range(RECIPE['tail_epochs']):
        ordinary_opt.param_groups[0]['lr'] = continuation_lr(i)
        active = [a for n, a in averages.items() if i >= RECIPE['tail_epochs']-n]
        epoch([(head, opt), (ordinary, ordinary_opt)], warmup_epochs+i, active)
        history.append(dict(tail_epoch=i+1, updates=updates, continuation_lr=continuation_lr(i)))
        print(json.dumps(history[-1]), flush=True)
    choices = {n: a.head(head) for n, a in averages.items()}
    losses = {n: evaluate(h, validation) for n, h in choices.items()}
    # Tie-break favours the longer, less selectively trimmed fixed tail.
    selected = min(losses, key=lambda n: (losses[n], -n))
    arms = dict(averaged=choices[selected], terminal=head, ordinary=ordinary)
    checkpoints = {}
    for name, h in arms.items():
        path = out/(name+'.npz')
        np.savez(path, weight=h.weight.detach().numpy(), bias=h.bias.detach().numpy())
        checkpoints[name] = dict(path=str(path), sha256=digest(path))
    write_json(out/'selection.json', dict(schema=1, plan_sha256=digest(out/'plan.json'),
        baseline_validation=baseline_validation, warmup_epochs=warmup_epochs, updates=updates,
        selected_last_epochs=selected, average_counts={n: a.count for n, a in averages.items()},
        validation={**{f'average-last-{n}': v for n, v in losses.items()},
                    'terminal': evaluate(head, validation), 'ordinary': evaluate(ordinary, validation)},
        history=history, arms=checkpoints, identical_minibatches=True, identical_update_counts=True,
        selection='reserved training speakers: complete CTC plus shipped-output preservation'))


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

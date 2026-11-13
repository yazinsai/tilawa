"""Training-only quadratic mode connection on a frozen deployed encoder.

prepare, fit, export are separate so the immutable selection is recorded before
any practice audio is inspected. Neither command accepts practice incidents.
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
import re
import shutil
import time

from student_basin_features import (ROOT, authorized_records, clean_target, digest,
    feature_runtime, has_clean_transcript, stable, stream_features,
    training_helpers, write_json)

RECIPE = dict(seed=2401, original_limit=128, views=2, max_seconds=30.,
    width=64, batch=4, lr=.001, anchor=2., endpoint_epochs=8,
    connection_epochs=8, path_samples=3, interior=[.25, .5, .75], cpu_seconds=86400)
EXPERIMENT = 'listener-connected-basin-readout'


class CpuBudget:
    """Cumulative process CPU time, including feature capture and validation."""
    def __init__(self, out):
        self.path = out/'cpu.json'
        self.used = json.loads(self.path.read_text())['seconds'] if self.path.exists() else 0.
        self.started = time.process_time()

    def check(self):
        used = self.used+time.process_time()-self.started
        write_json(self.path, dict(seconds=used, limit=RECIPE['cpu_seconds']))
        if used >= RECIPE['cpu_seconds']:
            raise RuntimeError('One CPU-day budget exhausted; no candidate may be declared')


def speaker_partitions(records):
    mapping = {}
    for source in ('original', 'help'):
        speakers = sorted({r['speaker'] for r in records if r['source'] == source}, key=stable)
        if len(speakers) < 3:
            raise ValueError('Need three authorized training speakers per source')
        reserved = max(1, len(speakers)//5)
        mapping.update({s: 'validation' if i < reserved else 'train' for i, s in enumerate(speakers)})
    return mapping


def preservation_records(max_seconds):
    """Scripts are not performed transcripts: use only shipped-output KL."""
    import soundfile as sf
    root = Path(os.environ['LAB_DATA'])/'help-train'
    labels = [json.loads(s) for s in (root/'labels.jsonl').read_text().splitlines() if s.strip()]
    banned = {str(l['speaker']) for l in labels if l.get('split') != 'train' or l.get('exclude') or l.get('excluded')}
    rows = []
    for l in labels:
        if l.get('split') != 'train' or str(l['speaker']) in banned or has_clean_transcript(l):
            continue
        path = (root/l['wav']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Unauthorized preservation recording')
        if 0 < sf.info(path).duration <= max_seconds:
            rows.append(dict(key='help/'+l['id'], source='preservation', file=str(path),
                             speaker='help/'+str(l['speaker']), target=[]))
    if not rows:
        raise ValueError('Training-only substitution preservation takes required')
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
    budget = CpuBudget(out)
    if (out/'plan.json').exists():
        raise ValueError('Plan already frozen; use a new output directory')
    records, excluded, labels_sha = authorized_records(RECIPE['original_limit'], RECIPE['max_seconds'])
    partition = speaker_partitions(records)
    records += preservation_records(RECIPE['max_seconds'])
    for r in records:
        r['partition'] = partition.get(r['speaker'], 'train')
    manifest, head, runtime = feature_runtime(model, out, RECIPE['seed'])
    if manifest['tokens'] != TOKENS or manifest['blank'] != 0 or manifest['frameMs'] != 40:
        raise ValueError('Shipped inventory or clock mismatch')
    ref = json.loads((ROOT/'packages/core/assets/ref-v2.json').read_text())
    ayahs = {f"{a['s']}:{a['a']}": a['p'] for a in ref['ayahs']}
    waves = {}
    for path in sorted({r['file'] for r in records if r['source'] == 'original'}):
        selected = [r for r in records if r['file'] == path]
        values = pq.read_table(path, columns=['flac', 'ref']).take([r['index'] for r in selected]).to_pylist()
        for r, value in zip(selected, values):
            wave, sr = sf.read(io.BytesIO(value['flac']), dtype='float32')
            if sr != 16000 or wave.ndim != 1:
                raise ValueError('Prepared original recording must be mono 16 kHz')
            waves[r['key']] = wave
            r['target'] = clean_target(value['ref'], TOKENS)
    for r in records:
        if r['source'] == 'original':
            continue
        wave, sr = sf.read(r['file'], dtype='float32')
        if wave.ndim == 2:
            wave = wave.mean(1)
        if sr != 16000:
            d = math.gcd(sr, 16000)
            wave = resample_poly(wave, 16000//d, sr//d).astype('float32')
        waves[r['key']] = wave
        if r['source'] == 'help':
            r['target'] = clean_target(''.join(ayahs[a] for a in r['ayahs']), TOKENS)
    if any(not r['target'] for r in records if r['source'] != 'preservation'):
        raise ValueError('Supervised recording lacks complete transcript')
    # Exact duplicate performances cannot straddle speaker partitions.
    seen = {}
    for r in records:
        sha = hashlib.sha256(np.ascontiguousarray(waves[r['key']]).tobytes()).hexdigest()
        if sha in seen and seen[sha] != r['speaker']:
            raise ValueError('Duplicate performance has conflicting speaker provenance')
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
            budget.check()
            rng = np.random.default_rng(int(stable(f"{r['key']}:{view}:{RECIPE['seed']}")[:16], 16))
            wave = waves[r['key']].copy()
            if view:
                if rng.random() < .2:
                    wave = helpers['reverb'](wave, helpers['pick'](aug['rir'], rng))
                if rng.random() < .5:
                    wave = mix_noise(wave, helpers['pick'](aug['noise'], rng), rng.uniform(5., 25.), rng)
                wave *= 10**(rng.uniform(-10, 6)/20)
            f = torchaudio.compliance.kaldi.fbank(torch.from_numpy(wave).unsqueeze(0), num_mel_bins=80,
                sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400)
            if view:
                f = helpers['spec_augment'](f, rng)
            x, lp = stream_features(runtime, manifest, f.numpy(), head['feature'])
            if not np.isfinite(x).all() or not np.isfinite(lp).all():
                raise ValueError('Nonfinite encoder cache')
            path = cache/(stable(r['key'])+f'-{view}.npz')
            np.savez_compressed(path, features=x, baseline=lp, target=np.asarray(r['target'], dtype='int64'))
            views.append(dict(file=str(path), sha256=digest(path), recording=r['key'], speaker=r['speaker'],
                partition=r['partition'], supervised=r['source'] != 'preservation', view=view))
        print(json.dumps(dict(cached=len(views))), flush=True)
    budget.check()
    write_json(out/'plan.json', dict(schema=1, id=EXPERIMENT, recipe=RECIPE, records=records, views=views,
        excluded_speakers=excluded, labels_sha256=labels_sha, primary_sha256=manifest['sha256'],
        feature_output=head['feature'], feature_width=head['weight'].shape[0], tokens=manifest['tokens'],
        inventory=manifest['inventory'], feature_graph_sha256=digest(out/'encoder-features.onnx')))


def read_plan(out):
    plan = json.loads((out/'plan.json').read_text())
    if plan['recipe'] != RECIPE or plan['id'] != EXPERIMENT or digest(out/'encoder-features.onnx') != plan['feature_graph_sha256']:
        raise ValueError('Frozen training plan or encoder changed')
    groups = {}
    for v in plan['views']:
        if v['partition'] not in ('train', 'validation') or v['speaker'] in plan['excluded_speakers']:
            raise ValueError('Unauthorized partition')
        if groups.setdefault(v['speaker'], v['partition']) != v['partition'] or digest(v['file']) != v['sha256']:
            raise ValueError('Speaker leak or changed feature cache')
    return plan


def quadratic(left, control, right, t):
    """Parameter-space Bézier curve; endpoints never receive gradients."""
    if not 0 <= t <= 1 or left.keys() != right.keys() or left.keys() != control.keys():
        raise ValueError('Invalid parameter connection')
    return {k: (1-t)**2*left[k].detach()+2*t*(1-t)*control[k]+t*t*right[k].detach() for k in left}


def make_head(features, vocab, seed):
    import torch
    torch.manual_seed(seed)
    head = torch.nn.Sequential(torch.nn.Linear(features, RECIPE['width']), torch.nn.Tanh(),
                               torch.nn.Linear(RECIPE['width'], vocab))
    # Independent small residual readouts preserve the shipped head at startup.
    torch.nn.init.normal_(head[2].weight, std=.01)
    torch.nn.init.normal_(head[2].bias, std=.01)
    return head


def objective(residual, data):
    import torch
    import torch.nn.functional as F
    lp = (data['baseline']+residual).log_softmax(-1)
    loss = RECIPE['anchor']*F.kl_div(lp, data['baseline'].exp(), reduction='sum')/len(lp)
    if len(data['target']):
        loss += F.ctc_loss(lp[:, None], data['target'], torch.tensor([len(lp)]),
                          torch.tensor([len(data['target'])]), reduction='sum', zero_infinity=False)/len(lp)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite or impossible complete-transcript loss')
    return loss


def fit(out):
    import numpy as np
    import torch
    from torch.func import functional_call
    torch.set_num_threads(2)
    budget = CpuBudget(out)
    plan = read_plan(out)
    if (out/'selection.json').exists():
        raise ValueError('Selection already frozen')
    train = [v for v in plan['views'] if v['partition'] == 'train']
    val = [v for v in plan['views'] if v['partition'] == 'validation']
    if not train or not val or not any(not v['supervised'] for v in train):
        raise ValueError('Missing training, validation or substitution-preservation bank')
    # Normalization statistics see training speakers only and weight recordings equally.
    means, seconds = [], []
    for v in train:
        with np.load(v['file']) as a:
            means.append(a['features'].mean(0)); seconds.append((a['features']**2).mean(0))
    center = np.mean(means, axis=0).astype('float32')
    scale = np.sqrt(np.maximum(np.mean(seconds, axis=0)-center**2, 1e-4)).astype('float32')

    @lru_cache(maxsize=32)
    def cached(path):
        with np.load(path) as a:
            return dict(features=torch.from_numpy((a['features']-center)/scale),
                        baseline=torch.from_numpy(a['baseline'].copy()), target=torch.from_numpy(a['target'].copy()))

    def evaluate(head, params=None):
        speakers = {}
        with torch.no_grad():
            for v in val:
                budget.check()
                d = cached(v['file'])
                residual = head(d['features']) if params is None else functional_call(head, params, d['features'])
                speakers.setdefault(v['speaker'], []).append(float(objective(residual, d)))
        return float(np.mean([np.mean(v) for v in speakers.values()]))

    counts = dict(endpoint_a=0, endpoint_b=0, connection=0, ordinary=0)
    def batches(epoch):
        order = np.random.default_rng(RECIPE['seed']+epoch).permutation(len(train))
        return [[cached(train[i]['file']) for i in order[a:a+RECIPE['batch']]]
                for a in range(0, len(order), RECIPE['batch'])]

    def ordinary_step(head, opt, data, name):
        budget.check(); opt.zero_grad(set_to_none=True)
        sum(objective(head(d['features']), d) for d in data).div(len(data)).backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
        opt.step(); counts[name] += len(data)

    endpoints = [make_head(plan['feature_width'], len(plan['tokens'])+1, RECIPE['seed']+i) for i in range(2)]
    opts = [torch.optim.Adam(h.parameters(), lr=RECIPE['lr']) for h in endpoints]
    for epoch in range(RECIPE['endpoint_epochs']):
        for data in batches(epoch):
            for name, head, opt in zip(('endpoint_a', 'endpoint_b'), endpoints, opts):
                ordinary_step(head, opt, data, name)
        print(json.dumps(dict(stage='endpoints', epoch=epoch+1)), flush=True)
    left, right = [copy.deepcopy(h.state_dict()) for h in endpoints]
    control = copy.deepcopy(endpoints[0])
    average = {k: (left[k]+right[k])/2 for k in left}
    control.load_state_dict(average)
    ctrl_opt = torch.optim.Adam(control.parameters(), lr=RECIPE['lr'])
    # Only the control point is learned, with stratified interior samples.
    for epoch in range(RECIPE['connection_epochs']):
        rng = np.random.default_rng(RECIPE['seed']+1000+epoch)
        for data in batches(RECIPE['endpoint_epochs']+epoch):
            budget.check(); ctrl_opt.zero_grad(set_to_none=True)
            losses = []
            for sample in range(RECIPE['path_samples']):
                t = (sample+rng.random())/RECIPE['path_samples']
                params = quadratic(left, dict(control.named_parameters()), right, t)
                losses.extend(objective(functional_call(control, params, d['features']), d) for d in data)
            sum(losses).div(len(losses)).backward()
            torch.nn.utils.clip_grad_norm_(control.parameters(), 5., error_if_nonfinite=True)
            ctrl_opt.step(); counts['connection'] += len(losses)
        print(json.dumps(dict(stage='connection', epoch=epoch+1)), flush=True)
    for head, fixed in zip(endpoints, (left, right)):
        for k, v in head.state_dict().items():
            torch.testing.assert_close(v, fixed[k], rtol=0, atol=0)
    # Equal compute means the same number of per-recording objective evaluations
    # as BOTH endpoints plus all connection samples, not just extra updates.
    ordinary = make_head(plan['feature_width'], len(plan['tokens'])+1, RECIPE['seed'])
    opt = torch.optim.Adam(ordinary.parameters(), lr=RECIPE['lr'])
    ordinary_epochs = 2*RECIPE['endpoint_epochs']+RECIPE['path_samples']*RECIPE['connection_epochs']
    for epoch in range(ordinary_epochs):
        for data in batches(epoch):
            ordinary_step(ordinary, opt, data, 'ordinary')
    assert counts['ordinary'] == counts['endpoint_a']+counts['endpoint_b']+counts['connection']
    curve = {t: quadratic(left, dict(control.named_parameters()), right, t) for t in RECIPE['interior']}
    losses = {t: evaluate(control, p) for t, p in curve.items()}
    selected = min(losses, key=lambda t: (losses[t], abs(t-.5), t))
    curve_losses = {t: evaluate(control, quadratic(left, dict(control.named_parameters()), right, t))
                    for t in (0., .125, .25, .375, .5, .625, .75, .875, 1.)}
    chord_losses = {t: evaluate(control, {k: (1-t)*left[k]+t*right[k] for k in left}) for t in RECIPE['interior']}
    states = dict(endpoint_a=left, endpoint_b=right, average=average, ordinary=ordinary.state_dict(), connected=curve[selected])
    heads = {}
    for name, params in states.items():
        heads[name] = {k: v.detach().tolist() for k, v in params.items()}
    write_json(out/'readouts.json', dict(schema=1, inventory=plan['inventory'], tokens=plan['tokens'],
        primarySha256=plan['primary_sha256'], featureOutput=plan['feature_output'], width=plan['feature_width'],
        hidden=RECIPE['width'], center=center.tolist(), scale=scale.tolist(), heads=heads))
    budget.check()
    write_json(out/'selection.json', dict(schema=1, plan_sha256=digest(out/'plan.json'),
        readouts_sha256=digest(out/'readouts.json'), selected_t=selected, interior_validation=losses,
        curve_validation=curve_losses, chord_validation=chord_losses,
        control_validation={name: evaluate(control, params) for name, params in states.items()},
        objective_evaluations=counts, endpoints_fixed=True,
        selection='equal-weight whole training speakers; complete transcript CTC and output-preservation KL'))
    budget.check()


def export(model, out):
    budget = CpuBudget(out)
    budget.check()
    plan = read_plan(out)
    selection = json.loads((out/'selection.json').read_text())
    if selection['plan_sha256'] != digest(out/'plan.json') or selection['readouts_sha256'] != digest(out/'readouts.json'):
        raise ValueError('Frozen model selection changed')
    manifest = json.loads((model/'manifest.json').read_text())
    if digest(model/manifest['file']) != manifest['sha256'] or manifest['sha256'] != plan['primary_sha256']:
        raise ValueError('Primary differs from frozen encoder')
    # Keep the supplied manifest verbatim. Extra side assets have their own descriptor.
    assets = re.findall(r'file:\s*"([^"]+)"\s*,\s*bytes:\s*(\d+)\s*,\s*sha256:\s*"([0-9a-f]{64})"',
                        (ROOT/'packages/sdk/src/registry.ts').read_text())
    bundle = out/'bundle'
    bundle.mkdir(exist_ok=True)
    for file, size, sha in [(manifest['file'], manifest['bytes'], manifest['sha256']), *assets]:
        source = model/file
        if not source.is_file() or source.stat().st_size != int(size) or digest(source) != sha:
            raise ValueError('Missing/corrupt inherited asset: '+file)
        shutil.copyfile(source, bundle/file)
    shutil.copyfile(model/'manifest.json', bundle/'manifest.json')
    side = {}
    for file in ('encoder-features.onnx', 'readouts.json'):
        shutil.copyfile(out/file, bundle/file)
        side[file] = dict(bytes=(bundle/file).stat().st_size, sha256=digest(bundle/file))
    write_json(bundle/'connected-basin-asset.json', dict(schema=1, primarySha256=manifest['sha256'], files=side))
    budget.check()
    print(json.dumps(dict(bundle=str(bundle), selected_t=selection['selected_t'], assets=side)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'fit', 'export'))
    p.add_argument('--model', type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    if args.command != 'fit' and args.model is None:
        p.error('prepare/export require --model')
    if args.command == 'fit': fit(args.out)
    elif args.command == 'prepare': prepare(args.model, args.out)
    else: export(args.model, args.out)

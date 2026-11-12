"""Matched recording-level confusion-balanced CTC head refitting.

Only prepare reads authorized training audio; fit selects grouping and epoch
using reserved training speakers. Export retains the ordinary deployed head.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path

from student_head_refit import (ROOT, authorized_records, clean_target, digest,
    export, feature_runtime, stable, stream_features, training_helpers, write_json)
from student_confusion_batches import (batch_plan,
    mine_confusion_pairs, recording_descriptors, verify_matched_plans)

EXPERIMENT = 'listener-confusion-balanced-minibatches'
RECIPE = dict(seed=831, original_limit=512, views=3, batch=8,
    lr=.001, momentum=.9, anchor=2., max_seconds=30., epochs=12,
    pair_limits=[4, 8], budget_seconds=23*3600)


def speaker_partitions(records):
    mapping = {}
    for source in ('original', 'help'):
        speakers = sorted({r['speaker'] for r in records if r['source'] == source}, key=stable)
        if len(speakers) < 3:
            raise ValueError('At least three authorized training speakers required per source')
        reserved = max(1, len(speakers)//5)
        for i, speaker in enumerate(speakers):
            mapping[speaker] = 'validation' if i < reserved else 'train'
    return mapping


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
    import time
    began = time.monotonic()
    records, excluded, labels_sha = authorized_records(RECIPE['original_limit'], RECIPE['max_seconds'])
    partitions = speaker_partitions(records)
    for r in records:
        r['partition'] = partitions[r['speaker']]
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
    if any(not r['target'] for r in records):
        raise ValueError('Authorized supervised take has no transcript')
    seen = {}
    seen_partition = {}
    for r in records:
        sha = hashlib.sha256(np.ascontiguousarray(waves[r['key']]).tobytes()).hexdigest()
        if sha in seen and seen[sha] != r['speaker']:
            raise ValueError('Identical performance has conflicting speaker identities')
        if sha in seen_partition and seen_partition[sha] != r['partition']:
            raise ValueError('Identical waveform leaks across speaker partitions')
        seen_partition[sha] = r['partition']
        seen[sha] = r['speaker']
        r['waveform_sha256'] = sha
    helpers = training_helpers()
    aug = {k: np.load(Path('/student/aug')/(k+'.npz')) for k in ('noise', 'rir')}
    cache = out/'features'
    cache.mkdir(exist_ok=True)
    views = []
    torch.set_num_threads(2)
    for r in records:
        if time.monotonic()-began > RECIPE['budget_seconds']:
            raise TimeoutError('Training preparation exhausted the one-day budget')
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
                partition=r['partition'], supervised=True, view=view))
        print(json.dumps(dict(cached=len(views), recordings=len(records))), flush=True)
    write_json(out/'plan.json', dict(schema=1, id=EXPERIMENT, recipe=RECIPE, records=records, views=views,
        excluded_speakers=excluded, labels_sha256=labels_sha, primary_sha256=manifest['sha256'],
        preparation_seconds=time.monotonic()-began, initial_head_sha256=digest(out/'initial-head.npz'), feature_graph_sha256=digest(out/'encoder-features.onnx')))


def read_plan(out):
    plan = json.loads((out/'plan.json').read_text())
    if plan['schema'] != 1 or plan['recipe'] != RECIPE or plan['id'] != EXPERIMENT:
        raise ValueError('Frozen experiment or recipe changed')
    for file, sha in [('initial-head.npz', plan['initial_head_sha256']),
                      ('encoder-features.onnx', plan['feature_graph_sha256'])]:
        if digest(out/file) != sha:
            raise ValueError('Frozen head or feature graph changed')
    records = {r['key']: r for r in plan['records']}
    if not records or len(records) != len(plan['records']):
        raise ValueError('Frozen recording multiset is empty or duplicated')
    partitions = speaker_partitions(plan['records'])
    signatures = set()
    for view in plan['views']:
        record = records.get(view['recording'])
        signature = (view['recording'], view['view'])
        if (record is None or signature in signatures or
                view['view'] not in range(RECIPE['views']) or not view['supervised']):
            raise ValueError('Frozen recording views are incomplete or duplicated')
        signatures.add(signature)
        if (view['speaker'] != record['speaker'] or view['partition'] != record['partition'] or
                partitions[view['speaker']] != view['partition'] or
                view['speaker'] in plan['excluded_speakers']):
            raise ValueError('Whole-speaker separation violated')
        if not Path(view['file']).resolve().is_relative_to((out/'features').resolve()):
            raise ValueError('Feature cache escapes the frozen output directory')
        if digest(view['file']) != view['sha256']:
            raise ValueError('Frozen feature cache changed')
    if len(signatures) != len(records)*RECIPE['views']:
        raise ValueError('Frozen recording views are incomplete or duplicated')
    return plan


def objective(head, data):
    """Ordinary full-inventory sequence likelihood with shipped-output anchor."""
    import torch
    import torch.nn.functional as F

    lp = head(data['features']).log_softmax(-1)
    if not len(data['target']):
        raise ValueError('Only authorized complete transcripts enter the refit')
    loss = F.ctc_loss(lp[:, None], data['target'], torch.tensor([len(lp)]),
        torch.tensor([len(data['target'])]), reduction='sum', zero_infinity=False)/len(lp)
    loss += RECIPE['anchor']*F.kl_div(lp, data['baseline'].exp(), reduction='sum')/len(lp)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite ordinary CTC/preservation loss')
    return loss


def train_update(head, optimizer, records, batch_size):
    """Accumulate simultaneous recording gradients, then make one SGD update.

    Sequence computation per recording avoids padding-dependent features and
    excessive memory. Dividing by the fixed effective batch size even for a
    final partial batch preserves each recording's objective weight in all arms.
    """
    import torch

    optimizer.zero_grad(set_to_none=True)
    for data in records:
        (objective(head, data)/batch_size).backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
        raise ValueError('Invalid head-only gradient')
    optimizer.step()


def selected_plan(out):
    """Export only a complete training-selected, matched three-head result."""
    plan = read_plan(out)
    selection = json.loads((out/'selection.json').read_text())
    names = {'shuffled', 'length'} | {f'balanced-{n}' for n in RECIPE['pair_limits']}
    training = [v for v in plan['views'] if v['partition'] == 'train']
    recordings = len({v['recording'] for v in training})
    per_epoch = math.ceil(recordings/RECIPE['batch'])*RECIPE['views']
    if (selection['schema'] != 1 or set(selection['arms']) != {'balanced', 'shuffled', 'length'} or
            selection['pair_limit'] not in RECIPE['pair_limits'] or
            not 1 <= selection['epoch'] <= RECIPE['epochs'] or
            len(selection['history']) != RECIPE['epochs'] or
            selection['final_exposures_per_arm'] != len(training)*RECIPE['epochs'] or
            selection['updates'] != dict.fromkeys(names, per_epoch*selection['epoch']) or
            selection['final_updates'] != dict.fromkeys(names, per_epoch*RECIPE['epochs'])):
        raise ValueError('Selected result is incomplete or control exposure differs')
    choices = []
    for epoch, row in enumerate(selection['history'], 1):
        if (row['epoch'] != epoch or set(row['validation']) != names or
                not all(math.isfinite(value) for value in row['validation'].values()) or
                row['updates'] != dict.fromkeys(names, per_epoch*epoch)):
            raise ValueError('Training selection history is incomplete')
        for limit in RECIPE['pair_limits']:
            choices.append((row['validation'][f'balanced-{limit}'], epoch, limit))
    _, epoch, limit = min(choices)
    if (selection['epoch'] != epoch or selection['pair_limit'] != limit or
            selection['validation'] != selection['history'][epoch-1]['validation'] or
            selection['fit_seconds']+plan['preparation_seconds'] > RECIPE['budget_seconds']):
        raise ValueError('Checkpoint was not selected within the frozen training budget')
    for stem, suffix in [('grouping', '.json'), ('batches', '.jsonl')]:
        if digest(out/(stem+suffix)) != selection[stem+'_sha256']:
            raise ValueError('Frozen grouping or exposure audit changed')
    grouping = json.loads((out/'grouping.json').read_text())
    if (grouping['plan_sha256'] != digest(out/'plan.json') or
            grouping['mining_speakers'] != sorted({v['speaker'] for v in training}) or
            grouping['selection_speakers'] != sorted({v['speaker'] for v in plan['views']
                                                      if v['partition'] == 'validation'})):
        raise ValueError('Grouping or selection speaker provenance changed')
    return plan


def fit(out):
    import time
    import numpy as np
    import torch
    from student import TOKENS
    from student_confusion_batches import batch_quality

    began = time.monotonic()
    if any((out/file).exists() for file in ('selection.json', 'batches.jsonl', 'grouping.json')):
        raise ValueError('Refit already started; use a fresh prepared output directory')
    torch.set_num_threads(2)
    torch.manual_seed(RECIPE['seed'])
    torch.use_deterministic_algorithms(True)
    plan = read_plan(out)
    views = plan['views']
    training = [v for v in views if v['partition'] == 'train']
    validation = [v for v in views if v['partition'] == 'validation']
    if not training or not validation or any(not v['supervised'] for v in views):
        raise ValueError('Fit and independent-speaker transcript banks are required')

    @lru_cache(maxsize=32)
    def arrays(path):
        with np.load(path) as data:
            return {k: data[k].copy() for k in ('features', 'baseline', 'target')}

    def tensors(path):
        return {k: torch.from_numpy(v) for k, v in arrays(path).items()}

    def budget():
        if time.monotonic()-began+plan.get('preparation_seconds', 0.) > RECIPE['budget_seconds']:
            raise TimeoutError('The frozen training-only one-day budget is exhausted')

    grouped = {}
    for v in training:
        budget()
        data = arrays(v['file'])
        grouped.setdefault(v['recording'], []).append((data['baseline'], data['target']))
    descriptors = {}
    for key, rows in grouped.items():
        budget()
        descriptors.update(recording_descriptors({key: rows}))

    def decodes():
        for rows in grouped.values():
            for row in rows:
                budget()
                yield row
    # The supplied inventory puts the ordinary Arabic consonants before vowels
    # and duration symbols. No practice-specific phone list enters mining.
    consonants = list(range(1, TOKENS.index('ا')+1))
    pairs = mine_confusion_pairs(decodes(), consonants, max(RECIPE['pair_limits']))
    budget()
    if not pairs:
        raise ValueError('Training decodes contain no supported two-way consonant confusion')
    write_json(out/'grouping.json', dict(plan_sha256=digest(out/'plan.json'), pairs=pairs,
        descriptors={k: v.tolist() for k, v in descriptors.items()},
        mining_speakers=sorted({v['speaker'] for v in training}),
        selection_speakers=sorted({v['speaker'] for v in validation})))

    initial = np.load(out/'initial-head.npz')
    template = torch.nn.Linear(initial['weight'].shape[1], initial['weight'].shape[0])
    with torch.no_grad():
        template.weight.copy_(torch.from_numpy(initial['weight']))
        template.bias.copy_(torch.from_numpy(initial['bias']))
    import copy
    names = ['shuffled', 'length']+[f'balanced-{n}' for n in RECIPE['pair_limits']]
    heads = {name: copy.deepcopy(template) for name in names}
    optimizers = {name: torch.optim.SGD(h.parameters(), lr=RECIPE['lr'], momentum=RECIPE['momentum'])
                  for name, h in heads.items()}

    def evaluate(h):
        by_speaker = {}
        with torch.no_grad():
            for v in validation:
                budget()
                by_speaker.setdefault(v['speaker'], []).append(float(objective(h, tensors(v['file']))))
        return float(np.mean([np.mean(values) for values in by_speaker.values()]))

    lookup = {(v['recording'], v['view']): v for v in training}
    keys = sorted(grouped)
    if len(lookup) != len(keys)*RECIPE['views']:
        raise ValueError('Every fit recording must have exactly the same cached view count')
    lengths = {key: len(arrays(lookup[key, 0]['file'])['features']) for key in keys}
    history, exposures = [], {name: Counter() for name in names}
    updates = {name: 0 for name in names}
    best = None
    for epoch in range(1, RECIPE['epochs']+1):
        for view in range(RECIPE['views']):
            seed = RECIPE['seed']+epoch*RECIPE['views']+view
            schedules = {}
            for name in names:
                mode = name.split('-')[0] if name.startswith('balanced-') else name
                subset = [p['phones'] for p in pairs[:int(name.split('-')[1])]] if mode == 'balanced' else []
                schedules[name] = batch_plan(keys, lengths, descriptors, subset, RECIPE['batch'], seed, mode)
            verify_matched_plans(keys, schedules, RECIPE['batch'])
            # Saving the exact schedules makes recording exposure reviewable.
            with (out/'batches.jsonl').open('a') as log:
                log.write(json.dumps(dict(epoch=epoch, view=view, schedules=schedules))+'\n')
            for name, batches in schedules.items():
                for batch in batches:
                    budget()
                    train_update(heads[name], optimizers[name],
                                 [tensors(lookup[key, view]['file']) for key in batch], RECIPE['batch'])
                    exposures[name].update((key, view) for key in batch)
                    updates[name] += 1
            if len({tuple(sorted(c.items())) for c in exposures.values()}) != 1 or len(set(updates.values())) != 1:
                raise ValueError('Matched recording exposure or optimizer update counts diverged')
        losses = {name: evaluate(head) for name, head in heads.items()}
        candidate = min((name for name in names if name.startswith('balanced-')),
                        key=lambda name: (losses[name], int(name.split('-')[1])))
        quality = {name: batch_quality(schedules[name], descriptors, [p['phones'] for p in pairs]) for name in names}
        row = dict(epoch=epoch, validation=losses, updates=updates.copy(), grouping=quality)
        history.append(row)
        print(json.dumps(row), flush=True)
        choice = (losses[candidate], epoch, int(candidate.split('-')[1]))
        if best is None or choice < best:
            best = choice
            # Both matched controls are saved at this exact exposure/epoch,
            # rather than chosen later for a favorable practice comparison.
            checkpoints = {}
            for arm, name in [('balanced', candidate), ('shuffled', 'shuffled'), ('length', 'length')]:
                path = out/(arm+'.npz')
                h = heads[name]
                np.savez(path, weight=h.weight.detach().numpy(), bias=h.bias.detach().numpy())
                checkpoints[arm] = dict(path=str(path), sha256=digest(path))
            chosen = dict(epoch=epoch, pair_limit=best[2], validation=losses,
                          updates=updates.copy(), arms=checkpoints)
    expected = len(keys)*RECIPE['views']*RECIPE['epochs']
    if any(sum(count.values()) != expected for count in exposures.values()):
        raise ValueError('Total recording exposure is incomplete')
    budget()
    write_json(out/'selection.json', dict(schema=1, plan_sha256=digest(out/'plan.json'),
        grouping_sha256=digest(out/'grouping.json'), batches_sha256=digest(out/'batches.jsonl'),
        **chosen, final_updates=updates, final_exposures_per_arm=expected, history=history,
        fit_seconds=time.monotonic()-began,
        optimizer=dict(name='SGD', lr=RECIPE['lr'], momentum=RECIPE['momentum']),
        selection='minimum speaker-averaged ordinary CTC plus preservation on reserved training speakers; matched control epoch'))


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
        export(args.model, args.out, selected_plan, RECIPE['seed'])

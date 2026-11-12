"""Matched frozen-head refits with exact negative-transcription CTC mass.

Mine first with the SDK's training/mine-sequence-cli.ts on help-train only.
Then prepare, fit and export under the authorized supervisor. Neither training
nor checkpoint selection reads practice audio or incident keys. Mistake scripts
are output-preservation controls, never assumed performed transcripts.
"""
import argparse
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from student_head_refit import (ROOT, clean_target, digest, export, feature_runtime,
                               reserved_speakers, stream_features, write_json)

# Both arms share this preregistered recipe; no practice-driven search.
RECIPE = dict(seed=831, epochs=12, batch=8, lr=.0001, anchor=2., negative_weight=1.)


def sequence_log_probability(log_probs, target, blank=0):
    """All complete CTC paths for one specific transcript, not phone penalties."""
    import torch
    import torch.nn.functional as F

    if log_probs.ndim != 2 or not len(log_probs) or not torch.isfinite(log_probs).all():
        raise ValueError('Invalid acoustic rows')
    target = torch.as_tensor(target, dtype=torch.int64, device=log_probs.device)
    if target.ndim != 1 or ((target < 0) | (target >= log_probs.shape[1]) | (target == blank)).any():
        raise ValueError('Invalid complete transcript')
    required = len(target) + int((target[1:] == target[:-1]).sum())
    if required > len(log_probs):
        # Empty path set has zero probability and zero negative gradient.
        return log_probs.sum()*0 + float('-inf')
    return -F.ctc_loss(log_probs[:, None, :], target,
        torch.tensor([len(log_probs)]), torch.tensor([len(target)]),
        blank=blank, reduction='sum', zero_infinity=False)


def sequence_unlikelihood(log_probs, targets):
    """-log(1-P(transcript)), summed over unique erroneous transcriptions.

    Preserve repeated articulations in negatives. Duplicating an accusation must
    not duplicate its penalty. Stable log1mexp avoids cancellation near zero.
    """
    import torch

    total = log_probs.sum()*0
    for target in sorted({tuple(t) for t in targets}):
        z = sequence_log_probability(log_probs, target)
        if z.detach() > 0:
            raise ValueError('CTC transcript probability exceeds one')
        if z.detach() < -math.log(2):
            total = total-torch.log1p(-torch.exp(z))
        else:
            total = total-torch.log(-torch.expm1(z))
    return total


def supervised(row):
    return row['kind'] == 'none' and row.get('scored') is not False and not any(
        row.get(k) for k in ('events', 'non_errors', 'unsupported'))


def read_mining(mining):
    """Recheck source labels and whole-speaker exclusion before feature fitting."""
    import os

    mined = json.loads(mining.read_text())
    source = (Path(os.environ['LAB_DATA'])/'help-train').resolve()
    if Path(mined['labels_root']).resolve() != source:
        raise ValueError('Mining source is outside authorized help-train')
    labels = source/'labels.jsonl'
    if digest(labels) != mined['labels_sha256']:
        raise ValueError('Immutable source labels differ from the mining inventory')
    if digest(ROOT/'packages/core/assets/ref-v2.json') != mined['reference_sha256']:
        raise ValueError('Lawful-pronunciation reference changed after mining')
    rows = [json.loads(s) for s in labels.read_text().splitlines() if s]
    if len({r['id'] for r in rows}) != len(rows) or any(not r['speaker'] for r in rows):
        raise ValueError('Ambiguous training provenance')
    excluded = {r['speaker'] for r in rows if r['split'] != 'train' or r.get('exclude') or r.get('excluded')}
    if set(mined['excluded_speakers']) != excluded:
        raise ValueError('Excluded-speaker audit differs from source labels')
    admitted = {r['id']: r for r in rows if r['speaker'] not in excluded}
    records = mined['records']
    if len(records) != len(admitted) or {r['id'] for r in records} != set(admitted):
        raise ValueError('Mining must retain all admitted takes, including preservation takes')
    reserved = reserved_speakers(list(admitted.values()))
    if set(mined['reserved']) != reserved:
        raise ValueError('Speaker reservation differs from the frozen rule')
    for r in records:
        original = admitted[r['id']]
        if any(r.get(k) != v for k, v in original.items()):
            raise ValueError('Mining changed source-label provenance')
        file = (source/original['wav']).resolve()
        if not file.is_relative_to(source) or Path(r['file']).resolve() != file:
            raise ValueError('Recording escapes authorized training root')
        if digest(file) != r['source_sha256'] or digest(r['feature_file']) != r['feature_sha256']:
            raise ValueError('Recording or frontend features changed after mining')
        if r['supervised'] != supervised(original) or r['partition'] != (
                'validation' if r['speaker'] in reserved else 'train'):
            raise ValueError('Invalid canonical supervision or speaker partition')
        if r['negatives'] and not r['supervised']:
            raise ValueError('Mistake and repeat takes cannot supply negative transcripts')
    return mined


def prepare(model, out, mining):
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise ValueError('Training plan already frozen; use a new output directory')
    mined = read_mining(mining)
    records = mined['records']
    train = [r for r in records if r['partition'] == 'train']
    if not any(r['negatives'] for r in train):
        raise ValueError('Premise absent: no eligible fit-speaker false sequences with positive coverage')
    reserved = set(mined['reserved'])
    if any(r['split'] != 'train' or r.get('exclude') or r.get('excluded') or
           (r['speaker'] in reserved) != (r['partition'] == 'validation') for r in records):
        raise ValueError('Mining violated training-speaker partitions')
    manifest, head, runtime = feature_runtime(model, out, RECIPE['seed'])
    if manifest['sha256'] != mined['primary_sha256'] or manifest['blank'] != 0:
        raise ValueError('Mining and frozen feature encoder differ')
    if (manifest['sampleRate'] != 16000 or manifest['frameMs'] != 40 or
            manifest['fbank']['mels'] != 80 or manifest['shift'] % 4 or
            manifest['window'] != manifest['shift']+13):
        raise ValueError('Model differs from the deployed streaming frontend geometry')
    np.savez(out/'initial-head.npz', weight=head['weight'].T, bias=head['bias'])
    ref = json.loads((ROOT/'packages/core/assets/ref-v2.json').read_text())
    ayahs = {f"{a['s']}:{a['a']}": a['p'] for a in ref['ayahs']}
    ids = {p: i+1 for i, p in enumerate(manifest['tokens'])}
    cache = out/'features'
    cache.mkdir(exist_ok=True)
    views, seen = [], {}
    for r in records:
        feats = np.fromfile(r['feature_file'], dtype='<f4').reshape(-1, 80)
        if not np.isfinite(feats).all():
            raise ValueError('Invalid deployed frontend features')
        sha = r['feature_sha256']
        if sha in seen and seen[sha] != r['speaker']:
            raise ValueError('Identical recording crosses speaker identities')
        seen[sha] = r['speaker']
        x, lp = stream_features(runtime, manifest, feats, head['feature'])
        logits = x@head['weight']+head['bias']
        reconstructed = logits-np.logaddexp.reduce(logits, axis=1, keepdims=True)
        if not np.allclose(reconstructed, lp, atol=.002, rtol=0):
            raise ValueError('Cached features do not reproduce the shipped phone head')
        target = clean_target(''.join(ayahs[a] for a in r['ayahs']), manifest['tokens']) if r['supervised'] else []
        negatives, unique = [], set()
        for n in r['negatives']:
            if not (0 <= n['from'] < n['to'] <= len(x)) or not n['positives']:
                raise ValueError('Incomplete negative acoustic witness or positive coverage')
            if any(not any(p['id'] == key and p['supervised'] and p['partition'] == 'train'
                           for p in records) for key in n['positives']):
                raise ValueError('Negative positive-coverage source is not a fit-speaker clean take')
            key = (n['from'], n['to'], n['heard'])
            if key not in unique:
                negatives.append(dict(from_frame=n['from'], to_frame=n['to'], target=[ids[p] for p in n['heard']]))
                unique.add(key)
        file = cache/(hashlib.sha256(r['id'].encode()).hexdigest()+'.npz')
        np.savez_compressed(file, features=x, baseline=lp, target=np.asarray(target, dtype='int64'))
        views.append(dict(recording=r['id'], speaker=r['speaker'], partition=r['partition'],
            supervised=r['supervised'], file=str(file), sha256=digest(file), negatives=negatives,
            frontend_sha256=sha))
        print(json.dumps(dict(cached=r['id'], negatives=len(negatives))), flush=True)
    write_json(out/'plan.json', dict(schema=1, recipe=RECIPE, views=views, excluded_speakers=mined['excluded_speakers'],
        primary_sha256=manifest['sha256'], mining_sha256=digest(mining), mining_path=str(mining),
        initial_head_sha256=digest(out/'initial-head.npz'),
        feature_graph_sha256=digest(out/'encoder-features.onnx')))


def read_plan(out):
    plan = json.loads((out/'plan.json').read_text())
    if plan['recipe'] != RECIPE:
        raise ValueError('Frozen training recipe changed')
    for file, sha in [(out/'initial-head.npz', plan['initial_head_sha256']),
                      (out/'encoder-features.onnx', plan['feature_graph_sha256']),
                      (Path(plan['mining_path']), plan['mining_sha256'])]:
        if digest(file) != sha:
            raise ValueError('Frozen training provenance changed')
    speakers = {k: {v['speaker'] for v in plan['views'] if v['partition'] == k}
                for k in ('train', 'validation')}
    if speakers['train'] & speakers['validation'] or not all(speakers.values()):
        raise ValueError('Speaker partition overlap or empty partition')
    for v in plan['views']:
        if digest(v['file']) != v['sha256']:
            raise ValueError('Frozen acoustic features changed')
    return plan


def fit(out):
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(2)
    torch.manual_seed(RECIPE['seed'])
    plan = read_plan(out)
    initial = np.load(out/'initial-head.npz')
    head = torch.nn.Linear(initial['weight'].shape[1], initial['weight'].shape[0])
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(initial['weight']))
        head.bias.copy_(torch.from_numpy(initial['bias']))
    arms = {'positive': head, 'unlikelihood': copy.deepcopy(head)}
    train = [v for v in plan['views'] if v['partition'] == 'train']
    validation = [v for v in plan['views'] if v['partition'] == 'validation']
    if not any(v['supervised'] for v in validation) or not any(not v['supervised'] for v in train):
        raise ValueError('Missing reserved transcript or mistake preservation takes')

    @lru_cache(maxsize=32)
    def cached(file):
        with np.load(file) as d:
            return {k: torch.from_numpy(d[k].copy()) for k in ('features', 'baseline', 'target')}

    def objective(h, v, negative):
        d = cached(v['file'])
        lp = h(d['features']).log_softmax(-1)
        loss = RECIPE['anchor']*F.kl_div(lp, d['baseline'].exp(), reduction='sum')/len(lp)
        if v['supervised']:
            loss = loss-sequence_log_probability(lp, d['target'])/len(lp)
        if negative:
            by_window = {}
            for n in v['negatives']:
                by_window.setdefault((n['from_frame'], n['to_frame']), []).append(n['target'])
            for (a, b), targets in by_window.items():
                loss = loss+RECIPE['negative_weight']*sequence_unlikelihood(
                    lp[a:b], targets)
        return loss

    def evaluate(h):
        # Same objective for both arms, speaker-balanced and training-only.
        by_speaker = {}
        with torch.no_grad():
            for v in validation:
                by_speaker.setdefault(v['speaker'], []).append(float(objective(h, v, True)))
        score = float(np.mean([np.mean(values) for values in by_speaker.values()]))
        if not math.isfinite(score):
            raise ValueError('Invalid reserved-speaker objective')
        return score

    with torch.no_grad():
        masses = []
        for v in train:
            d = cached(v['file'])
            lp = head(d['features']).log_softmax(-1)
            for n in v['negatives']:
                masses.append(float(sequence_log_probability(lp[n['from_frame']:n['to_frame']], n['target']).exp()))
    negative_audit = dict(count=len(masses), maximum_probability=max(masses, default=0.),
                          mean_probability=float(np.mean(masses)) if masses else 0.)
    print(json.dumps(dict(initial_negative_mass=negative_audit)), flush=True)

    history, updates = [], 0
    for epoch in range(RECIPE['epochs']):
        order = np.random.default_rng(RECIPE['seed']+epoch).permutation(len(train))
        for at in range(0, len(order), RECIPE['batch']):
            batch = [train[i] for i in order[at:at+RECIPE['batch']]]
            for name, h in arms.items():
                h.zero_grad(set_to_none=True)
                loss = sum(objective(h, v, name == 'unlikelihood') for v in batch)/len(batch)
                loss.backward()
                if not math.isfinite(float(loss)) or any(not torch.isfinite(p.grad).all() for p in h.parameters()):
                    raise ValueError('Invalid phone-head gradient')
                with torch.no_grad():
                    for p in h.parameters():
                        p.add_(p.grad, alpha=-RECIPE['lr'])
            updates += 1
        scores = {name: evaluate(h) for name, h in arms.items()}
        for name, h in arms.items():
            np.savez(out/f'{name}-{epoch+1}.npz', weight=h.weight.detach().numpy(), bias=h.bias.detach().numpy())
        history.append(dict(epoch=epoch+1, updates=updates, validation=scores))
        print(json.dumps(history[-1]), flush=True)
    selected = min(history, key=lambda v: (v['validation']['unlikelihood'], v['epoch']))
    paths = {name: out/f"{name}-{selected['epoch']}.npz" for name in arms}
    write_json(out/'selection.json', dict(schema=1, plan_sha256=digest(out/'plan.json'),
        selected=selected, history=history, identical_updates=True, identical_recordings=True,
        initial_negative_mass=negative_audit,
        checkpoint_selection='reserved training-speaker mean transcript CTC plus unlikelihood and preservation',
        arms={name: dict(path=str(path), sha256=digest(path)) for name, path in paths.items()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'fit', 'export'))
    p.add_argument('--model', type=Path)
    p.add_argument('--mining', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.command != 'fit' and args.model is None:
        p.error('prepare/export require --model')
    if args.command == 'prepare':
        if args.mining is None:
            p.error('prepare requires --mining')
        prepare(args.model, args.out, args.mining)
    elif args.command == 'fit':
        fit(args.out)
    else:
        export(args.model, args.out, read_plan, RECIPE['seed'])

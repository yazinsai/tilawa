"""Cross-speaker bilevel weighting of a frozen streaming listener's phone head.

Run prepare, fit, then export in the supervisor training image. Training reads
authorized original recordings and help-train only. Both arms use the same
support batches, augmentation cache, preservation bank and optimizer updates.
The query speakers are disjoint from support; reserved training speakers select
one weighted checkpoint and its uniform control at the identical update count.
No practice input is read by this utility.
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
                               export, feature_runtime, has_clean_transcript,
                               stable, stream_features, training_helpers, write_json)

# Frozen before any incident diagnostic. The step size applies to both arms.
RECIPE = dict(seed=831, original_limit=512, views=3, epochs=12, batch=8,
              query_batch=4, preservation_batch=4, lr=.0001, anchor=2., max_seconds=30.)


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
    from student import TOKENS, phoneme_ids, spliced_target, mix_noise

    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise ValueError('Training plan already frozen; use a new output directory')
    records, excluded, labels_sha = authorized_records(RECIPE['original_limit'], RECIPE['max_seconds'])
    # Reserve all takes of a speaker together, including preservation controls.
    validation = {r['speaker'] for r in records if r['partition'] == 'validation'}
    preservation = help_preservation(RECIPE['max_seconds'])
    for r in preservation:
        r['partition'] = 'validation' if r['speaker'] in validation else 'train'
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
        table = pq.read_table(path, columns=['flac', 'ref', 'hyp']).take([r['index'] for r in selected]).to_pylist()
        for r, value in zip(selected, table):
            wave, sr = sf.read(io.BytesIO(value['flac']), dtype='float32')
            if sr != 16000 or wave.ndim != 1:
                raise ValueError('Original prepared audio is not mono 16 kHz')
            waves[r['key']] = wave
            r['target'] = phoneme_ids(spliced_target(value['ref'], value['hyp']))
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
    # Exact waveform copies across speaker partitions would invalidate query
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
            views.append(dict(recording=r['key'], speaker=r['speaker'], partition=r['partition'],
                              supervised=r['source'] != 'preservation', file=str(file), sha256=digest(file)))
        print(json.dumps(dict(cached=r['key'], partition=r['partition'], supervised=r['source'] != 'preservation')), flush=True)
    write_json(out/'plan.json', dict(schema=1, recipe=RECIPE, records=records, views=views,
        excluded_speakers=excluded, help_labels_sha256=labels_sha, primary_sha256=manifest['sha256'],
        feature_graph_sha256=digest(out/'encoder-features.onnx'), initial_head_sha256=digest(out/'initial-head.npz')))


def ctc_loss(parameters, data):
    import torch
    import torch.nn.functional as F

    lp = F.linear(data['features'], *parameters).log_softmax(-1)
    return F.ctc_loss(lp[:, None, :], data['target'], torch.tensor([len(lp)]),
                      torch.tensor([len(data['target'])]), reduction='sum', zero_infinity=False)/len(lp)


def preservation_loss(parameters, data):
    import torch.nn.functional as F

    lp = F.linear(data['features'], *parameters).log_softmax(-1)
    return F.kl_div(lp, data['baseline'].exp(), reduction='sum')/len(lp)


def transfer_weights(parameters, support, query, anchor_data, lr):
    """Exact derivative of query NLL through a one-step epsilon-weighted update.

    theta'(epsilon) = theta - lr * (anchor_gradient + sum epsilon_i g_i).
    At epsilon=0, -d(query NLL)/d(epsilon_i) = lr * <g_query, g_i>.
    Compute that chain rule explicitly: PyTorch CTC lacks double backward.
    Positive derivatives receive normalized mass; wholly harmful batches receive
    zero transcript mass, retaining only the same ordinary preservation update.
    Neither support self-fit nor practice outcomes supply the example weight.
    """
    import torch

    params = tuple(parameters)
    anchor = RECIPE['anchor']*sum(preservation_loss(params, d) for d in anchor_data)/len(anchor_data)
    ga = torch.autograd.grad(anchor, params)
    virtual = tuple((p-lr*g).detach().requires_grad_() for p, g in zip(params, ga))
    q = sum(ctc_loss(virtual, d) for d in query)/len(query)
    gq = torch.autograd.grad(q, virtual)
    benefits = []
    for d in support:
        gs = torch.autograd.grad(ctc_loss(params, d), params)
        benefits.append(lr*sum((a*b).sum() for a, b in zip(gs, gq)))
    benefits = torch.stack(benefits).detach()
    if not torch.isfinite(benefits).all():
        raise ValueError('Non-finite cross-speaker meta gradient')
    mass = benefits.clamp_min(0)
    return (mass/mass.sum() if mass.sum() > 0 else mass), benefits


def episodes(views, epoch, batch, query_batch, seed):
    """Every view supports once per epoch; query roles rotate by whole speaker.

    The query sample uses distinct speakers, one item per speaker, so prolific
    speakers cannot determine the transfer test just by recording more clips.
    """
    import numpy as np

    by_speaker = {}
    for i, v in enumerate(views):
        by_speaker.setdefault(v['speaker'], []).append(i)
    speakers = sorted(by_speaker, key=stable)
    if len(speakers) < 4:
        raise ValueError('At least four supervised training speakers are required')
    rng = np.random.default_rng(seed+epoch)
    speakers = [speakers[i] for i in rng.permutation(len(speakers))]
    halves = [speakers[::2], speakers[1::2]]
    count = 0
    for side in (epoch % 2, 1-epoch % 2):
        support_speakers, query_speakers = halves[side], halves[1-side]
        order = np.asarray([i for s in support_speakers for i in by_speaker[s]])
        order = order[rng.permutation(len(order))]
        for start in range(0, len(order), batch):
            support = order[start:start+batch].tolist()
            qs = [query_speakers[(count+j) % len(query_speakers)]
                  for j in range(min(query_batch, len(query_speakers)))]
            query = [by_speaker[s][(epoch+count) % len(by_speaker[s])] for s in qs]
            if {views[i]['speaker'] for i in support} & {views[i]['speaker'] for i in query}:
                raise ValueError('Support and query speakers overlap')
            yield support, query
            count += 1


def read_plan(out):
    plan = json.loads((out/'plan.json').read_text())
    if plan['recipe'] != RECIPE:
        raise ValueError('Frozen training settings changed')
    for file, sha in [('initial-head.npz', plan['initial_head_sha256']),
                      ('encoder-features.onnx', plan['feature_graph_sha256'])]:
        if digest(out/file) != sha:
            raise ValueError('Frozen head or encoder feature graph changed')
    for view in plan['views']:
        if digest(view['file']) != view['sha256']:
            raise ValueError('Frozen training features changed')
    train_speakers = {v['speaker'] for v in plan['views'] if v['partition'] == 'train'}
    reserved = {v['speaker'] for v in plan['views'] if v['partition'] == 'validation'}
    if train_speakers & reserved or (train_speakers | reserved) & set(plan['excluded_speakers']):
        raise ValueError('Training-speaker separation violated')
    return plan


def preservation_indices(bank, query_speakers, update, batch):
    """Keep even transcript-free update inputs disjoint from query speakers."""
    eligible = [i for i, v in enumerate(bank) if v['speaker'] not in query_speakers]
    if not eligible:
        raise ValueError('No preservation recordings outside the query speakers')
    return [eligible[(update*batch+j) % len(eligible)] for j in range(batch)]


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
    arms = {'uniform': head, 'weighted': copy.deepcopy(head)}
    train = [v for v in plan['views'] if v['partition'] == 'train' and v['supervised']]
    bank = [v for v in plan['views'] if v['partition'] == 'train' and not v['supervised']]
    validation = [v for v in plan['views'] if v['partition'] == 'validation']
    if not train or not bank or not validation or not any(v['supervised'] for v in validation):
        raise ValueError('Missing supervised, preservation or reserved-speaker input')

    @lru_cache(maxsize=32)
    def cached(path):
        with np.load(path) as a:
            return {k: torch.from_numpy(a[k].copy()) for k in ('features', 'baseline', 'target')}

    def evaluate(h):
        by_speaker = {}
        with torch.no_grad():
            for v in validation:
                d = cached(v['file'])
                loss = RECIPE['anchor']*preservation_loss(tuple(h.parameters()), d)
                if v['supervised']:
                    loss = loss+ctc_loss(tuple(h.parameters()), d)
                by_speaker.setdefault(v['speaker'], []).append(float(loss))
        result = float(np.mean([np.mean(values) for values in by_speaker.values()]))
        if not math.isfinite(result):
            raise ValueError('Invalid reserved-speaker validation loss')
        return result

    history, updates, statistics = [], 0, {}
    baseline = evaluate(head)
    for epoch in range(RECIPE['epochs']):
        positive, total, empty = 0, 0, 0
        weights_log = []
        for support_ids, query_ids in episodes(train, epoch, RECIPE['batch'], RECIPE['query_batch'], RECIPE['seed']):
            support = [cached(train[i]['file']) for i in support_ids]
            query = [cached(train[i]['file']) for i in query_ids]
            query_speakers = {train[i]['speaker'] for i in query_ids}
            protected_ids = preservation_indices(bank, query_speakers, updates, RECIPE['preservation_batch'])
            protected = [cached(bank[i]['file']) for i in protected_ids]
            anchor_data = support+protected
            weights, benefits = transfer_weights(tuple(arms['weighted'].parameters()), support, query,
                                                 anchor_data, RECIPE['lr'])
            positive += int((weights > 0).sum())
            total += len(weights)
            empty += int(weights.sum() == 0)
            weights_log.append(dict(support=[train[i]['recording'] for i in support_ids],
                query=[train[i]['recording'] for i in query_ids],
                preservation=[bank[i]['recording'] for i in protected_ids],
                weights=weights.tolist(), benefits=benefits.tolist()))
            for name, h in arms.items():
                h.zero_grad(set_to_none=True)
                w = weights if name == 'weighted' else torch.full((len(support),), 1/len(support))
                parameters = tuple(h.parameters())
                loss = sum(ctc_loss(parameters, d)*weight for d, weight in zip(support, w))
                loss = loss+RECIPE['anchor']*sum(preservation_loss(parameters, d) for d in anchor_data)/len(anchor_data)
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in h.parameters()):
                    raise ValueError('Invalid phone-head update')
                # Exact first-order SGD has no optimizer state for this refit.
                # Keeping the update explicit also avoids importing graph-
                # compilation machinery for a tiny frozen-feature classifier.
                with torch.no_grad():
                    for p in h.parameters():
                        p.add_(p.grad, alpha=-RECIPE['lr'])
            updates += 1
        losses = {name: evaluate(h) for name, h in arms.items()}
        for name, h in arms.items():
            np.savez(out/f'{name}-{epoch+1}.npz', weight=h.weight.detach().numpy(), bias=h.bias.detach().numpy())
        write_json(out/f'weights-{epoch+1}.json', weights_log)
        statistics = dict(positive=positive, total=total, empty_batches=empty)
        history.append(dict(epoch=epoch+1, updates=updates, validation=losses, weighting=statistics))
        print(json.dumps(history[-1]), flush=True)
    selected = min(history, key=lambda r: (r['validation']['weighted'], r['epoch']))
    paths = {name: out/f"{name}-{selected['epoch']}.npz" for name in arms}
    write_json(out/'selection.json', dict(schema=1, plan_sha256=digest(out/'plan.json'), baseline_validation=baseline,
        selected=selected, history=history, arms={name: dict(path=str(path), sha256=digest(path)) for name, path in paths.items()},
        identical_updates=True, identical_support_batches=True, disjoint_query_speakers=True,
        checkpoint_selection='reserved-training-speaker mean CTC plus output preservation'))


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

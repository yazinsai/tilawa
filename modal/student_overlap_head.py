"""Phone-conditioned overlap weighting of a frozen streaming listener head.

Preparation admits authorized original training recordings and help-train clean
training takes only. No practice input is read. Transcript-constrained CTC
occupancies are frozen before domain fitting. Speaker-cross-fitted domain odds
weight both populations toward their common acoustic support. A uniform arm
uses identical examples, phone mass, initialization, batches and update count.
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
                               export, feature_runtime, stable, stream_features,
                               training_helpers, verify_bundle, write_json)

# Frozen before any incident diagnostic. The step size applies to both arms.
RECIPE = dict(seed=831, original_limit=512, views=3, epochs=12, batch=8,
              lr=.0001, anchor=2., max_seconds=30., domain_steps=200,
              domain_frames=64, domain_folds=3, domain_l2=[.01, .1, 1.])


def soft_occupancies(log_probs, target):
    """Exact frame-label posterior over all CTC paths spelling the transcript.

    Includes blank: pauses retain their original training mass. Repeated target
    labels require an intervening blank; no greedy word boundaries are used.
    """
    import numpy as np

    lp = np.asarray(log_probs, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    if lp.ndim != 2 or len(lp) == 0 or len(target) == 0 or np.any(target <= 0) or np.any(target >= lp.shape[1]):
        raise ValueError('Invalid transcript or acoustic rows')
    labels = np.zeros(2*len(target)+1, dtype=np.int64)
    labels[1::2] = target
    skip = np.zeros(len(labels), dtype=bool)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    alpha = np.full((len(lp), len(labels)), -np.inf)
    alpha[0, :2] = lp[0, labels[:2]]
    for t in range(1, len(lp)):
        prev = alpha[t-1]
        step = np.logaddexp(prev, np.r_[-np.inf, prev[:-1]])
        jump = np.r_[-np.inf, -np.inf, prev[:-2]]
        alpha[t] = np.logaddexp(step, np.where(skip, jump, -np.inf))+lp[t, labels]
    total = np.logaddexp(alpha[-1, -1], alpha[-1, -2])
    if not np.isfinite(total):
        raise ValueError('Transcript has no complete CTC path through recording')
    beta = np.full_like(alpha, -np.inf)
    beta[-1, -2:] = 0.
    for t in range(len(lp)-2, -1, -1):
        nxt = beta[t+1]+lp[t+1, labels]
        step = np.logaddexp(nxt, np.r_[nxt[1:], -np.inf])
        jump = np.r_[nxt[2:], -np.inf, -np.inf]
        beta[t] = np.logaddexp(step, np.where(np.r_[skip[2:], False, False], jump, -np.inf))
    states = np.exp(alpha+beta-total)
    result = np.zeros_like(lp)
    for state, label in enumerate(labels):
        result[:, label] += states[:, state]
    np.testing.assert_allclose(result.sum(1), 1., atol=1e-6)
    return result.astype('float32')


def speaker_folds(views, count):
    """Round-robin whole speakers within each population; no empty domain fold."""
    groups = {}
    for source in ('original', 'help'):
        speakers = sorted({v['speaker'] for v in views if v['source'] == source}, key=stable)
        if len(speakers) < count:
            raise ValueError('Too few training speakers for domain cross-fitting')
        groups.update({s: i % count for i, s in enumerate(speakers)})
    return [groups[v['speaker']] for v in views]


def phone_scales(occupancies, probabilities, sources):
    """Both arms preserve original per-phone mass, including blank.

    Balanced-source domain odds estimate P(learner | features, phone), so
    e(x) and 1-e(x) downweight population-specific extremes symmetrically.
    Normalization is learned globally on training partitions, never per batch
    or from validation/practice. Absent original phones get zero mass.
    """
    import numpy as np

    original = sum((q.sum(0) for q, s in zip(occupancies, sources) if s == 'original'),
                   np.zeros(occupancies[0].shape[1]))
    totals = {'uniform': np.zeros_like(original), 'weighted': np.zeros_like(original)}
    for q, p, s in zip(occupancies, probabilities, sources):
        if p.shape != q.shape or not np.isfinite(p).all() or np.any(p <= 0) or np.any(p >= 1):
            raise ValueError('Invalid cross-fitted domain probability')
        totals['uniform'] += q.sum(0)
        totals['weighted'] += (q*(p if s == 'original' else 1-p)).sum(0)
    return {name: np.divide(original, mass, out=np.zeros_like(original), where=mass > 0).astype('float32')
            for name, mass in totals.items()}


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
    verify_bundle(model)
    records, excluded, labels_sha = authorized_records(RECIPE['original_limit'], RECIPE['max_seconds'])
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
    if any(not r['target'] for r in records):
        raise ValueError('Transcribed recording has no authorized phone target')
    # Exact waveform copies across speakers invalidate partition independence.
    # Audit content before producing any feature cache.
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
            target = np.asarray(r['target'], dtype=np.int64)
            occupancy = soft_occupancies(lp, target)
            np.savez_compressed(file, features=x, baseline=lp, target=target, occupancy=occupancy)
            views.append(dict(recording=r['key'], speaker=r['speaker'], partition=r['partition'],
                              source=r['source'], file=str(file), sha256=digest(file)))
        print(json.dumps(dict(cached=r['key'], partition=r['partition'])), flush=True)
    write_json(out/'plan.json', dict(schema=1, recipe=RECIPE, records=records, views=views,
        excluded_speakers=excluded, help_labels_sha256=labels_sha, primary_sha256=manifest['sha256'],
        feature_graph_sha256=digest(out/'encoder-features.onnx'), initial_head_sha256=digest(out/'initial-head.npz')))


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
    selection_path = out/'selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        domain_sha = selection.get('domain_selection_sha256')
        if domain_sha is not None:
            domain_path = out/'domain-selection.json'
            if digest(domain_path) != domain_sha:
                raise ValueError('Frozen domain selection changed')
            domain = json.loads(domain_path.read_text())
            if domain['plan_sha256'] != digest(out/'plan.json'):
                raise ValueError('Domain weights belong to another training plan')
            for fold in domain['folds']:
                fitted, predicted = set(fold['fit_speakers']), set(fold['predicted_speakers'])
                if fitted & predicted or (fitted | predicted)-train_speakers:
                    raise ValueError('Domain discriminator speaker separation violated')
            for artifact in domain['folds']+domain['weights']:
                if digest(artifact['path']) != artifact['sha256']:
                    raise ValueError('Frozen domain model or overlap weights changed')
    return plan


def domain_bank(views):
    """Bounded deterministic frame sample; equal aggregate mass per speaker."""
    import numpy as np

    counts = {}
    for v in views:
        counts[v['speaker']] = counts.get(v['speaker'], 0)+1
    xs, qs, ys = [], [], []
    for v in views:
        with np.load(v['file']) as a:
            n = len(a['features'])
            rng = np.random.default_rng(int(stable(v['file'])[:16], 16))
            ids = np.sort(rng.choice(n, min(n, RECIPE['domain_frames']), replace=False))
            xs.append(a['features'][ids])
            qs.append(a['occupancy'][ids]/(len(ids)*counts[v['speaker']]))
            ys.append(np.full(len(ids), v['source'] == 'help', dtype='float32'))
    return np.concatenate(xs), np.concatenate(qs), np.concatenate(ys)


def domain_mass(q, y):
    """Each phone has equal learner/original prior, independent of text mix."""
    import numpy as np

    mass = np.zeros_like(q)
    counts = [q[y == label].sum(0) for label in (0, 1)]
    present = (counts[0] > 1e-7) & (counts[1] > 1e-7)
    if not present.any():
        raise ValueError('Populations share no transcript phone support')
    for label in (0, 1):
        scale = np.divide(present, counts[label], out=np.zeros_like(counts[label]), where=counts[label] > 0)
        mass[y == label] = q[y == label]*scale/(2*present.sum())
    return mass, present


def fit_domain(bank, regularization, device):
    """Regularized phone-specific logits: acoustic-feature x phone interaction."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    x, q, y = bank
    mean, std = x.mean(0), np.maximum(x.std(0), .1)
    mass, present = domain_mass(q, y)
    x = torch.as_tensor((x-mean)/std, device=device)
    weights = torch.zeros((x.shape[1], q.shape[1]), device=device, requires_grad=True)
    bias = torch.zeros(q.shape[1], device=device, requires_grad=True)
    mass = torch.as_tensor(mass, device=device)
    y = torch.as_tensor(y, device=device)[:, None].expand_as(mass)
    optimizer = torch.optim.Adam([weights, bias], lr=.02)
    for _ in range(RECIPE['domain_steps']):
        optimizer.zero_grad(set_to_none=True)
        loss = (F.binary_cross_entropy_with_logits(x@weights+bias, y, reduction='none')*mass).sum()
        loss = loss+regularization*(weights.square().sum()+bias.square().sum())/present.sum()
        loss.backward()
        optimizer.step()
    return dict(mean=mean, std=std, weight=weights.detach().cpu().numpy(),
                bias=bias.detach().cpu().numpy(), present=present)


def domain_probability(model, features):
    import numpy as np

    logits = ((features-model['mean'])/model['std'])@model['weight']+model['bias']
    probabilities = 1/(1+np.exp(-np.clip(logits, -12, 12)))
    # No cross-population evidence for this phone: retain uniform overlap odds.
    probabilities[:, ~model['present']] = .5
    return probabilities.astype('float32')


def domain_validation(model, bank):
    import numpy as np

    x, q, y = bank
    mass, _ = domain_mass(q, y)
    p = domain_probability(model, x)
    return float(-(mass*(y[:, None]*np.log(p)+(1-y[:, None])*np.log1p(-p))).sum())


def fit(out):
    import copy
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(2)
    torch.manual_seed(RECIPE['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plan = read_plan(out)
    train = [v for v in plan['views'] if v['partition'] == 'train']
    validation = [v for v in plan['views'] if v['partition'] == 'validation']
    if not train or not validation:
        raise ValueError('Missing training or reserved-speaker recordings')
    bank, val_bank = domain_bank(train), domain_bank(validation)
    regularization_history = []
    for l2 in RECIPE['domain_l2']:
        model = fit_domain(bank, l2, device)
        regularization_history.append(dict(l2=l2, loss=domain_validation(model, val_bank)))
    l2 = min(regularization_history, key=lambda r: (r['loss'], -r['l2']))['l2']
    print(json.dumps(dict(domain_regularization=l2, validation=regularization_history)), flush=True)
    folds = speaker_folds(train, RECIPE['domain_folds'])
    probabilities = [None]*len(train)
    crossfit = []
    for fold in range(RECIPE['domain_folds']):
        fitted = [v for v, f in zip(train, folds) if f != fold]
        predicted = [v for v, f in zip(train, folds) if f == fold]
        fit_speakers = {v['speaker'] for v in fitted}
        predict_speakers = {v['speaker'] for v in predicted}
        if fit_speakers & predict_speakers:
            raise ValueError('Domain discriminator sees its predicted speaker')
        model = fit_domain(domain_bank(fitted), l2, device)
        file = out/f'domain-{fold}.npz'
        np.savez(file, **model)
        crossfit.append(dict(fold=fold, fit_speakers=sorted(fit_speakers),
            predicted_speakers=sorted(predict_speakers), path=str(file), sha256=digest(file)))
        for i, (v, f) in enumerate(zip(train, folds)):
            if f == fold:
                with np.load(v['file']) as a:
                    probabilities[i] = domain_probability(model, a['features'])
    occupancies = []
    for v in train:
        with np.load(v['file']) as a:
            occupancies.append(a['occupancy'].copy())
    scales = phone_scales(occupancies, probabilities, [v['source'] for v in train])
    original_mass = sum(q.sum(0) for q, v in zip(occupancies, train) if v['source'] == 'original')
    actual_mass = {}
    for name, scale in scales.items():
        actual_mass[name] = sum((q*scale*(p if v['source'] == 'original' else 1-p)).sum(0)
                                if name == 'weighted' else (q*scale).sum(0)
                                for q, p, v in zip(occupancies, probabilities, train))
        np.testing.assert_allclose(actual_mass[name], original_mass, rtol=2e-5, atol=1e-4,
                                   err_msg='Overlap weighting changes original phone mass')
    weighted_files = []
    for i, v in enumerate(train):
        file = out/f'overlap-{i}.npz'
        np.savez_compressed(file, probability=probabilities[i])
        weighted_files.append(dict(path=str(file), sha256=digest(file)))
    write_json(out/'domain-selection.json', dict(regularization=l2, history=regularization_history,
        folds=crossfit, weights=weighted_files, scales={k: v.tolist() for k, v in scales.items()},
        original_phone_mass=original_mass.tolist(), actual_phone_mass={k: v.tolist() for k, v in actual_mass.items()},
        plan_sha256=digest(out/'plan.json'), source_priors='equal within phone'))

    @lru_cache(maxsize=32)
    def cached(path):
        with np.load(path) as a:
            return {k: torch.as_tensor(a[k].copy(), device=device) for k in ('features', 'baseline', 'occupancy')}

    initial = np.load(out/'initial-head.npz')
    head = torch.nn.Linear(initial['weight'].shape[1], initial['weight'].shape[0]).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(initial['weight'], device=device))
        head.bias.copy_(torch.as_tensor(initial['bias'], device=device))
    arms = {'uniform': head, 'weighted': copy.deepcopy(head)}
    scale_tensors = {k: torch.as_tensor(v, device=device) for k, v in scales.items()}

    def objective(h, d, mass):
        lp = h(d['features']).log_softmax(-1)
        transcript = -(mass*lp).sum()
        anchor = F.kl_div(lp, d['baseline'].exp(), reduction='sum')
        return transcript+RECIPE['anchor']*anchor

    def evaluate(h):
        speakers = {}
        with torch.no_grad():
            for v in validation:
                d = cached(v['file'])
                loss = objective(h, d, d['occupancy'])/len(d['features'])
                speakers.setdefault(v['speaker'], []).append(float(loss))
        result = float(np.mean([np.mean(values) for values in speakers.values()]))
        if not math.isfinite(result):
            raise ValueError('Non-finite reserved-speaker head loss')
        return result

    baseline = evaluate(head)
    history, updates = [], 0
    normalizer = sum(len(q) for q in occupancies)/math.ceil(len(train)/RECIPE['batch'])
    for epoch in range(RECIPE['epochs']):
        order = np.random.default_rng(RECIPE['seed']+epoch).permutation(len(train))
        for start in range(0, len(order), RECIPE['batch']):
            ids = order[start:start+RECIPE['batch']]
            for name, h in arms.items():
                h.zero_grad(set_to_none=True)
                loss = 0.
                for i in ids:
                    d = cached(train[i]['file'])
                    mass = d['occupancy']*scale_tensors[name]
                    if name == 'weighted':
                        p = torch.as_tensor(probabilities[i], device=device)
                        mass = mass*(p if train[i]['source'] == 'original' else 1-p)
                    loss = loss+objective(h, d, mass)/normalizer
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in h.parameters()):
                    raise ValueError('Non-finite phone-head gradient')
                with torch.no_grad():
                    for p in h.parameters():
                        p.add_(p.grad, alpha=-RECIPE['lr'])
            updates += 1
        losses = {name: evaluate(h) for name, h in arms.items()}
        for name, h in arms.items():
            np.savez(out/f'{name}-{epoch+1}.npz', weight=h.weight.detach().cpu().numpy(),
                     bias=h.bias.detach().cpu().numpy())
        history.append(dict(epoch=epoch+1, updates=updates, validation=losses))
        print(json.dumps(history[-1]), flush=True)
    selected = min(history, key=lambda r: (r['validation']['weighted'], r['epoch']))
    paths = {name: out/f"{name}-{selected['epoch']}.npz" for name in arms}
    write_json(out/'selection.json', dict(schema=1, plan_sha256=digest(out/'plan.json'),
        domain_selection_sha256=digest(out/'domain-selection.json'), baseline_validation=baseline,
        selected=selected, history=history,
        arms={name: dict(path=str(path), sha256=digest(path)) for name, path in paths.items()},
        identical_updates=True, identical_examples=True, identical_phone_mass=True,
        checkpoint_selection='reserved-training-speaker mean soft-transcript CE plus output preservation'))


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

"""Fresh phone-head training with per-example early-learning retention.

prepare/fit/export never read practice recordings or scores. All schedule and
checkpoint decisions use whole-speaker partitions of authorized training data.
All choices are made before target-window evaluation.
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
    epochs=24, warmups=[4, 8, 12], strengths=[.1, .3, 1.], bank_decay=.9)
EXPERIMENT = 'listener-early-learning-retention'


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
            manifest['sampleRate'] != 16000 or manifest['fbank']['mels'] != 80 or manifest['shift'] % 4 or
            manifest['window'] != manifest['shift']+13):
        raise ValueError('Feature capture must retain the shipped inventory and clock')
    # The encoder stays shipped; both arms begin with this one fresh head.
    torch.manual_seed(RECIPE['seed'])
    fresh = torch.nn.Linear(head['weight'].shape[0], head['weight'].shape[1])
    np.savez(out/'initial-head.npz', weight=fresh.weight.detach().numpy(),
             bias=fresh.bias.detach().numpy())
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
            logits = features@head['weight']+head['bias']
            reconstructed = logits-np.logaddexp.reduce(logits, axis=1, keepdims=True)
            if not np.allclose(reconstructed, baseline, atol=.002, rtol=0):
                raise ValueError('Features do not reproduce the deployed head')
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


class PredictionBank:
    """Per-example temporal predictions; never weights or transcript labels.

    Views have separate keys because their acoustic frames differ. The bank is
    updated from detached predictions, including during transcript-only warm-up.
    No validation example can update it. A clone is needed for each matched arm.
    """
    def __init__(self, decay):
        if not 0 < decay < 1:
            raise ValueError('Prediction decay must lie strictly between zero and one')
        self.decay, self.rows, self.counts = decay, {}, {}

    def observe(self, key, probabilities):
        import torch

        p = probabilities.detach().cpu().clone()
        if (p.ndim != 2 or not len(p) or not torch.isfinite(p).all() or
                (p < 0).any() or not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5)):
            raise ValueError('Invalid per-example probabilities')
        if key in self.rows:
            if self.rows[key].shape != p.shape:
                raise ValueError('Temporal prediction bank frame geometry changed')
            self.rows[key].mul_(self.decay).add_(p, alpha=1-self.decay)
        else:
            self.rows[key] = p
        self.counts[key] = self.counts.get(key, 0)+1

    def penalty(self, key, log_probs):
        # Minimize log(1-<early target,current prediction>), the ELR objective.
        # Use history strictly before the current update: a just-confident
        # prediction must not manufacture its own retention target.
        if key not in self.rows or self.rows[key].shape != log_probs.shape:
            raise ValueError('Missing or mismatched early prediction history')
        target = self.rows[key].to(log_probs.device).detach()
        overlap = (target*log_probs.exp()).sum(-1)
        return (1-overlap).clamp_min(1e-7).log().mean()


def transcript_loss(log_probs, target):
    import torch
    import torch.nn.functional as F

    if (log_probs.ndim != 2 or target.ndim != 1 or target.dtype != torch.long or
            not torch.isfinite(log_probs).all() or (target <= 0).any() or
            (target >= log_probs.shape[-1]).any()):
        raise ValueError('Invalid supervised phone sequence or posteriors')
    required = len(target)+int((target[1:] == target[:-1]).sum())
    if not len(target) or required > len(log_probs):
        raise ValueError('Impossible or empty supervised transcript')
    return F.ctc_loss(log_probs[:, None], target, torch.tensor([len(log_probs)]),
        torch.tensor([len(target)]), blank=0, reduction='sum', zero_infinity=False)/len(log_probs)


def preservation_loss(log_probs, baseline):
    import torch
    import torch.nn.functional as F

    if log_probs.ndim != 2 or not len(log_probs) or log_probs.shape != baseline.shape:
        raise ValueError('Preservation frame geometry changed')
    if not torch.isfinite(log_probs).all() or not torch.isfinite(baseline).all():
        raise ValueError('Invalid preservation posteriors')
    return F.kl_div(log_probs, baseline.exp(), reduction='sum')/len(log_probs)


def fit(out):
    import numpy as np
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(RECIPE['seed'])
    plan = read_plan(out)
    train = [v for v in plan['views'] if v['partition'] == 'train' and v['supervised']]
    protected = [v for v in plan['views'] if v['partition'] == 'train' and not v['supervised']]
    validation = [v for v in plan['views'] if v['partition'] == 'validation']
    if not train or not protected or not any(v['supervised'] for v in validation):
        raise ValueError('Fit supervision, mistake preservation and held-speaker validation are required')
    initial = np.load(out/'initial-head.npz')
    template = torch.nn.Linear(initial['weight'].shape[1], initial['weight'].shape[0])
    with torch.no_grad():
        template.weight.copy_(torch.from_numpy(initial['weight']))
        template.bias.copy_(torch.from_numpy(initial['bias']))

    @lru_cache(maxsize=32)
    def cached(path):
        with np.load(path) as a:
            return {k: torch.from_numpy(a[k].copy()) for k in ('features', 'baseline', 'target')}

    def evaluation(h):
        # Equal speaker weight; neither the retention penalty nor target
        # incidents participate in checkpoint/schedule selection.
        speakers = {}
        with torch.no_grad():
            for v in validation:
                d = cached(v['file'])
                lp = h(d['features']).log_softmax(-1)
                value = (transcript_loss(lp, d['target']) if v['supervised'] else
                         RECIPE['anchor']*preservation_loss(lp, d['baseline']))
                speakers.setdefault(v['speaker'], []).append(float(value))
        result = float(np.mean([np.mean(values) for values in speakers.values()]))
        if not math.isfinite(result):
            raise ValueError('Nonfinite speaker validation loss')
        return result

    def save(h, name):
        path = out/(name+'.npz')
        np.savez(path, weight=h.weight.detach().numpy(), bias=h.bias.detach().numpy())
        return dict(path=str(path), sha256=digest(path))

    def optimize(h, opt, support, preservation, bank, weight):
        opt.zero_grad(set_to_none=True)
        terms, predictions = [], []
        for v in support:
            d = cached(v['file'])
            lp = h(d['features']).log_softmax(-1)
            value = transcript_loss(lp, d['target'])
            # Clean takes learn only from their transcript. Anchoring these
            # same frames to the shipped head after warm-up could manufacture
            # an early-to-late reversal toward its existing false substitution.
            # Shipped-output preservation applies only to script-free takes.
            if weight:
                value += weight*bank.penalty(v['file'], lp)
            terms.append(value)
            predictions.append((v['file'], lp.exp().detach()))
        loss = sum(terms)/len(terms)
        if preservation:
            loss += RECIPE['anchor']*sum(preservation_loss(
                h(cached(v['file'])['features']).log_softmax(-1), cached(v['file'])['baseline'])
                for v in preservation)/len(preservation)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training objective')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in h.parameters()):
            raise ValueError('Invalid head gradient')
        opt.step()
        for key, p in predictions:
            bank.observe(key, p)

    histories, choices = {}, []
    for warmup in RECIPE['warmups']:
        control = copy.deepcopy(template)
        opt = torch.optim.SGD(control.parameters(), lr=RECIPE['lr'], momentum=RECIPE['momentum'])
        control_bank = PredictionBank(RECIPE['bank_decay'])
        arms, early, history = [], None, []
        for epoch in range(RECIPE['epochs']):
            order = np.random.default_rng(RECIPE['seed']+epoch).permutation(len(train))
            for start in range(0, len(order), RECIPE['batch']):
                support = [train[i] for i in order[start:start+RECIPE['batch']]]
                step = epoch*math.ceil(len(train)/RECIPE['batch'])+start//RECIPE['batch']
                preservation = [protected[(step*RECIPE['preservation_batch']+j) % len(protected)]
                    for j in range(RECIPE['preservation_batch'])] if epoch >= warmup else []
                optimize(control, opt, support, preservation, control_bank, 0.)
                for arm in arms:
                    optimize(arm['head'], arm['optimizer'], support, preservation, arm['bank'], arm['weight'])
            checkpoint = save(control, f'control-w{warmup}-e{epoch+1}')
            value = evaluation(control)
            row = dict(epoch=epoch+1, control_validation=value, control=checkpoint)
            if epoch+1 == warmup:
                early = checkpoint
                for weight in RECIPE['strengths']:
                    h = copy.deepcopy(control)
                    optimizer = torch.optim.SGD(h.parameters(), lr=RECIPE['lr'], momentum=RECIPE['momentum'])
                    optimizer.load_state_dict(copy.deepcopy(opt.state_dict()))
                    arms.append(dict(head=h, optimizer=optimizer, bank=copy.deepcopy(control_bank), weight=weight))
            if epoch >= warmup:
                row['retention'] = []
                for arm in arms:
                    score = evaluation(arm['head'])
                    ck = save(arm['head'], f"retention-w{warmup}-r{arm['weight']}-e{epoch+1}")
                    row['retention'].append(dict(weight=arm['weight'], validation=score, checkpoint=ck))
                    choices.append(dict(validation=score, warmup=warmup, weight=arm['weight'], epoch=epoch+1,
                        retention=ck, early_control=early, matched_control=checkpoint))
            history.append(row)
            print(json.dumps(dict(warmup=warmup, **row)), flush=True)
        histories[str(warmup)] = history
    chosen = min(choices, key=lambda c: (c['validation'], c['weight'], c['warmup'], c['epoch']))
    terminal = histories[str(chosen['warmup'])][-1]['control']
    write_json(out/'selection.json', dict(schema=1, plan_sha256=digest(out/'plan.json'),
        chosen=chosen, arms=dict(retention=chosen['retention'], control=chosen['matched_control'],
            early_control=chosen['early_control'], late_control=terminal), histories=histories,
        selection='speaker-grouped training CTC and shipped-output preservation only',
        identical_initialization=True, identical_orders=True, prediction_targets='detached past example predictions'))


def compare(out, windows):
    """Read-only premise probe on externally supplied full-stream windows.

    This command is deliberately separate from fit/selection. Requests contain
    features, frame bounds and COMPLETE written/accused phone sequences. It
    reports acoustic preferences, never claims an engine-warning win.
    """
    import numpy as np
    import torch

    plan = read_plan(out)
    selection = json.loads((out/'selection.json').read_text())
    if selection['plan_sha256'] != digest(out/'plan.json'):
        raise ValueError('Selection differs from frozen training plan')
    heads = {}
    for name in ('early_control', 'late_control', 'retention'):
        ck = selection['arms'][name]
        if digest(ck['path']) != ck['sha256']:
            raise ValueError('Frozen checkpoint changed')
        with np.load(ck['path']) as a:
            heads[name] = (torch.from_numpy(a['weight'].copy()), torch.from_numpy(a['bias'].copy()))
    requests = json.loads(Path(windows).read_text())
    results = []
    with torch.no_grad():
        for r in requests:
            if r['feature_graph_sha256'] != plan['feature_graph_sha256']:
                raise ValueError('Diagnostic features differ from the frozen encoder')
            with np.load(r['features']) as a:
                x = torch.from_numpy(a['features'][r['from_frame']:r['to_frame']].copy())
            if len(x) != r['to_frame']-r['from_frame'] or not len(x) or r['from_frame'] < 0:
                raise ValueError('Invalid complete accusation window')
            preferences = {}
            for name, (w, b) in heads.items():
                lp = (x@w.T+b).log_softmax(-1)
                written = torch.tensor(r['written'], dtype=torch.long)
                accused = torch.tensor(r['accused'], dtype=torch.long)
                margin = float((transcript_loss(lp, accused)-transcript_loss(lp, written))*len(lp))
                if not math.isfinite(margin):
                    raise ValueError('Unavailable complete-pronunciation comparison')
                preferences[name] = margin
            results.append(dict(key=r['key'], features_sha256=digest(r['features']), written_advantage=preferences,
                early_to_late_reversal=preferences['early_control'] > 0 and preferences['late_control'] < 0))
    write_json(out/'premise.json', dict(windows_sha256=digest(windows),
        selection_sha256=digest(out/'selection.json'), comparisons=results,
        reversals=sum(r['early_to_late_reversal'] for r in results),
        limitation='Complete CTC preference only; not engine-warning wins or a practice score'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'fit', 'export', 'compare'))
    p.add_argument('--model', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--windows', type=Path)
    args = p.parse_args()
    if args.command in ('prepare', 'export') and args.model is None:
        p.error('prepare/export require --model')
    if args.command == 'compare' and args.windows is None:
        p.error('compare requires --windows')
    if args.command == 'prepare':
        prepare(args.model, args.out)
    elif args.command == 'fit':
        fit(args.out)
    elif args.command == 'compare':
        compare(args.out, args.windows)
    else:
        export(args.model, args.out, read_plan, RECIPE['seed'])

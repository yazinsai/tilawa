"""Bounded, training-only null-attention experiment, run in the supervisor's train image.

Commands: cache, fit, export. All outputs belong under the experiment's scratch
root. Repeating cache/fit resumes completed recordings/updates. No practice
labels, pause labels, expected verse or candidate identity enter attention.

Example (after auditing the initial checkpoint and supplied training metadata):
  python modal/student_null_fit.py cache --checkpoint /student/prior-runs/<run>/ckpt.pt --output /scratch/<experiment>/null-fit
  python modal/student_null_fit.py fit --checkpoint ... --output ...
  python modal/student_null_fit.py export --checkpoint ... --output ... --auxiliary-dir <complete shipped teacher directory>

Two deterministic views (ordinary and shipped augmentation) share frozen lower
features across both arms. Speaker-grouped training validation selects among
fixed preservation weights and epoch checkpoints, before any target decoding.
All three commands together are capped at 86400 process CPU seconds, including
cache construction. The cap is persisted across resumed supervisor requests.
"""

import argparse
import copy
import hashlib
import io
import json
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from student_null_attention import install_query_attention, upper_block

SEED = 20261112
PRESERVATION = (1.0, 10.0, 100.0)
EPOCHS = 4
CPU_LIMIT = 86400.0


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    tmp.replace(path)


def save_torch(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


class Budget:
    def __init__(self, root):
        self.path = root / 'cpu-budget.json'
        self.spent = json.loads(self.path.read_text())['cpu_seconds'] if self.path.exists() else 0.0
        self.start = time.process_time()

    def check(self):
        used = self.spent + time.process_time() - self.start
        atomic_json(self.path, {'cpu_seconds': used, 'limit': CPU_LIMIT})
        if used >= CPU_LIMIT:
            raise RuntimeError('one CPU day exhausted; no target-based selection is permitted')


def group_partition(records):
    """Only admitted training speakers; a speaker cannot occur in both partitions."""
    speakers = sorted({r['speaker'] for r in records})
    if len(speakers) < 3:
        raise ValueError('at least three authorized training speakers are required')
    ordered = sorted(speakers, key=lambda s: hashlib.sha256(f'{SEED}:{s}'.encode()).digest())
    val = set(ordered[:max(1, len(ordered) // 5)])
    return ['validation' if r['speaker'] in val else 'fit' for r in records]


def help_records(root, reference):
    ayahs = {f"{a['s']}:{a['a']}": a['p'] for a in reference['ayahs']}
    rows = [json.loads(s) for s in (root / 'labels.jsonl').read_text().splitlines() if s.strip()]
    held = {r['speaker'] for r in rows if r.get('split') != 'train'}
    out = []
    for r in rows:
        if r['speaker'] in held or r.get('split') != 'train' or r.get('kind') != 'none':
            continue
        if not r.get('scored') or r.get('excluded') or r.get('events') or r.get('non_errors') or r.get('unsupported'):
            continue
        path = (root / r['wav']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('help audio must stay inside the authorized training directory')
        if not r.get('ayahs'):
            raise ValueError('none-labelled training take has no transcript')
        out.append({'key': f"help:{r['id']}", 'speaker': f"help:{r['speaker']}",
                    'target': ''.join(ayahs[a] for a in r['ayahs']), 'audio': path.read_bytes()})
    if not out:
        raise ValueError('no eligible none-labelled training-speaker recordings')
    return out


def training_records():
    import pyarrow.parquet as pq
    from student import exclusions
    gold = exclusions(Path('/root/gold-per-v1-exclude.txt').read_text())
    tables = [pq.read_table(p, columns=['shard', 'row', 'cluster', 'split', 'bucket', 'n_samples'])
              for p in sorted(Path('/student/s1').glob('*.parquet'))]
    files = sorted(Path('/student/s1').glob('*.parquet'))
    # Native cluster is the training pipeline's speaker grouping. Refuse missing
    # identities and exclude any cluster present outside train in this corpus.
    metadata = [t.to_pylist() for t in tables]
    held = {r['cluster'] for rows in metadata for r in rows if r['split'] != 'train'}
    held |= {r['cluster'] for rows in metadata for r in rows if (r['shard'], r['row']) in gold}
    selected = []
    for file, rows in zip(files, metadata):
        for i, r in enumerate(rows):
            if r['split'] != 'train' or r['bucket'] != 'clean-match' or r['cluster'] in held:
                continue
            if r['cluster'] is None or str(r['cluster']) in ('', '-1'):
                continue
            if (r['shard'], r['row']) in gold or not 16000 <= r['n_samples'] <= 30 * 16000:
                continue
            key = f"{r['shard']}:{r['row']}"
            selected.append((hashlib.sha256(f'{SEED}:{key}'.encode()).digest(), file, i, r))
    # Fixed two-hour bound, chosen before practice inspection. All admitted help
    # takes are additional ordinary-reading supervision, never scoring items.
    selected.sort(key=lambda r: r[0])
    seconds, chosen = 0, {}
    for _, file, i, r in selected:
        if seconds + r['n_samples'] / 16000 > 7200:
            continue
        seconds += r['n_samples'] / 16000
        chosen.setdefault(file, []).append(i)
    out = []
    for file, indices in chosen.items():
        for r in pq.read_table(file, columns=['shard', 'row', 'cluster', 'ref', 'flac']).take(indices).to_pylist():
            out.append({'key': f"{r['shard']}:{r['row']}", 'speaker': f"native:{r['cluster']}",
                        'target': r['ref'], 'audio': r['flac']})
    if not out:
        raise ValueError('no authorized speaker-grouped transcribed native recordings')
    reference = json.loads(Path('packages/core/assets/ref-v2.json').read_text())
    out += help_records(Path(os.environ['LAB_DATA']) / 'help-train', reference)
    partitions = group_partition(out)
    for r, partition in zip(out, partitions):
        r['partition'] = partition
    return sorted(out, key=lambda r: r['key'])


def load_base(checkpoint, baseline_dir):
    from student_train import build_model
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ck['cfg'].get('query_attention'):
        raise ValueError('initial checkpoint must have unchanged attention')
    model = build_model(ck['cfg']['size'])
    model.load_state_dict(ck['model'], strict=True)
    model.eval().requires_grad_(False)
    manifest = json.loads((baseline_dir / 'manifest.json').read_text())
    if file_hash(baseline_dir / manifest['file']) != manifest['sha256']:
        raise ValueError('supplied primary model checksum differs')
    if manifest.get('trainStep') is not None and manifest['trainStep'] != ck['state']['step']:
        raise ValueError('initial checkpoint is not the supplied listener training step')
    model.streaming_eval(manifest['chunk'], manifest['leftContext'])
    return model, ck, manifest


def acoustic_features(record, view, cfg, aug):
    import numpy as np
    import soundfile as sf
    import torchaudio
    from student import mix_noise
    from student_train import pick, reverb, spec_augment
    wave, sr = sf.read(io.BytesIO(record['audio']), dtype='float32')
    if sr != 16000 or wave.ndim != 1:
        raise ValueError('authorized audio must be supplied mono 16 kHz')
    seed = int.from_bytes(hashlib.sha256(f"{SEED}:{record['key']}:{view}".encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    if view:
        if rng.random() < cfg['rir_p']:
            wave = reverb(wave, pick(aug['rir'], rng))
        if rng.random() < cfg['noise_p']:
            wave = mix_noise(wave, pick(aug['noise'], rng), rng.uniform(*cfg['snr_db']), rng)
        wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
    feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave))[None],
        num_mel_bins=80, sample_frequency=16000, dither=0.0, energy_floor=1e-10, high_freq=-400)
    if view and cfg['spec_aug']:
        feats = spec_augment(feats, rng)
    return feats


def cache_features(model, feats, manifest):
    """Stream frozen prefix; cache input to just the last layer, plus bypass/tail.

    Upper-layer values and convolutions are replayed sequentially during fitting,
    so adapted state is carried. Baseline upper caches are not substituted for
    adapted caches. All preceding layers retain their exact original states.
    """
    from zipformer import convert_num_channels
    enc = model.encoder
    final_stage = enc.encoders[-1]
    inner = getattr(final_stage, 'encoder', final_stage)
    chunk, left = manifest['chunk'], manifest['leftContext']
    states = enc.get_init_states(1, torch.device('cpu'))
    embed = model.encoder_embed.get_init_states(1, torch.device('cpu'))
    initial_upper = [s.clone() for s in states[-6:]]
    need = ((len(feats) - 7) // 2 + 1) // 2
    windows = max(1, -(-need // (chunk // 2)))
    total = manifest['window'] + (windows - 1) * manifest['shift']
    padded = F.pad(feats, (0, 0, 0, max(0, total - len(feats))), value=manifest['fbank']['padValue'])
    processed, items, baseline = torch.zeros(1, dtype=torch.int64), [], []
    for start in range(0, total - manifest['window'] + 1, manifest['shift']):
        x, lens, embed = model.encoder_embed.streaming_forward(padded[None, start:start + manifest['window']],
            torch.tensor([manifest['window']]), cached_left_pad=embed)
        mask = (processed[:, None] <= torch.arange(left)[None]).to(torch.int32).flip(1)
        mask = torch.cat([mask, torch.zeros(1, chunk, dtype=torch.int32)], -1)
        x = x.permute(1, 0, 2)
        lower_outputs, offset, new_states = [], 0, []
        for i, stage in enumerate(enc.encoders[:-1]):
            ds = enc.downsampling_factor[i]
            x = convert_num_channels(x, enc.encoder_dim[i])
            x, st = stage.streaming_forward(x, states[offset:offset + stage.num_layers * 6],
                left_context_len=left // ds, src_key_padding_mask=mask[..., ::ds])
            new_states += st
            offset += stage.num_layers * 6
            lower_outputs.append(x)
        orig = convert_num_channels(x, enc.encoder_dim[-1])
        src = final_stage.downsample(orig) if hasattr(final_stage, 'encoder') else orig
        ds = enc.downsampling_factor[-1]
        upper_mask = mask[..., ::ds]
        pos = inner.encoder_pos(src, left // ds)
        for layer in inner.layers[:-1]:
            result = layer.streaming_forward(src, pos, *states[offset:offset + 6], left // ds, upper_mask)
            src, st = result[0], list(result[1:])
            new_states += st
            offset += 6
        tail = enc._get_full_dim_output(lower_outputs + [orig])[..., orig.shape[-1]:]
        item = dict(src=src.clone(), pos=pos.clone(), orig=orig.clone(), tail=tail.clone(), mask=upper_mask.clone())
        lp, st = replay_chunk(model, item, states[offset:offset + 6], manifest)
        new_states += st
        states = new_states
        items.append(item)
        baseline.append(lp)
        processed = processed + lens
    return dict(chunks=items, initial_upper=initial_upper, baseline=torch.cat(baseline, 1)[:, :need], frames=need)


def replay_chunk(model, item, states, manifest):
    enc = model.encoder
    stage = enc.encoders[-1]
    ds = enc.downsampling_factor[-1]
    result = upper_block(model).streaming_forward(item['src'], item['pos'], *states,
        manifest['leftContext'] // ds, item['mask'])
    x = result[0]
    if hasattr(stage, 'encoder'):
        x = stage.out_combiner(item['orig'], stage.upsample(x)[:item['orig'].shape[0]])
    x = enc.downsample_output(torch.cat([x, item['tail']], -1))
    return model.ctc_output(x.permute(1, 0, 2)).log_softmax(-1), list(result[1:])


def replay(model, record, manifest):
    states = record['initial_upper']
    outputs = []
    for item in record['chunks']:
        lp, states = replay_chunk(model, item, states, manifest)
        outputs.append(lp)
    return torch.cat(outputs, 1)[:, :record['frames']]


def acoustic_loss(lp, record):
    target = torch.tensor(record['target'], dtype=torch.long)
    ctc = F.ctc_loss(lp.transpose(0, 1), target, torch.tensor([lp.shape[1]]), torch.tensor([len(target)]),
                     blank=0, reduction='sum', zero_infinity=False) / max(1, len(target))
    preservation = F.kl_div(lp, record['baseline'].exp(), reduction='batchmean') / lp.shape[1]
    return ctc, preservation


def cache(args, model, ck, manifest, budget):
    import numpy as np
    from student import ctc_collapse, phoneme_ids
    from student_export import wrap
    records = training_records()
    aug = {k: dict(np.load(f'/student/aug/{k}.npz')) for k in ('noise', 'rir')}
    signature = {'checkpoint_sha256': file_hash(args.checkpoint), 'geometry': manifest,
                 'data': [{k: r[k] for k in ('key', 'speaker', 'target', 'partition')} | {'audio_sha256': hashlib.sha256(r['audio']).hexdigest()} for r in records],
                 'seed': SEED, 'preservation': PRESERVATION, 'epochs': EPOCHS}
    path = args.output / 'design.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(signature)):
        raise ValueError('cached design differs; use a fresh scratch directory')
    atomic_json(path, signature)
    directory = args.output / 'cache'
    directory.mkdir(exist_ok=True)
    index = []
    parity_checked = False
    with torch.no_grad():
        for r in records:
            for view in (0, 1):
                name = hashlib.sha256(f"{r['key']}:{view}".encode()).hexdigest() + '.pt'
                dest = directory / name
                if not dest.exists():
                    feats = acoustic_features(r, view, ck['cfg'], aug)
                    item = cache_features(model, feats, manifest)
                    item.update(target=ctc_collapse(phoneme_ids(r['target'])), speaker=r['speaker'], partition=r['partition'])
                    if not parity_checked:
                        streaming = wrap(model, manifest['chunk'], manifest['leftContext'])
                        states = streaming.init_states()
                        total = manifest['window'] + (len(item['chunks']) - 1) * manifest['shift']
                        padded = F.pad(feats, (0, 0, 0, max(0, total - len(feats))), value=manifest['fbank']['padValue'])
                        reference = []
                        for start in range(0, total - manifest['window'] + 1, manifest['shift']):
                            lp, states = streaming(padded[None, start:start + manifest['window']], states)
                            reference.append(lp)
                        torch.testing.assert_close(item['baseline'], torch.cat(reference, 1)[:, :item['frames']], rtol=1e-5, atol=1e-6)
                        parity_checked = True
                    save_torch(dest, item)
                index.append({'file': name, 'key': r['key'], 'speaker': r['speaker'], 'partition': r['partition'], 'view': view})
                budget.check()
                print('cached', len(index), r['partition'], flush=True)
    atomic_json(args.output / 'cache-index.json', index)


def validate(model, paths, manifest, budget):
    from student import ctc_collapse, edit_distance
    edits = phones = 0
    losses, preservation = [], []
    with torch.no_grad():
        for path in paths:
            budget.check()
            r = torch.load(path, weights_only=False)
            lp = replay(model, r, manifest)
            a = ctc_collapse(lp[0].argmax(-1).tolist())
            edits += edit_distance(a, r['target'])
            phones += len(r['target'])
            ctc, kl = acoustic_loss(lp, r)
            losses.append(float(ctc))
            preservation.append(float(kl))
    return {'per': edits / max(1, phones), 'ctc': sum(losses) / len(paths), 'preservation': sum(preservation) / len(paths)}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def frozen_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if name.endswith(('self_attn_weights.score_weight', 'self_attn_weights.score_bias')):
            continue
        h.update(name.replace('self_attn_weights.base.', 'self_attn_weights.').encode())
        h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def fit(args, base, ck, manifest, budget):
    design = json.loads((args.output / 'design.json').read_text())
    if design['checkpoint_sha256'] != file_hash(args.checkpoint) or design['geometry'] != manifest:
        raise ValueError('fitting checkpoint/geometry differs from cached prefix')
    if design['seed'] != SEED or design['epochs'] != EPOCHS or design['preservation'] != list(PRESERVATION):
        raise ValueError('refit schedule differs from the frozen training-only design')
    index = json.loads((args.output / 'cache-index.json').read_text())
    expected = {(r['key'], view):(r['speaker'], r['partition']) for r in design['data'] for view in (0, 1)}
    actual = {(r['key'], r['view']):(r['speaker'], r['partition']) for r in index}
    if actual != expected or len(actual) != len(index):
        raise ValueError('cached recordings differ from the frozen paired data design')
    fit_speakers = {r['speaker'] for r in index if r['partition'] == 'fit'}
    val_speakers = {r['speaker'] for r in index if r['partition'] == 'validation'}
    if fit_speakers & val_speakers:
        raise ValueError('training and validation speakers overlap')
    training = [args.output / 'cache' / r['file'] for r in index if r['partition'] == 'fit']
    validation = [args.output / 'cache' / r['file'] for r in index if r['partition'] == 'validation']
    if not training or not validation:
        raise ValueError('both speaker-grouped partitions must be nonempty')
    frozen = frozen_hash(base)
    summary = []
    for mode in ('null', 'temperature'):
        candidates = []
        for weight in PRESERVATION:
            model = copy.deepcopy(base)
            attention = install_query_attention(model, mode)
            params = [attention.score_weight, attention.score_bias]
            opt = torch.optim.Adam(params, lr=0.003)
            initial = {n: p.detach().clone() for n, p in attention.named_parameters() if p.requires_grad}
            run = args.output / f'{mode}-{weight:g}'
            run.mkdir(exist_ok=True)
            state_path = run / 'resume.pt'
            state = {'epoch': 0, 'record': 0, 'updates': 0, 'metrics': []}
            if state_path.exists():
                saved = torch.load(state_path, weights_only=False)
                attention.load_state_dict(saved['attention'])
                opt.load_state_dict(saved['optimizer'])
                state = saved['state']
            while state['epoch'] < EPOCHS:
                order = torch.randperm(len(training), generator=torch.Generator().manual_seed(SEED + state['epoch'])).tolist()
                for j in range(state['record'], len(order)):
                    budget.check()
                    r = torch.load(training[order[j]], weights_only=False)
                    lp = replay(model, r, manifest)
                    ctc, kl = acoustic_loss(lp, r)
                    regularization = sum((p - initial[n]).square().mean() for n, p in attention.named_parameters() if p.requires_grad)
                    loss = ctc + weight * kl + 1e-4 * regularization
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite complete-recording refit loss')
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 2.0, error_if_nonfinite=True)
                    opt.step()
                    state['record'], state['updates'] = j + 1, state['updates'] + 1
                    save_torch(state_path, {'state': state, 'attention': attention.state_dict(), 'optimizer': opt.state_dict()})
                    budget.check()
                metrics = validate(model, validation, manifest, budget)
                budget.check()
                if frozen_hash(model) != frozen:
                    raise ValueError('pre-existing acoustic parameters changed')
                epoch = state['epoch'] + 1
                metrics.update(mode=mode, preservation_weight=weight, epoch=epoch, updates=state['updates'])
                state['metrics'].append(metrics)
                save_torch(run / f'epoch-{epoch}.pt', {'projection': {n: p.detach().clone() for n, p in attention.named_parameters() if p.requires_grad}, 'metrics': metrics})
                state['epoch'], state['record'] = epoch, 0
                save_torch(state_path, {'state': state, 'attention': attention.state_dict(), 'optimizer': opt.state_dict()})
                print(metrics, flush=True)
            candidates += [(m, run / f"epoch-{m['epoch']}.pt") for m in state['metrics']]
        metrics, path = min(candidates, key=lambda pair: (pair[0]['per'], pair[0]['ctc'] + pair[0]['preservation'], pair[0]['epoch'], pair[0]['preservation_weight']))
        selected = copy.deepcopy(base)
        attention = install_query_attention(selected, mode)
        projection = torch.load(path, weights_only=False)['projection']
        with torch.no_grad():
            for n, p in attention.named_parameters():
                if p.requires_grad:
                    p.copy_(projection[n])
        if frozen_hash(selected) != frozen:
            raise ValueError('selected listener modified pre-existing parameters')
        save_torch(args.output / f'{mode}-selected.pt', {**{k:v for k,v in ck.items() if k not in ('opt', 'sched')},
            'cfg': {**ck['cfg'], 'query_attention': mode}, 'model': selected.state_dict(), 'selection': metrics})
        summary.append(metrics)
    atomic_json(args.output / 'selection.json', {'design_sha256': file_hash(args.output / 'design.json'), 'arms': summary, 'frozen_parameters_sha256': frozen})


def export(args, base, ck, manifest, budget):
    import re
    import shutil
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from student_export import export_onnx, stream_onnx, manifest as make_manifest
    (args.output / 'verified-bundles.json').unlink(missing_ok=True)
    # The SDK attaches the shipped auxiliary from the registry; include and
    # verify it, even though it is not named in the primary model manifest.
    registry = Path('packages/sdk/src/registry.ts').read_text()
    assets = re.findall(r'file: "([^"]+)", bytes: (\d+),\s*sha256: "([a-f0-9]+)"', registry)
    if not assets:
        raise ValueError('cannot verify the existing insertion-verification bundle')
    if args.auxiliary_dir is None:
        raise ValueError('export needs the complete unchanged insertion teacher assets')
    for file, size, digest in assets:
        source = args.auxiliary_dir / file
        if not source.is_file() or source.stat().st_size != int(size) or file_hash(source) != digest:
            raise ValueError(f'incomplete or changed shipped auxiliary: {file}')
        budget.check()
    index = json.loads((args.output / 'cache-index.json').read_text())
    sample = torch.load(args.output / 'cache' / next(r['file'] for r in index if r['partition'] == 'validation' and r['view'] == 0), weights_only=False)
    verified = []
    for mode in ('unchanged', 'null', 'temperature'):
        model = copy.deepcopy(base)
        selected_ck = ck
        if mode != 'unchanged':
            selected_ck = torch.load(args.output / f'{mode}-selected.pt', weights_only=False)
            install_query_attention(model, mode)
            model.load_state_dict(selected_ck['model'], strict=True)
        directory = args.output / f'{mode}-bundle'
        directory.mkdir(exist_ok=True)
        fp32 = directory / 'fp32.onnx'
        states = export_onnx(model, str(fp32), manifest['chunk'], manifest['leftContext'])
        # Replay cached suffix is an independent oracle for the deployed suffix.
        # Full streaming parity requires original fbank inputs, retained below by
        # rebuilding this training-only recording from its admitted audio.
        data = next(r for r in training_records() if r['key'] == next(i['key'] for i in index if i['partition'] == 'validation' and i['view'] == 0))
        feats = acoustic_features(data, 0, ck['cfg'], None)
        expected = replay(model, sample, manifest).detach().numpy()[0]
        got = stream_onnx(str(fp32), feats, states, manifest['chunk'])[:sample['frames']]
        import numpy as np
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)
        quantize_dynamic(str(fp32), str(directory / 'model.onnx'), op_types_to_quantize=['MatMul'], weight_type=QuantType.QInt8)
        onnx.checker.check_model(str(directory / 'model.onnx'))
        deployed = stream_onnx(str(directory / 'model.onnx'), feats, states, manifest['chunk'])[:sample['frames']]
        if deployed.shape != expected.shape or not np.isfinite(deployed).all():
            raise ValueError('exported int8 posterior clock or numeric contract differs')
        if mode == 'unchanged':
            supplied = stream_onnx(str(args.baseline_dir / manifest['file']), feats, manifest['states'], manifest['chunk'])[:sample['frames']]
            np.testing.assert_allclose(deployed, supplied, rtol=1e-4, atol=1e-4,
                err_msg='unchanged checkpoint/export does not reproduce the supplied listener')
        m = make_manifest(f'query-attention-{mode}', states, manifest['chunk'], manifest['leftContext'], str(directory / 'model.onnx'), selected_ck)
        m['format'] = 'onnx-int8'
        for field in ('inventory', 'tokens', 'blank', 'sampleRate', 'frameMs', 'window', 'shift', 'states'):
            if m[field] != manifest[field]:
                raise ValueError(f'exported primary listener contract differs: {field}')
        for file, _, _ in assets:
            shutil.copy2(args.auxiliary_dir / file, directory / file)
        atomic_json(directory / 'manifest.json', m)
        atomic_json(directory / 'parity.json', {'fp32_max_abs_logprob_difference': float(np.abs(got - expected).max()),
                    'int8_top1_agreement': float((deployed.argmax(-1) == expected.argmax(-1)).mean()),
                    'auxiliary': [{'file': f, 'sha256': file_hash(directory / f)} for f, _, _ in assets]})
        verified.append({'path': str(directory), 'model_sha256': m['sha256'],
                         'manifest_sha256': file_hash(directory / 'manifest.json')})
        budget.check()
    atomic_json(args.output / 'verified-bundles.json', {'bundles': verified})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('cache', 'fit', 'export'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, default=Path('/lab/data/models/spliced-clean-int8'))
    parser.add_argument('--auxiliary-dir', type=Path)
    args = parser.parse_args()
    scratch = Path('/scratch') / os.environ['AUTOLAB_EXP']
    if not args.output.resolve().is_relative_to(scratch.resolve()):
        raise ValueError('all experiment artifacts must stay in its authorized scratch root')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    budget = Budget(args.output)
    try:
        budget.check()
        model, ck, manifest = load_base(args.checkpoint, args.baseline_dir)
        globals()[args.command](args, model, ck, manifest, budget)
    finally:
        budget.check()


if __name__ == '__main__':
    main()

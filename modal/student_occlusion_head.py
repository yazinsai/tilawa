"""Matched ordinary and occlusion-censored refits of the frozen shipped CTC head.

prepare caches paired views and a frozen speaker partition; fit selects one
common update count using unmasked training-speaker validation only; export
replaces only classifier constants and verifies the inherited SDK bundle.
No practice recordings, incidents or scores enter this utility.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil

from student_occlusion import spec_augment, output_masks, censored_ctc_loss, imitation_loss

ROOT = Path(__file__).resolve().parents[1]
RECIPE = dict(seed=831, original_limit=512, pairs=2, epochs=12, batch=8,
              lr=.0001, anchor=2., max_seconds=30.)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False))
    temp.replace(path)


def reserved_speakers(rows):
    speakers = sorted({r['speaker'] for r in rows}, key=stable)
    if len(speakers) < 2:
        raise ValueError('At least two training speakers required per source')
    return set(speakers[:max(1, math.ceil(len(speakers)/5))])


def linear_head(graph, vocab):
    """Identify the shipped classifier, requiring an unambiguous known layout."""
    import numpy as np
    from onnx import numpy_helper

    nodes = [n for n in graph.graph.node if 'ctc_output' in n.name and n.op_type in ('MatMul', 'MatMulInteger')]
    if len(nodes) != 1:
        raise ValueError('Cannot identify exactly one shipped linear phone head')
    node = nodes[0]
    initial = {t.name: numpy_helper.to_array(t) for t in graph.graph.initializer}
    weight_name = node.input[1]
    if weight_name not in initial:
        raise ValueError('Classifier weight is not a constant tensor')
    weight = initial[weight_name]
    quantized = node.op_type == 'MatMulInteger'
    scale_name = None
    zero_name = None
    if quantized:
        base = weight_name.removesuffix('_quantized')
        scale_name, zero_name = base+'_scale', base+'_zero_point'
        if scale_name not in initial or zero_name not in initial or node.input[3] != zero_name:
            raise ValueError('Unrecognized classifier quantization layout')
        weight = (weight.astype(np.float32)-initial[zero_name])*initial[scale_name]
    biases = [(k, v) for k, v in initial.items() if 'ctc_output' in k and k.endswith('bias')]
    if weight.ndim != 2 or weight.shape[1] != vocab or len(biases) != 1 or biases[0][1].shape != (vocab,):
        raise ValueError('Classifier geometry differs from the phone inventory')
    feature = node.input[0]
    if quantized:
        producer = next((n for n in graph.graph.node if feature in n.output), None)
        if producer is None or producer.op_type != 'DynamicQuantizeLinear':
            raise ValueError('Unrecognized classifier activation quantization')
        feature = producer.input[0]
    return dict(weight=weight, bias=biases[0][1], weight_name=weight_name,
                bias_name=biases[0][0], scale_name=scale_name, zero_name=zero_name,
                quantized=quantized, feature=feature,
                quantized_input=node.input[0] if quantized else None,
                activation_scale=producer.output[1] if quantized else None,
                activation_zero=node.input[2] if quantized else None)


def feature_runtime(model, out):
    import numpy as np
    import onnx
    from onnx import helper, TensorProto
    import onnxruntime as ort

    manifest = json.loads((model/'manifest.json').read_text())
    path = model/manifest['file']
    if digest(path) != manifest['sha256']:
        raise ValueError('Shipped primary checksum mismatch')
    graph = onnx.load(str(path))
    head = linear_head(graph, len(manifest['tokens'])+1)
    if head['quantized']:
        # Cache the actual frozen activations consumed by the integer head, not
        # the unrounded pre-quantizer tensor. This isolates weight refitting from
        # an additional change to activation quantization in the two controls.
        name = 'occlusion_head_features'
        if any(name in n.output for n in graph.graph.node):
            raise ValueError('Feature debug output already exists')
        graph.graph.node.append(helper.make_node('DequantizeLinear',
            [head['quantized_input'], head['activation_scale'], head['activation_zero']], [name], name=name))
        head['feature'] = name
    graph.graph.output.append(helper.make_tensor_value_info(head['feature'], TensorProto.FLOAT,
                                                          [1, None, head['weight'].shape[0]]))
    onnx.checker.check_model(graph)
    feature_path = out/'encoder-features.onnx'
    onnx.save(graph, str(feature_path))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    original = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])
    extractor = ort.InferenceSession(str(feature_path), opts, providers=['CPUExecutionProvider'])
    st = {s['name']: np.zeros(s['shape'], dtype=np.int64 if s['dtype'] == 'int64' else np.float32)
          for s in manifest['states']}
    names = [v.name for v in original.get_outputs()]
    rng = np.random.default_rng(RECIPE['seed'])
    for _ in range(8):
        feed = dict(st, x=rng.normal(size=(1, manifest['window'], 80)).astype('float32'))
        a, b = original.run(names, feed), extractor.run(names, feed)
        for name, x, y in zip(names, a, b):
            np.testing.assert_array_equal(x, y, err_msg='Feature capture changes '+name)
        st = {s['name']: b[names.index('new_'+s['name'])] for s in manifest['states']}
    return manifest, head, extractor


def stream_features(runtime, manifest, feats, feature):
    import numpy as np

    state = {s['name']: np.zeros(s['shape'], dtype=np.int64 if s['dtype'] == 'int64' else np.float32)
             for s in manifest['states']}
    outputs = ['log_probs', *['new_'+s['name'] for s in manifest['states']], feature]
    window, shift = manifest['window'], manifest['shift']
    n = len(feats)
    need = ((n-7)//2+1)//2
    if need <= 0:
        raise ValueError('Recording is too short for complete streaming output')
    windows = max(1, math.ceil(need/(manifest['chunk']//2)))
    total = window+(windows-1)*shift
    padded = np.concatenate([feats, np.full((max(0, total-n), 80), manifest['fbank']['padValue'], dtype='float32')])
    features, log_probs = [], []
    for at in range(0, total-window+1, shift):
        found = runtime.run(outputs, dict(state, x=padded[None, at:at+window]))
        log_probs.append(found[0][0])
        features.append(found[-1][0])
        state = {s['name']: value for s, value in zip(manifest['states'], found[1:-1])}
    # The SDK pads input for the last call, but emits only outputsFor(n) rows.
    # Training on the extra rows would teach from audio absent in that stream.
    return np.concatenate(features)[:need], np.concatenate(log_probs)[:need]


def clean_target(pronunciation, tokens):
    """Collapse held-phone runs, as the shipped spliced CTC recipe does."""
    ids = {p: i+1 for i, p in enumerate(tokens)}
    return [ids[p] for i, p in enumerate(pronunciation) if i == 0 or p != pronunciation[i-1]]


def replace_head_parameters(graph, weight, bias):
    """Replace classifier constants; all encoder nodes/tensors stay identical."""
    import numpy as np
    from onnx import numpy_helper

    head = linear_head(graph, len(bias))
    weight, bias = np.asarray(weight, dtype='float32'), np.asarray(bias, dtype='float32')
    if weight.shape != head['weight'].shape or bias.shape != head['bias'].shape or \
            not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError('Refitted head has invalid geometry or parameters')
    changed = {head['bias_name']: bias}
    if head['quantized']:
        scale = np.asarray(max(float(np.max(np.abs(weight)))/127, 1e-8), dtype='float32')
        changed.update({head['weight_name']: np.clip(np.round(weight/scale), -127, 127).astype('int8'),
                        head['scale_name']: scale, head['zero_name']: np.asarray(0, dtype='int8')})
    else:
        changed[head['weight_name']] = weight
    before = {t.name: t.SerializeToString() for t in graph.graph.initializer}
    for i, tensor in enumerate(graph.graph.initializer):
        if tensor.name in changed:
            graph.graph.initializer[i].CopyFrom(numpy_helper.from_array(changed[tensor.name], tensor.name))
    for tensor in graph.graph.initializer:
        if tensor.name not in changed and tensor.SerializeToString() != before[tensor.name]:
            raise ValueError('Export changed an encoder tensor')
    return sorted(changed)


def authorized_records(original=None):
    """Admit complete clean takes; reserve speakers before feature extraction.

    Optional original s1 shards must be supplied under LAB_DATA. No fetching,
    practice audio, or mistake scripts are used as transcript supervision.
    """
    import soundfile as sf

    data_root = Path(os.environ['LAB_DATA']).resolve()
    help_root = data_root/'help-train'
    labels = [json.loads(s) for s in (help_root/'labels.jsonl').read_text().splitlines() if s.strip()]
    banned_help = {str(l['speaker']) for l in labels if l.get('split') != 'train'}
    rows = []
    for label in labels:
        if (label.get('split') != 'train' or str(label['speaker']) in banned_help or
                label.get('kind') != 'none' or label.get('exclude') or label.get('excluded') or
                label.get('scored') is False or any(label.get(k) for k in ('events', 'non_errors', 'unsupported'))):
            continue
        path = (help_root/label['wav']).resolve()
        if not path.is_relative_to(help_root) or not path.is_file():
            raise ValueError('Authorized help waveform is missing or escapes its source')
        if not 0 < sf.info(path).duration <= RECIPE['max_seconds']:
            continue
        rows.append(dict(key='help/'+label['id'], source='help', file=str(path),
                         speaker='help/'+str(label['speaker']), ayahs=label['ayahs']))
    reserved = reserved_speakers(rows)
    banned = {'help/'+s for s in banned_help}
    if original is not None:
        import pyarrow.parquet as pq

        original = Path(original).resolve()
        if not original.is_relative_to(data_root) or not original.is_dir():
            raise ValueError('Original shards must be a supplied authorized LAB_DATA directory')
        excluded = {tuple(line.rsplit(':', 1)) for line in
                    (ROOT/'docs/benchmarks/gold-per-v1-exclude.txt').read_text().split()}
        meta = []
        for path in sorted(original.glob('*.parquet')):
            cols = ['shard', 'row', 'cluster', 'bucket', 'split', 'n_samples']
            if not set(cols).issubset(pq.read_schema(path).names):
                raise ValueError('Original shards lack training-speaker provenance')
            for index, r in enumerate(pq.read_table(path, columns=cols).to_pylist()):
                meta.append(dict(r, file=str(path), index=index, speaker=f"original/{r['cluster']}",
                                 key=f"original/{r['shard']}:{r['row']}", source='original'))
        banned_original = {r['speaker'] for r in meta if r['split'] != 'train' or
                           (r['shard'], str(r['row'])) in excluded}
        ordinary = [r for r in meta if r['split'] == 'train' and r['speaker'] not in banned_original
                    and r['cluster'] is not None and str(r['cluster']) != '-1' and r['bucket'] == 'clean-match'
                    and 0 < r['n_samples'] <= RECIPE['max_seconds']*16000]
        # Sample per partition so a finite limit cannot remove validation.
        held = reserved_speakers(ordinary)
        for subset in (False, True):
            chosen = sorted([r for r in ordinary if (r['speaker'] in held) == subset],
                            key=lambda r: stable(r['key']))[:RECIPE['original_limit']]
            rows.extend(chosen)
        reserved |= held
        banned |= banned_original
    for r in rows:
        r['partition'] = 'validation' if r['speaker'] in reserved else 'train'
    if len({r['key'] for r in rows}) != len(rows):
        raise ValueError('Duplicate training recording keys')
    return rows, sorted(banned), digest(help_root/'labels.jsonl')


def prepare(model, out, original=None):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import torchaudio

    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists():
        raise ValueError('Frozen plan already exists; use a new output directory')
    records, excluded, labels_sha = authorized_records(original)
    manifest, head, runtime = feature_runtime(model, out)
    reference = json.loads((ROOT/'packages/core/assets/ref-v2.json').read_text())
    tokens = reference['inventory']['tokens']
    if (manifest['tokens'] != tokens or manifest['blank'] != 0 or manifest['frameMs'] != 40 or
            manifest['sampleRate'] != 16000 or manifest['fbank']['mels'] != 80):
        raise ValueError('Shipped inventory, clock or frontend differs from this training recipe')
    np.savez(out/'initial-head.npz', weight=head['weight'].T, bias=head['bias'])
    ayahs = {f"{a['s']}:{a['a']}": a['p'] for a in reference['ayahs']}
    cache = out/'features'
    cache.mkdir(exist_ok=True)
    pairs = []
    audit = dict(masked_frames=0, hidden_ctc_frames=0, masked_speech_frames=0, masked_consonant_frames=0,
                 train_recordings=0, validation_recordings=0)
    torch.set_num_threads(2)
    for r in records:
        if r['source'] == 'original':
            import pyarrow.parquet as pq
            # This source carries the shipped transcript/teacher-edge recipe.
            import sys
            sys.path.insert(0, str(ROOT/'tools'))
            from student import spliced_target
            value = pq.read_table(r['file'], columns=['flac', 'ref', 'hyp']).slice(r['index'], 1).to_pylist()[0]
            wave, sr = sf.read(io.BytesIO(value['flac']), dtype='float32')
            target = clean_target(spliced_target(value['ref'], value['hyp']), tokens)
        else:
            wave, sr = sf.read(r['file'], dtype='float32')
            target = clean_target(''.join(ayahs[a] for a in r['ayahs']), tokens)
        if wave.ndim == 2:
            wave = wave.mean(1)
        if sr != 16000:
            divisor = math.gcd(sr, 16000)
            wave = resample_poly(wave, 16000//divisor, sr//divisor).astype('float32')
        if not target:
            raise ValueError('Clean recording has no authorized pronunciation')
        r['waveform_sha256'] = hashlib.sha256(wave.tobytes()).hexdigest()
        feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(wave).unsqueeze(0), num_mel_bins=80,
             sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400)
        x, teacher = stream_features(runtime, manifest, feats.numpy(), head['feature'])
        if len(target) > len(x):
            raise ValueError('Complete recording cannot support its CTC transcript')
        # Whole-recording speech and pauses are retained in every unmasked arm.
        # The clean frozen student is the preservation teacher; the matching
        # ordinary arm imitates it even where SpecAugment removes observations.
        occ = transcript_occupancy(teacher, target) if r['partition'] == 'train' else None
        for view in range(RECIPE['pairs']):
            rng = np.random.default_rng(int(stable(f"{r['key']}:{view}:{RECIPE['seed']}")[:16], 16))
            masked, source_mask = spec_augment(feats, rng, return_mask=True)
            mx, _ = stream_features(runtime, manifest, masked.numpy(), head['feature'])
            hidden, touched = output_masks(source_mask, len(mx))
            if len(mx) != len(x):
                raise ValueError('Time masking changed the native output clock')
            file = cache/(stable(f"{r['key']}:{view}")+'.npz')
            np.savez_compressed(file, unmasked=x, masked=mx, teacher=teacher, target=np.asarray(target, dtype=np.int64),
                                source_mask=source_mask.numpy(), hidden=hidden.numpy(), touched=touched.numpy())
            pairs.append(dict(recording=r['key'], speaker=r['speaker'], partition=r['partition'],
                              file=str(file), sha256=digest(file)))
            if r['partition'] == 'train':
                audit['masked_frames'] += int(source_mask.sum())
                audit['hidden_ctc_frames'] += int(hidden.sum())
                # Transcript-constrained occupancies distinguish speech from
                # genuine pauses without treating argmax as planted truth.
                audit['masked_speech_frames'] += int((hidden.numpy() & (occ[:, 1:].sum(1) > .5)).sum())
                consonants = [i+1 for i, c in enumerate(tokens) if c in 'ءبتثجحخدذرزسشصضطظعغفقكلمنهويڇں۾']
                audit['masked_consonant_frames'] += int((hidden.numpy() & (occ[:, consonants].sum(1) > .5)).sum())
        audit[r['partition']+'_recordings'] += 1
        print(json.dumps(dict(cached=r['key'], partition=r['partition'])), flush=True)
    plan = dict(schema=1, recipe=RECIPE, records=records, pairs=pairs, audit=audit,
                excluded_speakers=excluded, help_labels_sha256=labels_sha, primary_sha256=manifest['sha256'],
                feature_graph_sha256=digest(out/'encoder-features.onnx'), initial_head_sha256=digest(out/'initial-head.npz'))
    write_json(out/'plan.json', plan)
    if not audit['masked_speech_frames']:
        raise ValueError('The bounded recipe did not mask any transcript-supervised speech')


def transcript_occupancy(teacher, target):
    """Exact ordinary CTC occupancy for the premise audit, never a training cut."""
    import torch
    import torch.nn.functional as F

    lp = torch.tensor(teacher, requires_grad=True)
    nll = F.ctc_loss(lp[:, None], torch.tensor(target), torch.tensor([len(lp)]),
                     torch.tensor([len(target)]), reduction='sum', zero_infinity=False)
    if not torch.isfinite(nll):
        raise ValueError('Unavailable transcript-constrained premise audit')
    # PyTorch returns softmax minus occupancy for its log-probability input.
    grad, = torch.autograd.grad(nll, lp)
    return (lp.exp()-grad).detach().numpy()


def pair_loss(head, data, censored):
    import torch

    clean = head(data['unmasked']).log_softmax(-1)
    masked = head(data['masked']).log_softmax(-1)
    no_mask = torch.zeros(len(clean), dtype=torch.bool)
    clean_loss = censored_ctc_loss(clean, data['target'], no_mask)/len(clean)
    masked_loss = censored_ctc_loss(masked, data['target'], data['hidden'] if censored else no_mask)/len(masked)
    return .5*(clean_loss + masked_loss + RECIPE['anchor']*(
        imitation_loss(clean, data['teacher'], no_mask) +
        imitation_loss(masked, data['teacher'], data['touched'] if censored else no_mask)))


def validation_loss(head, data):
    """Same full-transcript, unmasked evaluation for both arms."""
    import torch

    lp = head(data['unmasked']).log_softmax(-1)
    observed = torch.zeros(len(lp), dtype=torch.bool)
    return censored_ctc_loss(lp, data['target'], observed)/len(lp) + RECIPE['anchor']*imitation_loss(lp, data['teacher'], observed)


def read_plan(out):
    plan = json.loads((out/'plan.json').read_text())
    if plan['recipe'] != RECIPE:
        raise ValueError('Frozen training settings changed')
    for file, key in [('initial-head.npz', 'initial_head_sha256'), ('encoder-features.onnx', 'feature_graph_sha256')]:
        if digest(out/file) != plan[key]:
            raise ValueError('Frozen encoder or initial head changed')
    for pair in plan['pairs']:
        if digest(pair['file']) != pair['sha256']:
            raise ValueError('Frozen paired features changed')
    if not plan['audit']['masked_speech_frames']:
        raise ValueError('No masked supervised speech in the frozen plan')
    return plan


def fit(out):
    import copy
    from functools import lru_cache
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
    arms = dict(ordinary=head, censored=copy.deepcopy(head))
    optimizers = {k: torch.optim.SGD(h.parameters(), lr=RECIPE['lr']) for k, h in arms.items()}
    train = [v for v in plan['pairs'] if v['partition'] == 'train']
    validation = [v for v in plan['pairs'] if v['partition'] == 'validation']
    if not train or not validation or {v['speaker'] for v in train} & {v['speaker'] for v in validation}:
        raise ValueError('Missing or leaking training-speaker partition')

    @lru_cache(maxsize=32)
    def cached(path):
        with np.load(path) as a:
            return {k: torch.from_numpy(a[k].copy()) for k in ('unmasked', 'masked', 'teacher', 'target', 'hidden', 'touched')}

    history = []
    for epoch in range(RECIPE['epochs']):
        order = np.random.default_rng(RECIPE['seed']+epoch).permutation(len(train))
        for start in range(0, len(order), RECIPE['batch']):
            batch = order[start:start+RECIPE['batch']]
            for name, h in arms.items():
                opt = optimizers[name]
                opt.zero_grad()
                for i in batch:
                    loss = pair_loss(h, cached(train[i]['file']), name == 'censored')
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite head loss')
                    (loss/len(batch)).backward()
                if any(not torch.isfinite(p.grad).all() for p in h.parameters()):
                    raise ValueError('Non-finite head gradient')
                opt.step()
        losses = {}
        with torch.no_grad():
            for name, h in arms.items():
                speakers = {}
                for v in validation:
                    loss = float(validation_loss(h, cached(v['file'])))
                    if not math.isfinite(loss):
                        raise ValueError('Non-finite validation loss')
                    speakers.setdefault(v['speaker'], []).append(loss)
                losses[name] = float(np.mean([np.mean(values) for values in speakers.values()]))
                np.savez(out/f'{name}-{epoch+1}.npz', weight=h.weight.detach().numpy(), bias=h.bias.detach().numpy())
        history.append(dict(epoch=epoch+1, validation=losses, updates=(epoch+1)*math.ceil(len(train)/RECIPE['batch'])))
        print(json.dumps(history[-1]), flush=True)
    # Choose one common checkpoint step on speaker-grouped validation only.
    selected = min(history, key=lambda r: (r['validation']['censored'], r['epoch']))
    paths = {name: out/f"{name}-{selected['epoch']}.npz" for name in arms}
    write_json(out/'selection.json', dict(schema=1, plan_sha256=digest(out/'plan.json'), selected=selected, history=history,
               arms={name: dict(path=str(path), sha256=digest(path)) for name, path in paths.items()},
               identical_updates=True, identical_minibatches=True))


def export(model, out):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import re

    plan = read_plan(out)
    selection = json.loads((out/'selection.json').read_text())
    if selection['plan_sha256'] != digest(out/'plan.json'):
        raise ValueError('Selection differs from frozen plan')
    manifest = json.loads((model/'manifest.json').read_text())
    if manifest['sha256'] != plan['primary_sha256'] or digest(model/manifest['file']) != manifest['sha256']:
        raise ValueError('Installed primary differs from frozen encoder')
    assets = re.findall(r'file:\s*"([^"]+)"\s*,\s*bytes:\s*(\d+)\s*,\s*sha256:\s*"([0-9a-f]{64})"',
                        (ROOT/'packages/sdk/src/registry.ts').read_text())
    if not assets:
        raise ValueError('Cannot establish inherited SDK auxiliary declarations')
    for file, size, sha in assets:
        path = model/file
        if not path.is_file() or path.stat().st_size != int(size) or digest(path) != sha:
            raise ValueError('Inherited SDK asset unavailable or corrupt: '+file)
    reports = {}
    for name, checkpoint in selection['arms'].items():
        if digest(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Selected head changed')
        graph = onnx.load(str(model/manifest['file']))
        original_nodes = [n.SerializeToString() for n in graph.graph.node]
        with np.load(checkpoint['path']) as ck:
            changed = replace_head_parameters(graph, ck['weight'].T, ck['bias'])
        if original_nodes != [n.SerializeToString() for n in graph.graph.node]:
            raise ValueError('Head export changed encoder computation')
        directory = out/(name+'-bundle')
        directory.mkdir(exist_ok=True)
        path = directory/manifest['file']
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        a = ort.InferenceSession(str(model/manifest['file']), options, providers=['CPUExecutionProvider'])
        b = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        state = {s['name']: np.zeros(s['shape'], dtype=np.int64 if s['dtype'] == 'int64' else np.float32)
                 for s in manifest['states']}
        outputs = [v.name for v in a.get_outputs()]
        rng = np.random.default_rng(RECIPE['seed'])
        for _ in range(8):
            feed = dict(state, x=rng.normal(size=(1, manifest['window'], 80)).astype('float32'))
            original, candidate = a.run(outputs, feed), b.run(outputs, feed)
            if original[0].shape != candidate[0].shape or not np.isfinite(candidate[0]).all():
                raise ValueError('Invalid exported phone output')
            for output, x, y in zip(outputs[1:], original[1:], candidate[1:]):
                np.testing.assert_array_equal(x, y, err_msg='Head export changed '+output)
            state = {s['name']: value for s, value in zip(manifest['states'], candidate[1:])}
        for file, _, _ in assets:
            shutil.copyfile(model/file, directory/file)
        write_json(directory/'manifest.json', dict(manifest, id='occlusion-'+name, bytes=path.stat().st_size, sha256=digest(path)))
        reports[name] = dict(path=str(directory), model_sha256=digest(path), changed_tensors=changed,
                             cache_parity_windows=8, inherited_assets={f: digest(directory/f) for f, _, _ in assets})
    write_json(out/'export-report.json', reports)
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'fit', 'export'))
    p.add_argument('--model', type=Path)
    p.add_argument('--original', type=Path, help='Optional authorized original s1 shards under LAB_DATA')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.command != 'fit' and args.model is None:
        p.error('prepare/export require --model')
    if args.command == 'prepare':
        prepare(args.model, args.out, args.original)
    elif args.command == 'fit':
        fit(args.out)
    else:
        export(args.model, args.out)

"""Frozen streaming feature capture and phone-head-only bundle export.

Reused from exp/listener-noise-aware-teacher-loss for emission-erasure training.
Only feature capture, speaker enrollment and head replacement are reused; no
observation channel or cross-speaker weighting is used. No scoring inputs read.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


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


def has_clean_transcript(label):
    """Only clean takes without labelled repeats admit canonical supervision."""
    return (label.get('kind') == 'none' and label.get('scored') is not False and
            not any(label.get(k) for k in ('events', 'non_errors', 'unsupported')))


def authorized_records(original_limit, max_seconds):
    """Keep source provenance; remove held-out/gold speaker clusters in full."""
    import pyarrow.parquet as pq
    import soundfile as sf

    files = sorted(Path('/student/s1').glob('*.parquet'))
    if not files:
        raise FileNotFoundError('Authorized original training recordings are unavailable')
    excluded = {tuple(line.rsplit(':', 1)) for line in
                (ROOT/'docs/benchmarks/gold-per-v1-exclude.txt').read_text().split()}
    meta = []
    for path in files:
        cols = ['shard', 'row', 'cluster', 'bucket', 'split', 'n_samples']
        if not set(cols).issubset(pq.read_schema(path).names):
            raise ValueError('Original recordings lack training-speaker provenance')
        for index, r in enumerate(pq.read_table(path, columns=cols).to_pylist()):
            meta.append(dict(r, file=str(path), index=index, speaker=f"original/{r['cluster']}",
                             key=f"original/{r['shard']}:{r['row']}", source='original'))
    banned = {r['speaker'] for r in meta if r['split'] != 'train' or
              (r['shard'], str(r['row'])) in excluded}
    ordinary = [r for r in meta if r['split'] == 'train' and r['speaker'] not in banned
                and r['cluster'] is not None and str(r['cluster']) != '-1'
                and r['bucket'] == 'clean-match' and 0 < r['n_samples'] <= max_seconds*16000]
    reserved_original = reserved_speakers(ordinary)
    # Sampling is frozen before any acoustic comparisons, with no duplication
    # enrichment. Both arms see exactly this same finite training population.
    ordinary = sorted(ordinary, key=lambda r: stable(r['key']))[:original_limit]

    help_root = Path(os.environ['LAB_DATA'])/'help-train'
    labels = [json.loads(s) for s in (help_root/'labels.jsonl').read_text().splitlines() if s.strip()]
    banned_help = {str(l['speaker']) for l in labels if l.get('split') != 'train' or
                   l.get('exclude') or l.get('excluded')}
    help_rows = []
    for label in labels:
        if label.get('split') != 'train' or str(label['speaker']) in banned_help or not has_clean_transcript(label):
            continue
        path = (help_root/label['wav']).resolve()
        if not path.is_relative_to(help_root.resolve()) or not path.is_file():
            raise FileNotFoundError('Authorized help waveform is missing or escapes its source')
        if not 0 < sf.info(path).duration <= max_seconds:
            continue
        help_rows.append(dict(key=f"help/{label['id']}", source='help', file=str(path),
                              speaker=f"help/{label['speaker']}", ayahs=label['ayahs']))
    reserved = reserved_original | reserved_speakers(help_rows)
    rows = ordinary + help_rows
    if len({r['key'] for r in rows}) != len(rows):
        raise ValueError('Authorized recording keys are not unique')
    for r in rows:
        r['partition'] = 'validation' if r['speaker'] in reserved else 'train'
    for source in ('original', 'help'):
        for partition in ('train', 'validation'):
            if not any(r['source'] == source and r['partition'] == partition for r in rows):
                raise ValueError('Required training-speaker partition has no clean recordings')
    return rows, sorted(banned | {f'help/{s}' for s in banned_help}), digest(help_root/'labels.jsonl')


def training_helpers():
    """Reuse the shipped augmentation functions without initializing a Modal app."""
    tree = ast.parse((ROOT/'modal/student_train.py').read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                n.name in ('pick', 'reverb', 'spec_augment')]
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'student_train.py', 'exec'), namespace)
    return namespace


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
        if (weight.dtype != np.int8 or initial[scale_name].shape not in ((), (vocab,)) or
                initial[zero_name].shape != initial[scale_name].shape):
            raise ValueError('Unsupported classifier weight quantization geometry')
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
                scale_shape=initial[scale_name].shape if quantized else None,
                quantized=quantized, feature=feature,
                quantized_input=node.input[0] if quantized else None,
                activation_scale=producer.output[1] if quantized else None,
                activation_zero=node.input[2] if quantized else None)


def feature_runtime(model, out, seed=831):
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
        name = 'refit_head_features'
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
    rng = np.random.default_rng(seed)
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
        # Retain the shipped per-tensor or per-output-channel quantization
        # geometry; the experiment changes learned head weights only.
        maximum = np.max(np.abs(weight), axis=0) if head['scale_shape'] else np.max(np.abs(weight))
        scale = np.asarray(np.maximum(maximum/127, 1e-8), dtype='float32')
        changed.update({head['weight_name']: np.clip(np.round(weight/scale), -127, 127).astype('int8'),
                        head['scale_name']: scale, head['zero_name']: np.zeros(head['scale_shape'], dtype='int8')})
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


def inherited_assets(model):
    """Require the unchanged SDK teacher and every registered external file."""
    import re
    assets = re.findall(r'file:\s*"([^"]+)"\s*,\s*bytes:\s*(\d+)\s*,\s*sha256:\s*"([0-9a-f]{64})"',
                        (ROOT/'packages/sdk/src/registry.ts').read_text())
    if not assets:
        raise ValueError('Cannot establish the inherited SDK auxiliary asset declarations')
    for file, size, sha in assets:
        path = model/file
        if not path.is_file() or path.stat().st_size != int(size) or digest(path) != sha:
            raise ValueError('Inherited SDK asset unavailable or corrupt: '+file)
    return assets


def export(model, out, read_plan, seed=831, experiment='emission-erasure'):
    import numpy as np
    import onnx
    import onnxruntime as ort

    plan = read_plan(out)
    selection = json.loads((out/'selection.json').read_text())
    if selection['plan_sha256'] != digest(out/'plan.json'):
        raise ValueError('Selection does not match frozen training plan')
    manifest = json.loads((model/'manifest.json').read_text())
    if manifest['sha256'] != plan['primary_sha256'] or digest(model/manifest['file']) != manifest['sha256']:
        raise ValueError('Installed primary differs from the frozen encoder')
    # All inherited declared assets must exist and match before exporting either
    # head. Missing files must not silently remove the folded insertion teacher.
    assets = inherited_assets(model)
    reports = {}
    for name, checkpoint in selection['arms'].items():
        if digest(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Frozen selected head changed')
        graph = onnx.load(str(model/manifest['file']))
        ck = np.load(checkpoint['path'])
        weight, bias = ck['weight'].T.astype('float32'), ck['bias'].astype('float32')
        changed = replace_head_parameters(graph, weight, bias)
        effective = linear_head(graph, len(bias))
        # Validate the actual exported quantization on every held-speaker view.
        # This comparison uses only cached frozen features, never practice.
        reconstruction_max = 0.
        for view in plan.get('views', []):
            if view['partition'] != 'validation':
                continue
            with np.load(view['file']) as data:
                x = data['features']
            projected = x @ effective['weight'] + effective['bias']
            intended_weight = weight
            if effective['quantized']:
                maximum = np.max(np.abs(weight), axis=0) if effective['scale_shape'] else np.max(np.abs(weight))
                scale = np.asarray(np.maximum(maximum/127, 1e-8), dtype='float32')
                intended_weight = np.clip(np.round(weight/scale), -127, 127) * scale
            error = float(np.max(np.abs(projected - (x @ intended_weight + bias))))
            reconstruction_max = max(reconstruction_max, error)
            if error > 2e-4:
                raise ValueError('Exported head differs from training-selected quantized parameters')
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
        # Re-run real held-training-speaker streaming features through the whole
        # exported model, not just a NumPy projection of the cached activations.
        # All state comes from the frozen encoder; only the classifier changes.
        held_stream_max = 0.
        held_streams = 0
        for view in plan.get('views', []):
            if view['partition'] != 'validation' or view.get('view', 0) != 0:
                continue
            with np.load(view['file']) as data:
                if 'fbank' not in data:
                    raise ValueError('Missing frontend cache for complete export parity')
                x, feats = data['features'], data['fbank']
            actual, _ = stream_features(b, manifest, feats, 'log_probs')
            logits = x @ effective['weight'] + effective['bias']
            expected = logits - logits.max(-1, keepdims=True)
            expected -= np.log(np.exp(expected).sum(-1, keepdims=True))
            error = float(np.abs(np.exp(actual) - np.exp(expected)).max())
            held_stream_max = max(held_stream_max, error)
            held_streams += 1
            np.testing.assert_allclose(np.exp(actual), np.exp(expected), atol=2e-4, rtol=2e-4)
        states = {s['name']: np.zeros(s['shape'], dtype=np.int64 if s['dtype'] == 'int64' else np.float32)
                  for s in manifest['states']}
        outputs = [v.name for v in a.get_outputs()]
        rng = np.random.default_rng(seed)
        for _ in range(8):
            feeds = dict(states, x=rng.normal(size=(1, manifest['window'], 80)).astype('float32'))
            original, candidate = a.run(outputs, feeds), b.run(outputs, feeds)
            if original[0].shape != candidate[0].shape or not np.isfinite(candidate[0]).all():
                raise ValueError('Export changed the phone output geometry or produced invalid rows')
            for output, x, y in zip(outputs[1:], original[1:], candidate[1:]):
                np.testing.assert_array_equal(x, y, err_msg='Head export changed '+output)
            states = {s['name']: value for s, value in zip(manifest['states'], candidate[1:])}
        for file, _, _ in assets:
            shutil.copyfile(model/file, directory/file)
        for file, _, sha in assets:
            if digest(directory/file) != sha:
                raise ValueError('Exported auxiliary file checksum mismatch: '+file)
        exported = dict(manifest, id=experiment+'-'+name, file=manifest['file'], bytes=path.stat().st_size,
                        sha256=digest(path))
        write_json(directory/'manifest.json', exported)
        reports[name] = dict(path=str(directory), model_sha256=digest(path), changed_tensors=sorted(changed),
                             cache_parity_windows=8, held_speaker_projection_max_error=reconstruction_max,
                             held_speaker_streams=held_streams,
                             held_speaker_stream_max_probability_error=held_stream_max,
                             inherited_assets={f: digest(directory/f) for f, _, _ in assets})
    write_json(out/'export-report.json', reports)
    print(json.dumps(reports), flush=True)

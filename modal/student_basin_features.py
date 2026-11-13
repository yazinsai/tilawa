"""Frozen deployed streaming features for connected nonlinear readouts.

Feature capture and head replacement are adapted from the earlier frozen-head
experiments. No benchmark audio or incident selection is read by this utility.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path

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


def has_clean_transcript(label):
    """Only clean takes without labelled repeats admit canonical supervision."""
    return (label.get('kind') == 'none' and label.get('scored') is not False and
            not any(label.get(k) for k in ('events', 'non_errors', 'unsupported')))


def select_original_records(meta, excluded, original_limit, max_seconds):
    """Exclude full speaker clusters before deterministic recording sampling."""
    banned = {r['speaker'] for r in meta if r['split'] != 'train' or
              (str(r['shard']), str(r['row'])) in excluded}
    ordinary = [r for r in meta if r['split'] == 'train' and r['speaker'] not in banned
                and r['cluster'] is not None and str(r['cluster']) != '-1'
                and r['bucket'] == 'clean-match' and 0 < r['n_samples'] <= max_seconds*16000]
    return sorted(ordinary, key=lambda r: stable(r['key']))[:original_limit], banned


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
    # Sampling is frozen before any acoustic comparisons, with no duplication
    # enrichment. Both arms see exactly this same finite training population.
    ordinary, banned = select_original_records(meta, excluded, original_limit, max_seconds)

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
    rows = ordinary + help_rows
    if len({r['key'] for r in rows}) != len(rows):
        raise ValueError('Authorized recording keys are not unique')
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
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
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

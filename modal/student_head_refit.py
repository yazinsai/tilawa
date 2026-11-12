"""Frozen streaming feature capture and phone-head-only bundle export.

Feature capture and head replacement reuse the prior frozen-head experiments.
No training corpus, benchmark audio or incident selection is read by this module.
"""
import hashlib
import json
import math
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
    windows = max(1, math.ceil(need/(shift//4)))
    total = window+(windows-1)*shift
    padded = np.concatenate([feats, np.full((max(0, total-n), 80), manifest['fbank']['padValue'], dtype='float32')])
    features, log_probs = [], []
    for at in range(0, total-window+1, shift):
        found = runtime.run(outputs, dict(state, x=padded[None, at:at+window]))
        if len(found[0][0]) != shift//4 or len(found[-1][0]) != shift//4:
            raise ValueError('Encoder output rows differ from deployed 40 ms timing')
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


def export(model, out, read_plan, seed=831):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import re

    plan = read_plan(out)
    selection = json.loads((out/'selection.json').read_text())
    if selection['plan_sha256'] != digest(out/'plan.json'):
        raise ValueError('Selection does not match frozen training plan')
    manifest = json.loads((model/'manifest.json').read_text())
    if manifest['sha256'] != plan['primary_sha256'] or digest(model/manifest['file']) != manifest['sha256']:
        raise ValueError('Installed primary differs from the frozen encoder')
    # All inherited declared assets must exist and match before exporting either
    # head. Missing files must not silently remove the folded insertion teacher.
    assets = re.findall(r'file:\s*"([^"]+)"\s*,\s*bytes:\s*(\d+)\s*,\s*sha256:\s*"([0-9a-f]{64})"',
                        (ROOT/'packages/sdk/src/registry.ts').read_text())
    if not assets:
        raise ValueError('Cannot establish the inherited SDK auxiliary asset declarations')
    for file, size, sha in assets:
        path = model/file
        if not path.is_file() or path.stat().st_size != int(size) or digest(path) != sha:
            raise ValueError('Inherited SDK asset unavailable or corrupt: '+file)
    reports = {}
    for name, checkpoint in selection['arms'].items():
        if digest(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Frozen selected head changed')
        graph = onnx.load(str(model/manifest['file']))
        ck = np.load(checkpoint['path'])
        weight, bias = ck['weight'].T.astype('float32'), ck['bias'].astype('float32')
        changed = replace_head_parameters(graph, weight, bias)
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
        exported = dict(manifest, id='sequence-unlikelihood-'+name, file=manifest['file'], bytes=path.stat().st_size,
                        sha256=digest(path))
        write_json(directory/'manifest.json', exported)
        reports[name] = dict(path=str(directory), model_sha256=digest(path), changed_tensors=sorted(changed),
                             cache_parity_windows=8, inherited_assets={f: digest(directory/f) for f, _, _ in assets})
    write_json(out/'export-report.json', reports)
    print(json.dumps(reports), flush=True)

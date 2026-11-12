"""Expose the actual final linear head input in a copy of the deployed ONNX.

Dynamic quantization is part of the input: use dequantized activation codes,
not the pre-quantizer float features. Effective weights include every zero point
and scale. Primary nodes and outputs remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path


def projection(graph):
    import numpy as np
    from onnx import numpy_helper
    nodes = list(graph.graph.node)
    producers = {v: n for n in nodes for v in n.output}
    constants = {v.name: numpy_helper.to_array(v) for v in graph.graph.initializer}
    for n in nodes:
        if n.op_type == 'Constant':
            for a in n.attribute:
                if a.name == 'value':
                    constants[n.output[0]] = numpy_helper.to_array(a.t)
    softmax = next(n for n in nodes if n.op_type == 'LogSoftmax' and 'log_probs' in n.output)
    logits = softmax.input[0]
    ancestors, todo = set(), [logits]
    while todo:
        value = todo.pop()
        if value in ancestors:
            continue
        ancestors.add(value)
        if value in producers:
            todo.extend(producers[value].input)
    # Names identify architecture, not any reader, text or test occurrence.
    heads = [n for n in nodes if 'ctc_output' in n.name and
        any(v in ancestors for v in n.output) and n.op_type in ('MatMul', 'MatMulInteger', 'Gemm')]
    if len(heads) != 1:
        raise ValueError('Cannot identify one final deployed phone projection')
    head = heads[0]
    extra = []
    if head.op_type == 'MatMulInteger':
        from onnx import helper
        quant = producers.get(head.input[0])
        if quant is None or quant.op_type != 'DynamicQuantizeLinear':
            raise ValueError('Unrecognized final activation quantization')
        q, scale, zero = quant.output
        weight = constants[head.input[1]].astype(np.float64)
        weight -= constants[head.input[3]] if len(head.input) > 3 and head.input[3] else 0
        # The integer output is cast and multiplied by activation_scale * weight_scale.
        cast = next(n for n in nodes if n.op_type == 'Cast' and n.input[0] == head.output[0])
        rescale = next(n for n in nodes if n.op_type == 'Mul' and cast.output[0] in n.input)
        scale_name = next(v for v in rescale.input if v != cast.output[0])
        scale_node = producers[scale_name]
        if scale_node.op_type != 'Mul' or scale not in scale_node.input:
            raise ValueError('Unrecognized final weight scaling')
        weight_scale = constants[next(v for v in scale_node.input if v != scale)]
        weight *= weight_scale
        feature = 'multigram_encoder_features'
        extra = [helper.make_node('DequantizeLinear', [q, scale, zero], [feature], name='MultigramEncoderInput')]
    elif head.op_type == 'MatMul':
        feature, weight = head.input[0], constants[head.input[1]].astype(np.float64)
    else:
        attrs = {a.name: a for a in head.attribute}
        if attrs.get('transA') and attrs['transA'].i:
            raise ValueError('Unexpected transposed acoustic input')
        feature = head.input[0]
        weight = constants[head.input[1]].astype(np.float64)
        if attrs.get('transB') and attrs['transB'].i:
            weight = weight.T
        if attrs.get('alpha'):
            weight *= attrs['alpha'].f
    if weight.ndim != 2:
        raise ValueError('Projection is not a matrix')
    return feature, logits, weight, extra


def export(args):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper
    source, out = Path(args.model), Path(args.out)
    manifest = json.loads((source/'manifest.json').read_text())
    path = source/manifest['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Primary checksum mismatch')
    graph = onnx.load(str(path))
    feature, logits, weight, extra = projection(graph)
    graph.graph.node.extend(extra)
    graph.graph.output.extend([
        helper.make_tensor_value_info(feature, TensorProto.FLOAT, [1, None, weight.shape[0]]),
        helper.make_tensor_value_info(logits, TensorProto.FLOAT, [1, None, weight.shape[1]])])
    out.mkdir(parents=True, exist_ok=True)
    extractor = out/'multigram-extractor.onnx'
    onnx.checker.check_model(graph)
    onnx.save(graph, str(extractor))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    # Match the SDK's deployed CPU graph optimization, not a float/unoptimized parent.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    primary = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    diagnostic = ort.InferenceSession(str(extractor), options, providers=['CPUExecutionProvider'])
    feeds = {s['name']: np.zeros(s['shape'], dtype=np.int64 if s['dtype']=='int64' else np.float32)
        for s in manifest['states']}
    rng = np.random.default_rng(1729)
    names = [v.name for v in primary.get_outputs()]
    biases = []
    for _ in range(8):
        feeds['x'] = rng.normal(size=(1, manifest['window'], manifest['fbank']['mels'])).astype(np.float32)
        a = primary.run(names, feeds)
        b = diagnostic.run(names+[feature,logits], feeds)
        for name, x, y in zip(names, a, b):
            np.testing.assert_array_equal(x, y, err_msg='Diagnostic graph changed '+name)
        h, z = b[-2].reshape(-1, weight.shape[0]), b[-1].reshape(-1, weight.shape[1])
        if len(h) != len(z):
            raise ValueError('Features and logits have different acoustic clocks')
        biases.append(z-h.astype(np.float64) @ weight)
        feeds.update({s['name']: b[names.index('new_'+s['name'])] for s in manifest['states']})
    biases = np.concatenate(biases)
    # Verifies effective weight orientation/scales against the deployed graph.
    np.testing.assert_allclose(biases, np.broadcast_to(biases.mean(0), biases.shape), atol=2e-5, rtol=1e-5)
    metadata = dict(schema=1, featureOutput=feature, logitOutput=logits, width=weight.shape[0],
        inventory=manifest['inventory'], tokens=manifest['tokens'], primarySha256=manifest['sha256'],
        parityWindows=8)
    (out/'multigram-metadata.json').write_text(json.dumps(metadata, allow_nan=False))
    print(json.dumps(metadata), flush=True)


def bundle(args):
    """Preserve the complete primary/auxiliary bundle and declare both side assets."""
    import shutil
    source, out = Path(args.model), Path(args.bundle)
    manifest = json.loads((source/'manifest.json').read_text())
    if hashlib.sha256((source/manifest['file']).read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Primary checksum mismatch')
    fitted = json.loads(Path(args.fitted).read_text())
    if fitted['primarySha256'] != manifest['sha256'] or fitted['tokens'] != manifest['tokens']:
        raise ValueError('Readout does not belong to primary')
    shutil.copytree(source,out,dirs_exist_ok=True)
    files = [('multigram-heads.json',Path(args.fitted)),
        ('multigram-extractor.onnx',Path(args.out)/'multigram-extractor.onnx')]
    entries = []
    for name,path in files:
        data = path.read_bytes()
        (out/name).write_bytes(data)
        entries.append(dict(file=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    (out/'multigram-asset.json').write_text(json.dumps(dict(schema=1,files=entries)))
    print(json.dumps(dict(bundle=str(out),files=entries)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--fitted')
    p.add_argument('--bundle')
    args = p.parse_args()
    if args.bundle:
        if not args.fitted: p.error('--bundle requires --fitted')
        bundle(args)
    else:
        export(args)

"""Diagnostic activation outputs added to a COPY of the frozen streaming graph.

No weights, operators, primary outputs or supplied manifests are modified.
Recognizes noninjective primitives and the exported Zipformer Swoosh formula.
Site and regularization selection occur later, using training speakers only.
"""
import argparse
import hashlib
import json
from pathlib import Path


def activation_sites(graph):
    import numpy as np
    from onnx import numpy_helper
    nodes = list(graph.graph.node)
    producers = {v: n for n in nodes for v in n.output}
    consumers = {}
    constants = {v.name: numpy_helper.to_array(v) for v in graph.graph.initializer}
    for n in nodes:
        for v in n.input:
            consumers.setdefault(v, []).append(n)
        if n.op_type == 'Constant':
            for a in n.attribute:
                if a.name == 'value':
                    constants[n.output[0]] = numpy_helper.to_array(a.t)

    def scalar(name):
        x = constants.get(name)
        return float(x.reshape(-1)[0]) if x is not None and x.size == 1 else None

    sites = []
    for n in nodes:
        # Encoder only: exclude CTC probabilities, cache masks and front-end padding.
        if 'encoder' not in n.name or 'ctc_output' in n.name:
            continue
        if n.op_type in ('Relu', 'HardSwish', 'Gelu'):
            sites.append(dict(pre=n.input[0], post=n.output[0], kind=n.op_type, scope=n.name))
        elif n.op_type == 'Mul':
            sigmoid = next((producers.get(v) for v in n.input
                if producers.get(v) is not None and producers[v].op_type == 'Sigmoid'), None)
            if sigmoid is not None and sigmoid.input[0] in n.input:
                sites.append(dict(pre=sigmoid.input[0], post=n.output[0], kind='SiLU', scope=n.name))
        elif n.op_type == 'Clip' and any(scalar(v) is not None for v in n.input[1:]):
            sites.append(dict(pre=n.input[0], post=n.output[0], kind='Clip', scope=n.name))
        elif n.op_type == 'Sub':
            # Swoosh(x) = log(1+exp(x-c)) - beta*x - offset, 0 < beta < 1.
            # Its derivative changes sign, so two inputs can yield the same output.
            log, mul = (producers.get(v) for v in n.input)
            if log is None or mul is None or log.op_type not in ('Log', 'Softplus') or mul.op_type != 'Mul':
                continue
            beta_inputs = [(v, scalar(v)) for v in mul.input]
            beta = next((s for _, s in beta_inputs if s is not None), None)
            pre = next((v for v, s in beta_inputs if s is None), None)
            if beta is None or not 0 < beta < 1 or pre is None:
                continue
            shift = producers.get(log.input[0])
            if log.op_type == 'Log':
                if shift is None or shift.op_type != 'Add':
                    continue
                exp = next((producers.get(v) for v in shift.input if scalar(v) is None), None)
                one = next((scalar(v) for v in shift.input if scalar(v) is not None), None)
                if exp is None or exp.op_type != 'Exp' or one != 1:
                    continue
                shift = producers.get(exp.input[0])
            if shift is None or shift.op_type != 'Sub' or shift.input[0] != pre or scalar(shift.input[1]) is None:
                continue
            post = n.output[0]
            tail = consumers.get(post, [])
            if len(tail) == 1 and tail[0].op_type == 'Sub' and tail[0].input[0] == post and scalar(tail[0].input[1]) is not None:
                post = tail[0].output[0]
            sites.append(dict(pre=pre, post=post, kind='Swoosh', scope=n.name))
    return list({(s['pre'], s['post']): s for s in sites}.values())


def export(args):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper
    source, out = Path(args.model), Path(args.out)
    manifest = json.loads((source/'manifest.json').read_text())
    path = source/manifest['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Primary model checksum mismatch')
    graph = onnx.load(str(path))
    inferred = onnx.shape_inference.infer_shapes(graph)
    values = {v.name: v for v in list(inferred.graph.value_info)+list(inferred.graph.output)}
    sites = []
    for site in activation_sites(graph):
        pair = [values.get(site[k]) for k in ('pre', 'post')]
        if any(v is None or v.type.tensor_type.elem_type != TensorProto.FLOAT for v in pair):
            continue
        shapes = [[d.dim_value for d in v.type.tensor_type.shape.dim] for v in pair]
        if shapes[0] != shapes[1] or len(shapes[0]) != 3 or shapes[0][1] != 1 or not shapes[0][-1]:
            continue
        site['width'] = shapes[0][-1]
        sites.append(site)
    if not sites:
        raise ValueError('No identifiable float non-invertible encoder activation pairs')
    # Select architectural candidates AFTER validating their time-major layout.
    if len(sites) > 6:
        sites = [sites[i] for i in np.linspace(0,len(sites)-1,6).round().astype(int)]
    names = {v.name for v in graph.graph.output}
    for site in sites:
        for key in ('pre','post'):
            v = values[site[key]]
            if v.name not in names:
                graph.graph.output.append(v)
                names.add(v.name)
    out.mkdir(parents=True, exist_ok=True)
    target = out/'preactivation-residual.onnx'
    onnx.checker.check_model(graph)
    onnx.save(graph, str(target))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    primary = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    diagnostic = ort.InferenceSession(str(target), options, providers=['CPUExecutionProvider'])
    feeds = {s['name']: np.zeros(s['shape'], dtype=np.int64 if s['dtype']=='int64' else np.float32)
        for s in manifest['states']}
    rng = np.random.default_rng(1729)
    outputs = [v.name for v in primary.get_outputs()]
    for _ in range(8):
        feeds['x'] = rng.normal(size=(1, manifest['window'], manifest['fbank']['mels'])).astype(np.float32)
        a, b = primary.run(outputs, feeds), diagnostic.run(outputs, feeds)
        for name, x, y in zip(outputs, a, b):
            np.testing.assert_array_equal(x, y, err_msg='Diagnostics changed '+name)
        feeds.update({s['name']: b[outputs.index('new_'+s['name'])] for s in manifest['states']})
    for s in sites:
        width = s['width']
        # Independent post-only control has equal phone-model capacity, with the
        # first eight ordinary coordinates shared between the two arms.
        s['postProjection'] = (rng.choice([-1., 1.], (16, width))/np.sqrt(width)).tolist()
        s['preProjection'] = (rng.choice([-1., 1.], (8, width))/np.sqrt(width)).tolist()
        s['nonlinearProjection'] = (rng.choice([-1., 1.], (64, width))/np.sqrt(width)).tolist()
    metadata = dict(schema=1, sites=sites, inventory=manifest['inventory'], tokens=manifest['tokens'],
        primarySha256=manifest['sha256'], dimensions=16)
    (out/'preactivation-residual-metadata.json').write_text(json.dumps(metadata, allow_nan=False))
    print(json.dumps(dict(sites=[{k: s[k] for k in ('scope','kind','width')} for s in sites], parityWindows=8)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    export(p.parse_args())

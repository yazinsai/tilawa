"""Expose the frozen pre-head representation from an existing ONNX export.

Only adds graph outputs; never edits the supplied graph or manifest. Selection
is architectural: the final pre-CTC representation. Verify primary outputs/states
before writing metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path


def export(args):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper, TensorProto
    source = Path(args.model)
    manifest = json.loads((source/'manifest.json').read_text())
    path = source/manifest['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Supplied primary checksum does not match')
    graph = onnx.load(str(path))
    nodes = list(graph.graph.node)
    consumers = {}
    producers = {name: n for n in nodes for name in n.output}
    for n in nodes:
        for name in n.input:
            consumers.setdefault(name, []).append(n)
    head = [n for n in nodes if n.op_type in ('MatMul','MatMulInteger','QLinearMatMul','Gemm') and 'ctc_output' in n.name]
    if len(head) != 1:
        raise ValueError('Cannot identify the final encoder representation')
    final = head[0].input[0]
    activation = producers.get(final)
    if activation is not None and activation.op_type in ('DynamicQuantizeLinear','QuantizeLinear'):
        final = activation.input[0]
    selected = [final]
    inferred = onnx.shape_inference.infer_shapes(graph)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in list(inferred.graph.value_info)+list(inferred.graph.output)}
    widths = []
    for name in selected:
        shape = shapes.get(name)
        if not shape or len(shape) != 3 or not shape[-1]:
            raise ValueError('Encoder shape could not be established: '+name)
        widths.append(shape[-1])
        graph.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [d if d else None for d in shape]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out/'wavecodec-control.onnx'
    onnx.checker.check_model(graph)
    onnx.save(graph, str(target))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    primary = ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    extractor = ort.InferenceSession(str(target),options,providers=['CPUExecutionProvider'])
    feeds = {s['name']: np.zeros(s['shape'],dtype=np.int64 if s['dtype']=='int64' else np.float32) for s in manifest['states']}
    rng = np.random.default_rng(1729)
    for _ in range(8):
        feeds['x'] = rng.normal(size=(1,manifest['window'],manifest['fbank']['mels'])).astype(np.float32)
        names = [v.name for v in primary.get_outputs()]
        a = primary.run(names,feeds)
        b = extractor.run(names,feeds)
        for name,x,y in zip(names,a,b):
            np.testing.assert_array_equal(x,y,err_msg='Extractor changed '+name)
        for s in manifest['states']:
            feeds[s['name']] = b[names.index('new_'+s['name'])]
    control = rng.choice([-1.,1.],size=(16,widths[-1]))/np.sqrt(widths[-1])
    metadata = dict(output=selected[0],controlProjection=control.tolist(),
        inventory=manifest['inventory'],tokens=manifest['tokens'],primarySha256=manifest['sha256'])
    (out/'wavecodec-metadata.json').write_text(json.dumps(metadata))
    print(json.dumps(dict(selected=selected,widths=widths,parityWindows=8)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--out',required=True)
    export(p.parse_args())

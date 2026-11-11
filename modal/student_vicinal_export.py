"""Add one pre-CTC feature output to a separate copy of the frozen encoder.

Never replace or mutate the primary graph/manifest. Verify every primary output
and cache bit-for-bit with graph optimizations disabled before writing metadata.
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
    source, out = Path(args.model), Path(args.out)
    m = json.loads((source / 'manifest.json').read_text())
    path = source / m['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != m['sha256']:
        raise ValueError('Primary checksum mismatch')
    g = onnx.load(str(path))
    heads = [n for n in g.graph.node if n.op_type in ('MatMul', 'MatMulInteger', 'QLinearMatMul', 'Gemm') and 'ctc_output' in n.name]
    if len(heads) != 1:
        raise ValueError('Cannot identify the frozen pre-CTC encoder output')
    feature = heads[0].input[0]
    producer = next((n for n in g.graph.node if feature in n.output), None)
    if producer and producer.op_type in ('DynamicQuantizeLinear', 'QuantizeLinear'):
        feature = producer.input[0]
    inferred = onnx.shape_inference.infer_shapes(g)
    values = list(inferred.graph.value_info) + list(inferred.graph.output)
    shape = next(([d.dim_value for d in v.type.tensor_type.shape.dim] for v in values if v.name == feature), None)
    if not shape or len(shape) != 3 or not shape[-1]:
        raise ValueError('Cannot establish native feature shape')
    g.graph.output.append(helper.make_tensor_value_info(feature, TensorProto.FLOAT, [d or None for d in shape]))
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'vicinal-phone.onnx'
    onnx.checker.check_model(g)
    onnx.save(g, str(target))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    a = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    b = ort.InferenceSession(str(target), options, providers=['CPUExecutionProvider'])
    feeds = {s['name']: np.zeros(s['shape'], dtype=np.int64 if s['dtype'] == 'int64' else np.float32) for s in m['states']}
    rng = np.random.default_rng(1729)
    names = [v.name for v in a.get_outputs()]
    for _ in range(8):
        feeds['x'] = rng.normal(size=(1, m['window'], m['fbank']['mels'])).astype(np.float32)
        original, copy = a.run(names, feeds), b.run(names, feeds)
        for name, x, y in zip(names, original, copy):
            np.testing.assert_array_equal(x, y, err_msg='Feature export changed ' + name)
        for s in m['states']:
            feeds[s['name']] = copy[names.index('new_' + s['name'])]
    projection = rng.choice([-1., 1.], size=(32, shape[-1])) / np.sqrt(shape[-1])
    metadata = dict(schema=1, dimensions=32, featureOutput=feature, projection=projection.tolist(),
                    inventory=m['inventory'], tokens=m['tokens'], primarySha256=m['sha256'])
    (out / 'vicinal-phone-metadata.json').write_text(json.dumps(metadata))
    print(json.dumps(dict(feature=feature, width=shape[-1], parityWindows=8)), flush=True)


def bundle(args):
    """Persist all declared assets before any target diagnostic uses the heads."""
    import shutil
    import subprocess
    source, out = Path(args.model), Path(args.out)
    export(args)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, out/path.name)
    labels = [json.loads(line) for line in Path(args.labels).open() if line.strip()]
    excluded = sorted({r['speaker'] for r in labels if r['split'] != 'train'})
    (out/'excluded-speakers.json').write_text(json.dumps(excluded))
    data = out/'vicinal-phone-training.jsonl'
    subprocess.run(['pnpm', '--filter', '@tilawa/sdk', 'exec', 'tsx', 'src/bench/vicinal-phone-cli.ts',
        '--model', str(source), '--extractor', str(out), '--labels', args.labels,
        '--excluded-speakers', str(out/'excluded-speakers.json'), '--out', str(data)], check=True)
    subprocess.run(['python3', 'modal/student_vicinal_phone.py', '--input', str(data),
        '--metadata', str(out/'vicinal-phone-metadata.json'), '--excluded-speakers', str(out/'excluded-speakers.json'),
        '--out', str(out)], check=True)
    check = "import { loadNodeModel } from './src/node.ts'; (async () => { const m = await loadNodeModel(" + json.dumps(str(out)) + "); if (!m.vicinalPhone) throw new Error('Missing declared vicinal scorer'); await m.backend.release(); })();"
    subprocess.run(['pnpm','--filter','@tilawa/sdk','exec','tsx','-e',check],check=True)
    print(json.dumps(dict(bundle=str(out),normalSdkLoadVerified=True)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--labels', help='None-labelled training speakers and real substitution preservation takes')
    args = p.parse_args()
    bundle(args) if args.labels else export(args)

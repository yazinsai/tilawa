"""Expose the shipped pre-CTC feature tensor without modifying primary assets.

Export a separate graph and verify all primary outputs and streaming caches.
This is structural output exposure, not model retraining or logit extraction.
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
    source=Path(args.model)
    manifest=json.loads((source/'manifest.json').read_text())
    path=source/manifest['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:
        raise ValueError('Primary checksum mismatch')
    graph=onnx.load(str(path))
    nodes=list(graph.graph.node)
    producers={name:n for n in nodes for name in n.output}
    heads=[n for n in nodes if n.op_type in ('MatMul','MatMulInteger','QLinearMatMul','Gemm') and 'ctc_output' in n.name]
    if len(heads)!=1:
        raise ValueError('Cannot identify the single pre-head acoustic feature tensor')
    name=heads[0].input[0]
    activation=producers.get(name)
    if activation is not None and activation.op_type in ('DynamicQuantizeLinear','QuantizeLinear'):
        name=activation.input[0]
    inferred=onnx.shape_inference.infer_shapes(graph)
    shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in list(inferred.graph.value_info)+list(inferred.graph.output)}
    shape=shapes.get(name)
    if not shape or len(shape)!=3 or not shape[-1]:
        raise ValueError('Pre-head geometry is unknown')
    graph.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[d if d else None for d in shape]))
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    output=dest/'exemplar-features.onnx'
    onnx.checker.check_model(graph);onnx.save(graph,str(output))
    options=ort.SessionOptions();options.intra_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    primary=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    extractor=ort.InferenceSession(str(output),options,providers=['CPUExecutionProvider'])
    feeds={s['name']:np.zeros(s['shape'],dtype=np.int64 if s['dtype']=='int64' else np.float32) for s in manifest['states']}
    rng=np.random.default_rng(1729)
    names=[o.name for o in primary.get_outputs()]
    for _ in range(8):
        feeds['x']=rng.normal(size=(1,manifest['window'],manifest['fbank']['mels'])).astype(np.float32)
        a,b=primary.run(names,feeds),extractor.run(names,feeds)
        for key,x,y in zip(names,a,b):np.testing.assert_array_equal(x,y,err_msg='Extractor changed '+key)
        for s in manifest['states']:feeds[s['name']]=a[names.index('new_'+s['name'])]
    metadata=dict(output=name,width=shape[-1],inventory=manifest['inventory'],tokens=manifest['tokens'],
                  primarySha256=manifest['sha256'],parityWindows=8)
    (dest/'exemplar-extractor.json').write_text(json.dumps(metadata))
    print(json.dumps(metadata),flush=True)


def bundle(args):
    """Build the complete side bundle from training speakers before target use."""
    import shutil
    import subprocess
    source=Path(args.model)
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    export(args)
    # Preserve the primary model, manifest and any existing side-ear files.
    for path in source.iterdir():
        if path.is_file():shutil.copy2(path,dest/path.name)
    labels=[json.loads(line) for line in Path(args.labels).open() if line.strip()]
    excluded=sorted({r['speaker'] for r in labels if r['split']!='train'})
    (dest/'excluded-speakers.json').write_text(json.dumps(excluded))
    input_path=dest/'exemplar-training.jsonl'
    subprocess.run(['pnpm','--filter','@tilawa/sdk','exec','tsx','src/bench/exemplar-density-cli.ts',
                    '--model',str(source),'--extractor',str(dest),'--labels',args.labels,
                    '--excluded-speakers',str(dest/'excluded-speakers.json'),'--out',str(input_path)],check=True)
    metadata=json.loads((dest/'exemplar-extractor.json').read_text())
    subprocess.run(['python3','modal/student_exemplar_density.py','--input',str(input_path),'--out',str(dest),
                    '--tokens',metadata['tokens'],'--inventory',metadata['inventory'],
                    '--primary-sha256',metadata['primarySha256'],'--output',metadata['output']],check=True)
    descriptor={}
    for name in ('exemplar-density.json','exemplar-features.onnx'):
        path=dest/name
        descriptor[name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (dest/'exemplar-assets.json').write_text(json.dumps(descriptor))
    check = "import { loadNodeModel } from './src/node.ts'; (async () => { const m = await loadNodeModel(" + json.dumps(str(dest)) + "); if (!m.exemplarDensity) throw new Error('Missing declared exemplar scorer'); await m.backend.release(); })();"
    subprocess.run(['pnpm','--filter','@tilawa/sdk','exec','tsx','-e',check],check=True)
    print(json.dumps(dict(bundle=str(dest),files=descriptor,normalSdkLoadVerified=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    p.add_argument('--labels',help='Optional help-train labels; export and fit a complete bundle')
    args=p.parse_args()
    bundle(args) if args.labels else export(args)

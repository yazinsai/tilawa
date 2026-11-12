"""Append weighted value-vector variance to a copy of deployed attention.

The primary nodes, weights, attention masks and streaming state are untouched.
Uses the same A and V tensors as each chosen existing MatMul: A @ V^2 - (A @ V)^2.
The candidate head set is architectural, before any training/target inspection.
"""
import argparse
import hashlib
import json
from pathlib import Path


def add_variance(graph, node, name):
    from onnx import helper, numpy_helper
    import numpy as np
    if node.op_type != 'MatMul' or len(node.input) != 2:
        raise ValueError('Weighted mean must be an existing MatMul')
    weights, values = node.input
    mean = node.output[0]
    zero = name+'_zero'
    graph.graph.initializer.append(numpy_helper.from_array(np.array(0,dtype=np.float32),zero))
    graph.graph.node.extend([
        helper.make_node('Mul',[values,values],[name+'_values_square']),
        helper.make_node('MatMul',[weights,name+'_values_square'],[name+'_second_moment']),
        helper.make_node('Mul',[mean,mean],[name+'_mean_square']),
        helper.make_node('Sub',[name+'_second_moment',name+'_mean_square'],[name+'_raw_variance']),
        helper.make_node('Max',[name+'_raw_variance',zero],[name]),
    ])
    return mean,name


def value_attention_nodes(graph, manifest):
    """Use stage combiners and native value-cache wiring, independent of op names.

    Streaming methods bypass module tracing scopes, so ordinary value MatMuls
    need not have attention/layer names. Nonlinear attention has a different cache.
    """
    originals = list(graph.graph.node)
    producers = {v:n for n in originals for v in n.output}
    consumers = {}
    for node in originals:
        for name in node.input:
            consumers.setdefault(name,[]).append(node)
    stages = [(i,n) for i,n in enumerate(originals) if n.op_type == 'Mul' and
        any('out_combiner.bypass_scale' in name for name in n.input)]
    if len(stages) < 2:
        raise ValueError('Cannot establish the final encoder stage')
    previous = [n for n in consumers.get(stages[-2][1].output[0],[]) if n.op_type == 'Add']
    if len(previous) != 1:
        raise ValueError('Cannot identify the preceding completed encoder stage')
    start,stop = originals.index(previous[0]),stages[-1][0]
    caches = {s['name'] for s in manifest['states'] if s['name'].startswith(('cached_val1_','cached_val2_'))}
    def softmax(value,depth=0):
        node = producers.get(value)
        if node is None or depth > 12:
            return False
        if node.op_type == 'Softmax':
            return True
        if node.op_type in ('Reshape','Transpose','Identity','Cast','Slice','Unsqueeze','Squeeze'):
            return softmax(node.input[0],depth+1)
        return node.op_type == 'Mul' and any(softmax(v,depth+1) for v in node.input)
    def value_cache(value,depth=0):
        if value in caches:
            return True
        node = producers.get(value)
        if node is None or depth > 12:
            return False
        if node.op_type in ('Reshape','Transpose','Identity','Cast','Slice','Unsqueeze','Squeeze'):
            return value_cache(node.input[0],depth+1)
        return node.op_type == 'Concat' and any(value_cache(v,depth+1) for v in node.input)
    selected = [n for i,n in enumerate(originals) if start < i < stop and n.op_type == 'MatMul' and
        softmax(n.input[0]) and value_cache(n.input[1])]
    if not selected:
        raise ValueError('No normalized final-stage value attention was identified')
    return selected


def export(args):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper,TensorProto,numpy_helper
    source = Path(args.model)
    manifest = json.loads((source/'manifest.json').read_text())
    path = source/manifest['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Supplied primary checksum does not match')
    graph = onnx.load(str(path))
    candidates = value_attention_nodes(graph,manifest)
    heads, outputs = [], []
    declared_outputs = {v.name for v in graph.graph.output}
    for i,node in enumerate(candidates):
        mean,variance = add_variance(graph,node,'attention_variance_raw_'+str(i))
        outputs.extend([mean,variance,node.input[0],node.input[1]])
        # Runtime shape establishment below avoids trusting incomplete inferred shapes.
        for v in (mean,variance,node.input[0],node.input[1]):
            if v not in declared_outputs:
                graph.graph.output.append(helper.make_tensor_value_info(v,TensorProto.FLOAT,None))
                declared_outputs.add(v)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    target = out/'attention-dispersion.onnx'
    onnx.save(graph,str(target))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    primary = ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    extractor = ort.InferenceSession(str(target),options,providers=['CPUExecutionProvider'])
    feeds = {s['name']:np.zeros(s['shape'],dtype=np.int64 if s['dtype']=='int64' else np.float32) for s in manifest['states']}
    rng = np.random.default_rng(1729)
    names = [v.name for v in primary.get_outputs()]
    maximum_error = 0.
    for step in range(8):
        feeds['x'] = rng.normal(size=(1,manifest['window'],manifest['fbank']['mels'])).astype(np.float32)
        a,b = primary.run(names,feeds),extractor.run(names+outputs,feeds)
        for name,x,y in zip(names,a,b[:len(names)]):
            np.testing.assert_array_equal(x,y,err_msg='Extractor changed '+name)
        for i,node in enumerate(candidates):
            mean,variance,weights,values = b[len(names)+4*i:len(names)+4*i+4]
            if mean.ndim not in (3,4) or values.ndim != mean.ndim or weights.ndim != mean.ndim or (mean.ndim == 4 and any(v.shape[1] != 1 for v in (mean,values,weights))):
                raise ValueError('Attention geometry must have a singleton batch per head')
            np.testing.assert_allclose(weights.sum(-1),1,atol=2e-5,rtol=2e-5)
            if (weights < 0).any() or not np.isfinite(variance).all():
                raise ValueError('Invalid attention distribution/variance')
            expected = np.maximum(0,weights.astype(float) @ (values.astype(float)**2) - mean.astype(float)**2)
            np.testing.assert_allclose(variance,expected,atol=2e-4,rtol=2e-4)
            maximum_error = max(maximum_error,float(np.abs(variance-expected).max()))
            if step == 0:
                for head in range(mean.shape[0]):
                    heads.append(dict(meanOutput='attention_mean_'+str(i),varianceOutput='attention_variance_'+str(i),
                        head=head,heads=mean.shape[0],width=mean.shape[-1]))
        for state in manifest['states']:
            feeds[state['name']] = b[names.index('new_'+state['name'])]
    # Normalize the singleton batch axis into SDK [heads, frames, channels]
    # outputs, leaving every primary tensor and cache unchanged.
    retained = [v for v in graph.graph.output if v.name in names]
    del graph.graph.output[:]
    graph.graph.output.extend(retained)
    statistic_names = []
    for i,node in enumerate(candidates):
        head = next(h for h in heads if h['meanOutput'] == 'attention_mean_'+str(i))
        shape_name = 'attention_shape_'+str(i)
        graph.graph.initializer.append(numpy_helper.from_array(np.array([head['heads'],-1,head['width']],dtype=np.int64),shape_name))
        for raw,normalized in ((node.output[0],head['meanOutput']),('attention_variance_raw_'+str(i),head['varianceOutput'])):
            graph.graph.node.append(helper.make_node('Reshape',[raw,shape_name],[normalized]))
            graph.graph.output.append(helper.make_tensor_value_info(normalized,TensorProto.FLOAT,[head['heads'],None,head['width']]))
            statistic_names.append(normalized)
    onnx.checker.check_model(graph)
    onnx.save(graph,str(target))
    # Verify the final serialized bundle as well as the temporary diagnostic graph.
    finalized = ort.InferenceSession(str(target),options,providers=['CPUExecutionProvider'])
    primary_final = primary.run(names,feeds)
    final_outputs = finalized.run(names+statistic_names,feeds)
    for name,x,y in zip(names,primary_final,final_outputs):
        np.testing.assert_array_equal(x,y,err_msg='Final extractor changed '+name)
    raw_outputs = extractor.run(outputs,feeds)
    for i in range(len(candidates)):
        for j in range(2):
            raw = raw_outputs[4*i+j]
            normalized = final_outputs[len(names)+2*i+j]
            np.testing.assert_array_equal(normalized,raw.reshape(raw.shape[0],-1,raw.shape[-1]))
    metadata = dict(heads=heads,inventory=manifest['inventory'],tokens=manifest['tokens'],primarySha256=manifest['sha256'])
    (out/'attention-dispersion-metadata.json').write_text(json.dumps(metadata))
    report = dict(branches=[n.name for n in candidates],candidateHeads=len(heads),parityWindows=9,
        primaryAndAllStateOutputsExact=True,maxVarianceError=maximum_error,
        extractorSha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (out/'attention-dispersion-parity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--out',required=True)
    export(p.parse_args())

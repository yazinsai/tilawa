"""Read, refit and graft the deployed dynamic-int8 CTC projection.

The encoder and its state graph are immutable. Head evaluation keeps the full
SDK chunk, including final padding, because DynamicQuantizeLinear chooses its
activation scale over the entire input tensor. No recordings or labels here.
"""
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def runtime(graph):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(graph.SerializeToString(), options, providers=['CPUExecutionProvider'])


class QuantizedHead:
    """Fail closed on anything other than the existing dynamic MatMul recipe."""
    def __init__(self, model):
        self.source = copy.deepcopy(model)
        candidates = [n for n in model.graph.node if 'ctc_output' in n.name
                      and n.op_type == 'MatMulInteger']
        if len(candidates) != 1:
            raise ValueError('Need exactly one deployed dynamic-int8 phone head')
        matmul = candidates[0]
        producers = {v: n for n in model.graph.node for v in n.output}
        quantizer = producers.get(matmul.input[0])
        if (quantizer is None or quantizer.op_type != 'DynamicQuantizeLinear'
                or len(matmul.input) != 4 or list(quantizer.output)[2] != matmul.input[2]):
            raise ValueError('Unsupported activation quantization')
        self.feature = quantizer.input[0]
        self.weight_name = matmul.input[1]
        self.zero_name = matmul.input[3]
        suffix = '_quantized'
        if not self.weight_name.endswith(suffix):
            raise ValueError('Unsupported weight quantization naming')
        self.scale_name = self.weight_name[:-len(suffix)] + '_scale'
        inferred = onnx.shape_inference.infer_shapes(model)
        values = {v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
        if self.feature not in values:
            raise ValueError('Missing native feature shape')
        x_info = copy.deepcopy(values[self.feature])
        dims = x_info.type.tensor_type.shape.dim
        if len(dims) != 3 or not dims[-1].dim_value:
            raise ValueError('Need same-clock rank-three native features')
        self.width = dims[-1].dim_value
        # Trace the actual output computation, stopping only at native features.
        needed, keep = {'log_probs'}, []
        for node in reversed(model.graph.node):
            if not needed.intersection(node.output):
                continue
            keep.append(copy.deepcopy(node))
            needed.update(v for v in node.input if v != self.feature)
        constants = [copy.deepcopy(i) for i in model.graph.initializer if i.name in needed]
        inputs = {i.name for i in constants} | {self.feature}
        for node in reversed(keep):
            if any(v and v not in inputs for v in node.input):
                raise ValueError('Head has a dependency outside frozen native features')
            inputs.update(node.output)
        nodes = list(reversed(keep))
        if any(n.op_type not in {'DynamicQuantizeLinear', 'MatMulInteger', 'Cast', 'Mul', 'Add', 'LogSoftmax'}
               for n in nodes):
            raise ValueError('Unexpected operation in deployed phone head')
        output = next(v for v in model.graph.output if v.name == 'log_probs')
        graph = helper.make_graph(nodes, 'deployed-phone-head', [x_info], [copy.deepcopy(output)], constants)
        self.graph = helper.make_model(graph, opset_imports=copy.deepcopy(model.opset_import))
        self.graph.ir_version = model.ir_version
        onnx.checker.check_model(self.graph)
        tensors = {i.name: numpy_helper.to_array(i) for i in constants}
        self.initial_codes = tensors[self.weight_name].copy()
        self.scale = tensors[self.scale_name].copy()
        self.zero = tensors[self.zero_name].copy()
        if (self.initial_codes.dtype != np.int8 or self.initial_codes.ndim != 2
                or self.initial_codes.shape[0] != self.width or self.scale.size != 1
                or self.zero.size != 1 or not np.isfinite(self.scale).all() or not (self.scale > 0).all()):
            raise ValueError('Need the existing per-tensor signed-int8 weight recipe')
        adds = [n for n in nodes if n.op_type == 'Add']
        biases = [v for n in adds for v in n.input if v in tensors
                  and tensors[v].shape == (self.initial_codes.shape[1],)]
        if len(biases) != 1:
            raise ValueError('Cannot identify the existing float32 bias')
        self.bias = tensors[biases[0]].astype(np.float32).copy()

    def weights(self, codes=None):
        codes = self.initial_codes if codes is None else codes
        return (codes.astype(np.float32) - self.zero.astype(np.float32)) * self.scale

    def with_codes(self, codes):
        if codes.shape != self.initial_codes.shape or codes.dtype != np.int8:
            raise ValueError('Weight codes must retain their signed-int8 shape')
        graph = copy.deepcopy(self.graph)
        tensor = next(i for i in graph.graph.initializer if i.name == self.weight_name)
        tensor.CopyFrom(numpy_helper.from_array(codes.copy(), self.weight_name))
        return graph

    def floating(self, weights):
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != self.initial_codes.shape or not np.isfinite(weights).all():
            raise ValueError('Float control differs from native phone-head shape')
        # Preserve input/output names and LogSoftmax semantics for graph grafting.
        softmax = copy.deepcopy(next(n for n in self.graph.graph.node if n.op_type == 'LogSoftmax'))
        softmax.input[0] = 'ctc_output_refit/logits'
        nodes = [helper.make_node('MatMul', [self.feature, 'ctc_output_refit/weight'],
                                  ['ctc_output_refit/linear'], name='ctc_output_refit/MatMul'),
                 helper.make_node('Add', ['ctc_output_refit/linear', 'ctc_output_refit/bias'],
                                  ['ctc_output_refit/logits'], name='ctc_output_refit/Add'), softmax]
        graph = helper.make_graph(nodes, 'floating-phone-head', copy.deepcopy(self.graph.graph.input),
                                  copy.deepcopy(self.graph.graph.output),
                                  [numpy_helper.from_array(weights, 'ctc_output_refit/weight'),
                                   numpy_helper.from_array(self.bias, 'ctc_output_refit/bias')])
        result = helper.make_model(graph, opset_imports=copy.deepcopy(self.graph.opset_import))
        result.ir_version = self.graph.ir_version
        onnx.checker.check_model(result)
        return result

    def graft(self, head):
        if ([v.name for v in head.graph.input] != [self.feature]
                or [v.name for v in head.graph.output] != ['log_probs']):
            raise ValueError('Head changed the product interface')
        graph = copy.deepcopy(self.source)
        old_outputs = {v for n in self.graph.graph.node for v in n.output}
        old_constants = {i.name for i in self.graph.graph.initializer}
        retained = [n for n in graph.graph.node if not old_outputs.intersection(n.output)]
        del graph.graph.node[:]
        graph.graph.node.extend(retained + list(copy.deepcopy(head.graph.node)))
        # A quantizer/constant shared with the encoder cannot be removed safely.
        if any(v in old_outputs or v in old_constants for n in retained for v in n.input):
            raise ValueError('Phone-head parameters are shared with the frozen encoder')
        kept = [i for i in graph.graph.initializer if i.name not in old_constants]
        del graph.graph.initializer[:]
        graph.graph.initializer.extend(kept + list(copy.deepcopy(head.graph.initializer)))
        kept_info = [v for v in graph.graph.value_info if v.name not in old_outputs]
        del graph.graph.value_info[:]
        graph.graph.value_info.extend(kept_info)
        onnx.checker.check_model(graph)
        return graph

    def exposed(self):
        graph = copy.deepcopy(self.source)
        graph.graph.output.append(copy.deepcopy(self.graph.graph.input[0]))
        onnx.checker.check_model(graph)
        return graph


def quantize_control(head, directory):
    """Call the product export recipe, rather than approximating its rounding."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    from quantize_onnx import int8
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    src, dest = directory / 'head-fp32.onnx', directory / 'head-int8.onnx'
    onnx.save(head, src)
    int8(src, dest)
    return onnx.load(dest)


def stream_chunks(graph, manifest, fbank, feature):
    """Cache full chunks plus their valid counts; carry every encoder state."""
    session = runtime(graph)
    fbank = np.asarray(fbank, dtype=np.float32)
    if (fbank.ndim != 2 or fbank.shape[1] != 80 or not np.isfinite(fbank).all()
            or manifest['window'] != 2 * manifest['chunk'] + 13
            or manifest['shift'] != 2 * manifest['chunk']):
        raise ValueError('Invalid deployed streaming frontend')
    total = ((len(fbank) - 7) // 2 + 1) // 2
    if total < 1:
        raise ValueError('No acoustic output frames')
    states = {s['name']: np.zeros(s['shape'], dtype=s['dtype']) for s in manifest['states']}
    names = ['log_probs'] + ['new_' + s['name'] for s in manifest['states']] + [feature]
    chunks, baselines, emitted, start = [], [], 0, 0
    while emitted < total:
        window = np.full((1, manifest['window'], 80), manifest['fbank']['padValue'], dtype=np.float32)
        view = fbank[start:start + manifest['window']]
        window[0, :len(view)] = view
        outputs = session.run(names, {'x': window, **states})
        keep = min(total - emitted, outputs[0].shape[1])
        if keep <= 0 or outputs[-1].shape[:2] != outputs[0].shape[:2]:
            raise ValueError('Native features changed the posterior clock')
        chunks.append((outputs[-1].copy(), keep))
        baselines.append(outputs[0][0, :keep].copy())
        states = {s['name']: value for s, value in zip(manifest['states'], outputs[1:-1])}
        emitted += keep
        start += manifest['shift']
    return chunks, np.concatenate(baselines)


def predict(session, feature, chunks):
    return np.concatenate([session.run(['log_probs'], {feature: x})[0][0, :keep] for x, keep in chunks])


def bundle(source, manifest, graph, destination, name):
    """Write a new bundle and preserve the SDK's insertion teacher assets."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (source == destination or source in destination.parents or destination in source.parents
            or destination.exists() and any(destination.iterdir())):
        raise ValueError('Need a new directory separate from the shipped bundle')
    # The registry requires both members even if an external tensor is unused.
    for filename in ('teacher.onnx', 'teacher.onnx.data'):
        if not (source / filename).is_file():
            raise ValueError('Missing unchanged SDK insertion teacher asset: ' + filename)
    teacher = onnx.load(source / 'teacher.onnx', load_external_data=False)
    for tensor in teacher.graph.initializer:
        if tensor.data_location == TensorProto.EXTERNAL:
            spec = {v.key: v.value for v in tensor.external_data}
            file = (source / spec['location']).resolve()
            offset, length = int(spec.get('offset', 0)), spec.get('length')
            if (not file.is_relative_to(source) or not file.is_file() or offset < 0
                    or file.stat().st_size <= offset or length is not None and
                    (int(length) < 0 or offset + int(length) > file.stat().st_size)):
                raise ValueError('Missing or truncated insertion teacher external data')
    destination.mkdir(parents=True, exist_ok=True)
    for member in source.rglob('*'):
        if member.is_file() and member.name not in (manifest['file'], 'manifest.json'):
            target = destination / member.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(member, target)
            if digest(member) != digest(target):
                raise ValueError('Auxiliary bundle copy mismatch')
    path = destination / 'model.onnx'
    onnx.checker.check_model(graph)
    onnx.save(graph, path)
    updated = copy.deepcopy(manifest)
    # The old quantization lineage does not describe a refitted/code-edited head.
    updated.pop('quantizedFrom', None)
    updated.update(id=manifest['id'] + '-' + name, file=path.name, bytes=path.stat().st_size,
                   sha256=digest(path), derivedFrom=manifest['sha256'],
                   format='onnx-int8' if name != 'control-fp32' else 'onnx-int8-encoder-fp32-head')
    (destination / 'manifest.json').write_text(json.dumps(updated, ensure_ascii=False, indent=2))
    return path

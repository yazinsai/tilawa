"""Expose frozen native features and replace the existing linear phone head.

Inputs/caches, ordered alphabet and frame clock are inherited from the shipped
bundle. Existing source files are never rewritten. New manifests describe new
graphs, and every declared auxiliary file must be copied and checksum-verified.
"""
import copy
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def insertion_assets(source):
    """Require the existing SDK teacher, including all ONNX external weights.

    The primary phone-head experiment must never silently disable the shipped
    insertion verification just because its training bundle is incomplete.
    """
    import onnx
    source = Path(source).resolve()
    teacher = source / "teacher.onnx"
    if not teacher.is_file():
        raise ValueError("Source bundle lacks the unchanged SDK insertion teacher")
    graph = onnx.load(str(teacher), load_external_data=False)
    members = {teacher}
    for tensor in graph.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        spec = {v.key: v.value for v in tensor.external_data}
        member = (source / spec["location"]).resolve()
        if source not in member.parents or not member.is_file():
            raise ValueError("Insertion teacher external weights are missing from the bundle")
        offset, length = int(spec.get("offset", 0)), spec.get("length")
        if offset < 0 or member.stat().st_size <= offset or (length is not None and
                (int(length) < 0 or offset + int(length) > member.stat().st_size)):
            raise ValueError("Insertion teacher external weights are truncated")
        members.add(member)
    # The SDK's existing registry loads this member before inference.
    if not (source / "teacher.onnx.data").is_file():
        raise ValueError("Source bundle lacks the SDK insertion teacher data file")
    members.add(source / "teacher.onnx.data")
    return {str(member.relative_to(source)): digest(member) for member in sorted(members)}


def native_features(graph):
    import onnx
    heads = [n for n in graph.graph.node if "ctc_output" in n.name
             and n.op_type in ("MatMul", "MatMulInteger", "QLinearMatMul", "Gemm")]
    if len(heads) != 1:
        raise ValueError("Cannot uniquely identify the shipped CTC projection")
    feature = heads[0].input[0]
    producer = next((n for n in graph.graph.node if feature in n.output), None)
    if producer and producer.op_type in ("DynamicQuantizeLinear", "QuantizeLinear"):
        feature = producer.input[0]
    inferred = onnx.shape_inference.infer_shapes(graph)
    value = next((v for v in list(inferred.graph.value_info) + list(inferred.graph.output)
                  if v.name == feature), None)
    dims = [d.dim_value for d in value.type.tensor_type.shape.dim] if value else []
    if len(dims) != 3 or not dims[-1]:
        raise ValueError("Cannot establish native encoder feature width")
    return feature, dims[-1]


def load_source(source):
    import onnx
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    path = source / manifest["file"]
    if digest(path) != manifest["sha256"]:
        raise ValueError("Shipped model checksum mismatch")
    if (manifest["blank"] != 0 or manifest["frameMs"] != 40 or
            manifest["fbank"]["mels"] != 80 or manifest["sampleRate"] != 16000):
        raise ValueError("This experiment requires the shipped phone clock/front end")
    graph = onnx.load(str(path))
    feature, width = native_features(graph)
    return manifest, graph, feature, width


def expose(source, out):
    import onnx
    from onnx import helper, TensorProto
    manifest, graph, feature, width = load_source(source)
    graph.graph.output.append(helper.make_tensor_value_info(feature, TensorProto.FLOAT, [1, None, width]))
    onnx.checker.check_model(graph)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "frozen-features.onnx"
    onnx.save(graph, str(path))
    return manifest, feature, width, path


def session(path, optimize=False):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimize
                                       else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def stream(runtime, manifest, features, extra=None):
    """Exact SDK window/drain geometry, returning valid rows and optional features."""
    import numpy as np
    if (features.ndim != 2 or features.shape[1] != manifest["fbank"]["mels"] or
            not np.isfinite(features).all()):
        raise ValueError("Invalid full-utterance filterbank features")
    if not 0 < manifest["shift"] <= manifest["window"]:
        raise ValueError("Invalid streaming window/shift")
    names = ["log_probs"] + ["new_" + s["name"] for s in manifest["states"]] + ([extra] if extra else [])
    state = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32)
             for s in manifest["states"]}
    frames = (int((len(features) - 7) // 2) + 1) // 2
    if frames < 1:
        raise ValueError("No complete acoustic frames")
    outputs, observations, consumed, emitted = [], [], 0, 0
    while emitted < frames:
        x = np.full((1, manifest["window"], manifest["fbank"]["mels"]),
                    manifest["fbank"]["padValue"], dtype=np.float32)
        view = features[consumed:consumed + manifest["window"]]
        x[0, :len(view)] = view
        result = runtime.run(names, {"x": x, **state})
        keep = min(result[0].shape[1], frames - emitted)
        if keep < 1 or not np.isfinite(result[0]).all():
            raise ValueError("Invalid primary output while draining the stream")
        outputs.append(result[0][0, :keep].copy())
        if extra:
            if result[-1].shape[:2] != result[0].shape[:2]:
                raise ValueError("Encoder features do not share the primary frame clock")
            observations.append(result[-1][0, :keep].copy())
        state = {s["name"]: r.copy() for s, r in zip(manifest["states"], result[1:1 + len(manifest["states"])])}
        consumed += manifest["shift"]
        emitted += keep
    return np.concatenate(outputs), np.concatenate(observations) if extra else None


def head_parameters(graph):
    """Read the shipped affine projection, dequantizing only its weights.

    Dynamic activation rounding disappears in both refit arms. The frozen
    encoder remains quantized; no residual head or extra capacity is added.
    Unsupported projection graphs fail rather than guessing parameters.
    """
    import numpy as np
    from onnx import numpy_helper, helper
    tensors = {v.name: numpy_helper.to_array(v) for v in graph.graph.initializer}
    heads = [n for n in graph.graph.node if "ctc_output" in n.name
             and n.op_type in ("MatMul", "MatMulInteger", "QLinearMatMul", "Gemm")]
    if len(heads) != 1:
        raise ValueError("Cannot uniquely identify native phone head")
    head = heads[0]
    attrs = {v.name: helper.get_attribute_value(v) for v in head.attribute}
    feature, width = native_features(graph)
    if head.op_type in ("MatMul", "Gemm"):
        if head.input[1] not in tensors:
            raise ValueError("Native projection weight is not constant")
        w = tensors[head.input[1]].astype(np.float64)
        if head.op_type == "Gemm":
            if attrs.get("transA", 0) or attrs.get("alpha", 1.) != 1. or attrs.get("beta", 1.) != 1.:
                raise ValueError("Unsupported native Gemm layout")
            if attrs.get("transB", 0):
                w = w.T
    else:
        if head.op_type == "QLinearMatMul":
            weight_name, scale_name, zero_name = head.input[3:6]
        else:
            weight_name = head.input[1]
            if not weight_name.endswith("_quantized"):
                raise ValueError("Unknown dynamic projection quantization layout")
            scale_name = weight_name[:-len("_quantized")] + "_scale"
            zero_name = head.input[3] if len(head.input) > 3 else None
        if weight_name not in tensors or scale_name not in tensors or (zero_name and zero_name not in tensors):
            raise ValueError("Missing quantized native projection parameters")
        w = tensors[weight_name].astype(np.float64)
        scale = tensors[scale_name].astype(np.float64)
        zero = tensors[zero_name].astype(np.float64) if zero_name else 0.
        if np.any(scale <= 0) or np.size(scale) not in (1, w.shape[-1]) or np.size(zero) not in (1, w.shape[-1]):
            raise ValueError("Unsupported native per-channel quantization")
        w = (w - zero) * scale
    terminal = [n for n in graph.graph.node if "log_probs" in n.output]
    if len(terminal) != 1 or terminal[0].op_type != "LogSoftmax":
        raise ValueError("Native phone output is not a terminal LogSoftmax")
    # Trace only the head-to-softmax chain, excluding encoder dependencies.
    producers = {out: n for n in graph.graph.node for out in n.output}
    biases, reached = [], False
    pending, seen = [terminal[0].input[0]], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in head.output:
            reached = True
            continue
        node = producers.get(name)
        if node is None:
            continue
        if node.op_type == "DynamicQuantizeLinear":
            continue  # Activation scale dependency of dynamic MatMulInteger.
        if node.op_type not in ("Add", "Mul", "Cast", "DequantizeLinear"):
            raise ValueError("Non-affine computation after native phone head")
        if node.op_type == "Add":
            biases.extend(tensors[x] for x in node.input if x in tensors)
        pending.extend(x for x in node.input if x not in tensors)
    if head.op_type == "Gemm" and len(head.input) == 3:
        biases.append(tensors[head.input[2]])
    if not reached or w.ndim != 2 or w.shape[0] != width or len(biases) != 1 or biases[0].shape != (w.shape[1],):
        raise ValueError("Cannot establish existing affine phone-head layout")
    b = biases[0]
    if not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite native phone head")
    return w.T.astype(np.float32), b.astype(np.float32)


def replace_head(graph, weight, bias):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    native_w, native_b = head_parameters(graph)
    weight, bias = np.asarray(weight, dtype=np.float32), np.asarray(bias, dtype=np.float32)
    if weight.shape != native_w.shape or bias.shape != native_b.shape or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("Refit differs from existing head dimensions")
    feature, _ = native_features(graph)
    prefix = "tilawa_refit_head/"
    if any(prefix in n.name for n in graph.graph.node):
        raise ValueError("Cannot replace head twice")
    terminal = next(n for n in graph.graph.node if "log_probs" in n.output)
    terminal.input[0] = prefix + "logits"
    index = list(graph.graph.node).index(terminal)
    nodes = [helper.make_node("MatMul", [feature, prefix + "weight"], [prefix + "linear"], name=prefix + "linear"),
             helper.make_node("Add", [prefix + "linear", prefix + "bias"], [prefix + "logits"], name=prefix + "logits")]
    for i, node in enumerate(nodes):
        graph.graph.node.insert(index + i, node)
    graph.graph.initializer.extend([numpy_helper.from_array(weight.T.copy(), prefix + "weight"),
                                   numpy_helper.from_array(bias.copy(), prefix + "bias")])
    # Remove the now-unused quantized projection, not the encoder/state graph.
    needed = {v.name for v in graph.graph.output}
    kept = []
    for node in reversed(graph.graph.node):
        if needed.intersection(node.output):
            kept.append(node)
            needed.update(node.input)
    del graph.graph.node[:]
    graph.graph.node.extend(reversed(kept))
    retained = [v for v in graph.graph.initializer if v.name in needed]
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(retained)
    retained = [v for v in graph.graph.value_info if v.name in needed]
    del graph.graph.value_info[:]
    graph.graph.value_info.extend(retained)
    onnx.checker.check_model(graph)
    return graph


def graft(source, head, out, arm):
    import onnx
    manifest, graph, _, _ = load_source(source)
    auxiliary_hashes = insertion_assets(source)
    graph = replace_head(graph, head.weight.detach().cpu().numpy(), head.bias.detach().cpu().numpy())
    out, from_file = Path(out).resolve(), Path(source).resolve()
    if out == from_file or from_file in out.parents or out in from_file.parents:
        raise ValueError("Export must be separate from shipped bundle")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Export directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model.onnx"
    onnx.save(graph, str(path))
    for member in from_file.rglob("*"):
        relative = member.relative_to(from_file)
        if member.is_file() and relative.as_posix() not in (manifest["file"], "manifest.json"):
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(member, target)
            if digest(member) != digest(target):
                raise ValueError("Auxiliary bundle copy mismatch")
    if insertion_assets(out) != auxiliary_hashes:
        raise ValueError("Export changed existing insertion verification assets")
    new_manifest = copy.deepcopy(manifest)
    new_manifest.update(id=manifest["id"] + "-" + arm + "-head-refit", file=path.name,
                        bytes=path.stat().st_size, sha256=digest(path), format="onnx-int8-encoder-fp32-head")
    (out / "manifest.json").write_text(json.dumps(new_manifest, ensure_ascii=False, indent=1))
    return path

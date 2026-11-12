"""Expose frozen native features and graft one selected nonlinear phone head.

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


def graft(source, head, center, scale, out, arm):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    from student_peer_head import PhoneHead
    manifest, graph, feature, width = load_source(source)
    if (center.shape != (width,) or scale.shape != (width,) or np.any(scale <= 0) or
            not np.isfinite(center).all() or not np.isfinite(scale).all()):
        raise ValueError("Invalid training-only feature normalization")
    if any(not np.isfinite(p.detach().cpu().numpy()).all() for p in head.parameters()):
        raise ValueError("Non-finite trained phone-head parameters")
    if not isinstance(head, PhoneHead) or head.hidden.in_features != width or head.output.out_features != len(manifest["tokens"]) + 1:
        raise ValueError("Selected head does not match the frozen encoder/alphabet")
    prefix = "tilawa_phone_head/"
    if any(prefix in n.name for n in graph.graph.node):
        raise ValueError("Cannot graft twice")
    def constant(name, value):
        name = prefix + name
        graph.graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name))
        return name
    def node(op, inputs, name):
        output = prefix + name
        graph.graph.node.append(helper.make_node(op, inputs, [output], name=output))
        return output
    def linear(x, layer, name):
        weight = constant(name + "/weight", layer.weight.detach().cpu().numpy().T)
        y = node("MatMul", [x, weight], name + "/matmul")
        if layer.bias is not None:
            bias = constant(name + "/bias", layer.bias.detach().cpu().numpy())
            y = node("Add", [y, bias], name + "/affine")
        return y
    # Keep encoder computations and cache outputs. The old CTC readout becomes
    # dead code and is pruned below; it is never an inference ensemble member.
    frozen = prefix + "shipped_log_probs"
    for n in graph.graph.node:
        for i, name in enumerate(n.input):
            if name == "log_probs": n.input[i] = frozen
        for i, name in enumerate(n.output):
            if name == "log_probs": n.output[i] = frozen
    x = node("Sub", [feature, constant("center", center)], "centered")
    x = node("Div", [x, constant("scale", scale)], "normalized")
    hidden = node("Tanh", [linear(x, head.hidden, "hidden")], "tanh")
    scores = linear(hidden, head.output, "output")
    graph.graph.node.append(helper.make_node("LogSoftmax", [scores], ["log_probs"],
                                             axis=-1, name=prefix + "log_softmax"))
    required = {v.name for v in graph.graph.output}
    retained = []
    for n in reversed(graph.graph.node):
        if required.intersection(n.output):
            retained.append(n)
            required.update(n.input)
    del graph.graph.node[:]
    graph.graph.node.extend(reversed(retained))
    used = [v for v in graph.graph.initializer if v.name in required]
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(used)
    onnx.checker.check_model(graph)
    out = Path(out).resolve()
    from_file = Path(source).resolve()
    if out == from_file or from_file in out.parents or out in from_file.parents:
        raise ValueError("Export must be separate from the shipped bundle")
    # An old bundle may contain a teacher that the source no longer declares.
    if out.exists() and any(out.iterdir()):
        raise ValueError("Export directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model.onnx"
    onnx.save(graph, str(path))
    # Auxiliary files travel with the primary model, including teacher external
    # data. Copy only bundle members, never evaluation inputs or inferred assets.
    for member in from_file.rglob("*"):
        relative = member.relative_to(from_file)
        if member.is_file() and relative.as_posix() not in (manifest["file"], "manifest.json"):
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(member, target)
            if digest(member) != digest(target):
                raise ValueError("Auxiliary bundle copy mismatch")
    new_manifest = copy.deepcopy(manifest)
    new_manifest.update(id=manifest["id"] + "-" + arm + "-phone-head",
                        file=path.name, bytes=path.stat().st_size, sha256=digest(path),
                        format="onnx-int8-encoder-fp32-head")
    (out / "manifest.json").write_text(json.dumps(new_manifest, ensure_ascii=False, indent=1))
    return path

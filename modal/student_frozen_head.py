"""Read and replace the shipped ONNX phone head without changing its encoder.

CPU utilities for training-only, full-recording head experiments. The graph's
public inputs, outputs and recurrent-state computations remain intact.
"""

import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def output_frames(n):
    return max(0, ((n - 7) // 2 + 1) // 2)


def readout(graph, vocabulary):
    """Recover a float readout from either a float or dynamic-int8 final layer."""
    import numpy as np
    from onnx import numpy_helper

    producers = {o: n for n in graph.node for o in n.output}
    arrays = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    softmax = producers[graph.output[0].name]
    if softmax.op_type != "LogSoftmax":
        raise ValueError("primary output must end in LogSoftmax")
    add = producers[softmax.input[0]]
    if add.op_type != "Add":
        raise ValueError("expected a final linear readout bias")
    biases = [i for i in add.input if i in arrays and arrays[i].shape == (vocabulary,)]
    if len(biases) != 1:
        raise ValueError("ambiguous final head bias")
    stack, seen, matches = [i for i in add.input if i != biases[0]], set(), []
    while stack:
        name = stack.pop()
        if name in seen or name not in producers:
            continue
        seen.add(name)
        node = producers[name]
        if node.op_type in ("MatMulInteger", "MatMul") and node.input[1] in arrays:
            w = arrays[node.input[1]]
            if w.ndim == 2 and w.shape[1] == vocabulary:
                matches.append(node)
                continue
        stack.extend(node.input)
    if len(matches) != 1:
        raise ValueError("cannot identify a unique linear phone head")
    linear = matches[0]
    weight, feature = arrays[linear.input[1]].copy(), linear.input[0]
    if linear.op_type == "MatMulInteger":
        quantizer = producers[feature]
        if quantizer.op_type != "DynamicQuantizeLinear":
            raise ValueError("unsupported activation quantization")
        feature = quantizer.input[0]
        zero = arrays[linear.input[3]] if len(linear.input) > 3 else 0
        scale = linear.input[1].removesuffix("_quantized") + "_scale"
        if scale not in arrays:
            raise ValueError("missing head weight quantization scale")
        weight = (weight.astype(np.float32) - zero) * arrays[scale]
    return feature, weight.astype(np.float32).T.copy(), arrays[biases[0]].astype(np.float32), softmax


class Encoder:
    def __init__(self, directory, threads=2):
        import numpy as np
        import onnx
        import onnxruntime as ort

        self.manifest = m = json.loads((Path(directory) / "manifest.json").read_text())
        self.path = Path(directory) / m["file"]
        if digest(self.path) != m["sha256"]:
            raise ValueError("source model checksum mismatch")
        if (m["frameMs"] != 40 or m["shift"] != 2 * m["chunk"]
                or m["window"] != m["shift"] + 13 or m["fbank"]["mels"] != 80):
            raise ValueError("unsupported streaming geometry")
        graph = onnx.load(self.path)
        if [o.name for o in graph.graph.output] != ["log_probs"] + [f'new_{s["name"]}' for s in m["states"]]:
            raise ValueError("source output/state order differs from the SDK interface")
        feature, self.weight, self.bias, _ = readout(graph.graph, len(m["tokens"]) + 1)
        graph.graph.output.append(onnx.helper.make_tensor_value_info(
            feature, onnx.TensorProto.FLOAT, [1, None, self.weight.shape[1]]))
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(graph.SerializeToString(), options, providers=["CPUExecutionProvider"])
        self.states, self.np = m["states"], np

    def zero_states(self):
        return {s["name"]: self.np.zeros(s["shape"], self.np.int64 if s["dtype"] == "int64" else self.np.float32)
                for s in self.states}

    def stream(self, features):
        """Use the ordinary chunk origin, state carries, padding and valid clock."""
        np, m = self.np, self.manifest
        count = output_frames(len(features))
        if not count:
            raise ValueError("recording is too short")
        windows = (count + m["chunk"] // 2 - 1) // (m["chunk"] // 2)
        total = m["window"] + (windows - 1) * m["shift"]
        f = np.concatenate((features, np.full((max(0, total - len(features)), 80), m["fbank"]["padValue"], np.float32)))
        state, lp, hidden = self.zero_states(), [], []
        for start in range(0, total - m["window"] + 1, m["shift"]):
            result = self.session.run(None, {"x": f[None, start:start + m["window"]], **state})
            lp.append(result[0][0])
            hidden.append(result[-1][0])
            state = {s["name"]: v for s, v in zip(self.states, result[1:-1])}
        return dict(features=np.concatenate(hidden)[:count], log_probs=np.concatenate(lp)[:count],
                    times_ms=np.arange(count, dtype=np.int64) * m["frameMs"])


def export(model, checkpoint, out, threads=2, required_assets=("teacher.onnx", "teacher.onnx.data")):
    """Export a complete bundle and verify the new head plus exact state parity."""
    import shutil
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    source, out = Path(model), Path(out)
    encoder = Encoder(source, threads)
    fit = torch.load(checkpoint, weights_only=False)
    if fit["source_sha256"] != encoder.manifest["sha256"]:
        raise ValueError("refit and export source models differ")
    for name in required_assets:
        if not (source / name).is_file():
            raise ValueError(f"missing unchanged SDK asset: {name}")
    out.mkdir(parents=True, exist_ok=False)
    # Preserve every auxiliary file, including externally stored teacher weights.
    for path in source.iterdir():
        if path.is_file() and path.name not in (encoder.path.name, "manifest.json"):
            shutil.copyfile(path, out / path.name)
            if digest(path) != digest(out / path.name):
                raise ValueError(f"copied asset checksum mismatch: {path.name}")
    graph = onnx.load(encoder.path)
    feature, _, _, softmax = readout(graph.graph, len(encoder.manifest["tokens"]) + 1)
    softmax.output[0] = "refit_original_log_probs"
    graph.graph.initializer.extend([
        onnx.numpy_helper.from_array(fit["head"]["weight"].numpy().T.copy(), "refit_weight"),
        onnx.numpy_helper.from_array(fit["head"]["bias"].numpy().copy(), "refit_bias")])
    graph.graph.node.extend([
        onnx.helper.make_node("MatMul", [feature, "refit_weight"], ["refit_linear"]),
        onnx.helper.make_node("Add", ["refit_linear", "refit_bias"], ["refit_logits"]),
        onnx.helper.make_node("LogSoftmax", ["refit_logits"], [graph.graph.output[0].name], axis=-1)])
    onnx.checker.check_model(graph)
    onnx.save(graph, out / "model.onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(out / "model.onnx"), options, providers=["CPUExecutionProvider"])
    rng, states, maximum = np.random.default_rng(0), encoder.zero_states(), 0.
    for _ in range(4):
        x = rng.normal(size=(1, encoder.manifest["window"], 80)).astype(np.float32)
        old = encoder.session.run(None, {"x": x, **states})
        new = session.run(None, {"x": x, **states})
        expected = torch.nn.functional.linear(torch.from_numpy(old[-1]), fit["head"]["weight"],
                                              fit["head"]["bias"]).log_softmax(-1).numpy()
        np.testing.assert_allclose(new[0], expected, rtol=1e-5, atol=1e-4)
        maximum = max(maximum, float(np.abs(new[0] - expected).max()))
        if len(new) != len(old) - 1:
            raise ValueError("public model outputs changed")
        for before, after in zip(old[1:-1], new[1:]):
            np.testing.assert_array_equal(before, after)
        states = {s["name"]: v for s, v in zip(encoder.states, new[1:])}
    manifest = {**encoder.manifest, "id": f'degradation-head-{fit["arm"]}', "file": "model.onnx",
                "sha256": digest(out / "model.onnx"), "bytes": (out / "model.onnx").stat().st_size,
                "headRefit": {k: fit[k] for k in ("source_sha256", "selected_updates", "selected_course", "arm", "cache_sha256")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    files = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in out.iterdir() if p.is_file()}
    (out / "verification.json").write_text(json.dumps(dict(max_head_parity_error=maximum,
                                                          recurrent_states="exact", files=files), indent=2))
    print(json.dumps(dict(bundle=str(out), verification="passed", files=list(files))), flush=True)

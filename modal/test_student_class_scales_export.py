"""Native-graph fixtures verify exact initialization and complete head exports."""

import hashlib
import json

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from student_class_scales import ClassLocalHead
from student_class_scales_export import (digest, expose, graft, head_parameters,
                                     insertion_assets, load_source, session)


def bundle(path, quantized):
    path.mkdir()
    rng = np.random.default_rng(9)
    encoder = rng.normal(0, .08, (80, 7)).astype(np.float32)
    w = rng.normal(0, .15, (7, 5)).astype(np.float32)
    bias = rng.normal(0, .1, 5).astype(np.float32)
    init = [numpy_helper.from_array(encoder, "encoder"), numpy_helper.from_array(bias, "bias")]
    nodes = [helper.make_node("MatMul", ["x", "encoder"], ["features"], name="encoder")]
    if quantized:
        scale, zero = np.array(.002, np.float32), np.array(0, np.int8)
        init += [numpy_helper.from_array(np.rint(w / scale).clip(-128, 127).astype(np.int8), "weight_quantized"),
                 numpy_helper.from_array(scale, "weight_scale"), numpy_helper.from_array(zero, "weight_zero_point")]
        nodes += [helper.make_node("DynamicQuantizeLinear", ["features"], ["xq", "xs", "xz"], name="head_quantize"),
                  helper.make_node("MatMulInteger", ["xq", "weight_quantized", "xz", "weight_zero_point"],
                                   ["int_logits"], name="ctc_output/MatMul"),
                  helper.make_node("Cast", ["int_logits"], ["float_logits"], to=TensorProto.FLOAT),
                  helper.make_node("Mul", ["xs", "weight_scale"], ["scale"]),
                  helper.make_node("Mul", ["float_logits", "scale"], ["linear"])]
    else:
        init.append(numpy_helper.from_array(w, "weight"))
        nodes += [helper.make_node("MatMul", ["features", "weight"], ["linear"], name="ctc_output/MatMul")]
    nodes += [helper.make_node("Add", ["linear", "bias"], ["logits"]),
              helper.make_node("LogSoftmax", ["logits"], ["log_probs"], axis=-1),
              helper.make_node("Identity", ["state"], ["new_state"])]
    graph = helper.make_graph(nodes, "fixture",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, None, 80]),
         helper.make_tensor_value_info("state", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("log_probs", TensorProto.FLOAT, [1, None, 5]),
         helper.make_tensor_value_info("new_state", TensorProto.FLOAT, [1])], init)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, path / "model.onnx")
    manifest = dict(file="model.onnx", sha256=digest(path / "model.onnx"), id="fixture",
                    blank=0, frameMs=40, sampleRate=16000, tokens="abcd", format="onnx-int8",
                    fbank=dict(mels=80, padValue=-23.0259), window=5, shift=4,
                    states=[dict(name="state", shape=[1], dtype="float32")])
    (path / "manifest.json").write_text(json.dumps(manifest))
    teacher = helper.make_model(helper.make_graph(
        [helper.make_node("Identity", ["constant"], ["y"])], "teacher", [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [80, 7])],
        [numpy_helper.from_array(encoder, "constant")]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save_model(teacher, path / "teacher.onnx", save_as_external_data=True,
                    all_tensors_to_one_file=True, location="teacher.onnx.data", size_threshold=0)
    return manifest


@pytest.mark.parametrize("quantized", [False, True])
def test_native_exposure_preserves_source_outputs_and_rounding_then_exports_the_fitted_head(tmp_path, quantized):
    source, exposed = tmp_path / "source", tmp_path / "features"
    original_manifest = bundle(source, quantized)
    original_bytes = (source / "model.onnx").read_bytes()
    m, feature, _, path = expose(source, exposed)
    inputs = dict(x=np.random.default_rng(29).normal(0, 1, (1, 5, 80)).astype(np.float32),
                  state=np.array([3.], np.float32))
    original = session(source / "model.onnx").run(None, inputs)
    frozen = session(path).run(["log_probs", "new_state", feature], inputs)
    for a, b in zip(original, frozen[:2]):
        np.testing.assert_array_equal(a, b)
    _, graph, _, _ = load_source(source)
    w, bias = head_parameters(graph)
    linear = torch.nn.Linear(w.shape[1], w.shape[0])
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(w))
        linear.bias.copy_(torch.tensor(bias))
    head = ClassLocalHead(linear)
    x = torch.tensor(frozen[-1])
    with torch.no_grad():
        np.testing.assert_allclose(head(x).softmax(-1).numpy(), np.exp(original[0]), atol=2e-6, rtol=2e-5)
        for i, row in enumerate(head.rows):
            row.add_(.03 * i * torch.arange(row.numel()).reshape_as(row))
        for bias in head.biases:
            bias.add_(.02)
    out = tmp_path / "export"
    model_path = graft(source, head, out, "class-local")
    result = session(model_path, optimize=True).run(None, inputs)
    with torch.no_grad():
        np.testing.assert_allclose(np.exp(result[0]), head(x).softmax(-1).numpy(), atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(result[1], original[1])
    assert insertion_assets(source) == insertion_assets(out)
    assert original_bytes == (source / "model.onnx").read_bytes()
    exported_manifest = json.loads((out / "manifest.json").read_text())
    for field in ("blank", "tokens", "sampleRate", "frameMs", "fbank", "window", "shift", "states"):
        assert exported_manifest[field] == original_manifest[field]
    assert exported_manifest["sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()


def test_missing_verifier_asset_blocks_export_and_wrong_primary_checksum_blocks_loading(tmp_path):
    source = tmp_path / "source"
    bundle(source, False)
    (source / "teacher.onnx.data").unlink()
    with pytest.raises(ValueError, match="missing|lacks"):
        insertion_assets(source)
    (source / "model.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_source(source)

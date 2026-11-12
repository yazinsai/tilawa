"""Stateful and int8 ONNX integration checks for the frozen-head export."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_articulation_train import phone_head, log_probs
from student_head_export import digest, expose, graft, session, stream


def source_bundle(path):
    import onnx
    from onnx import helper as h, TensorProto as T, numpy_helper as n
    path.mkdir()
    # Stateful native feature emitter, with the same frame geometry as SDK.
    nodes = [h.make_node("ReduceMean", ["x"], ["mean"], axes=[1], keepdims=1),
             h.make_node("Expand", ["mean", "shape"], ["feature"]),
             h.make_node("MatMul", ["feature", "weight"], ["scores"], name="ctc_output/MatMul"),
             h.make_node("LogSoftmax", ["scores"], ["log_probs"], axis=-1),
             h.make_node("Add", ["state", "one"], ["new_state"])]
    g = h.make_graph(nodes, "test", [h.make_tensor_value_info("x", T.FLOAT, [1, 45, 80]),
        h.make_tensor_value_info("state", T.INT64, [1])],
        [h.make_tensor_value_info("log_probs", T.FLOAT, [1, 8, 3]), h.make_tensor_value_info("new_state", T.INT64, [1])],
        [n.from_array(np.array([1, 8, 80], dtype=np.int64), "shape"),
         n.from_array(np.random.default_rng(0).normal(0, .01, (80, 3)).astype(np.float32), "weight"),
         n.from_array(np.ones(1, dtype=np.int64), "one")])
    model = h.make_model(g, opset_imports=[h.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, path / "model.onnx")
    # Real external tensor storage exercises complete auxiliary bundle copying.
    teacher = h.make_model(h.make_graph([], "teacher", [], [],
        [n.from_array(np.ones(4, dtype=np.float32), "teacher_weight")]), ir_version=9)
    onnx.save_model(teacher, path / "teacher.onnx", save_as_external_data=True,
                   all_tensors_to_one_file=True, location="teacher.onnx.data", size_threshold=0)
    manifest = dict(file="model.onnx", sha256=digest(path / "model.onnx"), id="unit", tokens="ab", blank=0,
                    frameMs=40, sampleRate=16000, window=45, shift=32, fbank=dict(mels=80, padValue=-23.0259),
                    states=[dict(name="state", shape=[1], dtype="int64")])
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_export_preserves_caches_clock_assets_and_selected_head(tmp_path):
    source = tmp_path / "source"
    manifest = source_bundle(source)
    m, feature, width, extractor = expose(source, tmp_path / "extraction")
    assert width == 80
    head = phone_head(width, 3)
    with torch.no_grad():
        head.bias.copy_(torch.tensor([.2, -.1, .3]))
        head.weight[:, :2] = torch.tensor([[.1, .2], [.3, -.1], [0., .2]])
    new = graft(source, head, tmp_path / "export", "relative")
    new_m = json.loads((new.parent / "manifest.json").read_text())
    f = np.random.default_rng(1).normal(-10, 2, (99, 80)).astype(np.float32)
    old, x = stream(session(extractor), m, f, feature)
    got, _ = stream(session(new), new_m, f)
    with torch.no_grad():
        want = log_probs(head, dict(features=torch.tensor(x), shipped=torch.tensor(old))).numpy()
    np.testing.assert_allclose(got, want, atol=2e-6)
    for key in ("states", "window", "shift", "fbank", "tokens", "blank", "frameMs", "sampleRate"):
        assert manifest[key] == new_m[key]
    for name in ("teacher.onnx", "teacher.onnx.data"):
        assert digest(source / name) == digest(new.parent / name)
    feeds = dict(x=f[:45][None], state=np.array([3], dtype=np.int64))
    np.testing.assert_array_equal(session(source / "model.onnx").run(["new_state"], feeds)[0],
                                  session(new).run(["new_state"], feeds)[0])
    assert digest(source / "model.onnx") == manifest["sha256"]
    assert digest(new) == new_m["sha256"]
    with pytest.raises(ValueError, match="empty"):
        graft(source, head, tmp_path / "export", "relative")
    (source / "teacher.onnx.data").unlink()
    with pytest.raises(ValueError, match="missing"):
        graft(source, head, tmp_path / "bad-export", "relative")


def test_zero_head_export_preserves_distribution(tmp_path):
    source = tmp_path / "source"
    m = source_bundle(source)
    path = graft(source, phone_head(80, 3), tmp_path / "export", "identity")
    features = np.zeros((45, 80), dtype=np.float32)
    old, _ = stream(session(source / "model.onnx"), m, features)
    got, _ = stream(session(path), m, features)
    np.testing.assert_allclose(old, got, atol=1e-7)


def test_int8_encoder_extraction_and_optimized_export_preserve_source(tmp_path):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    source = tmp_path / "source"
    manifest = source_bundle(source)
    quantized = source / "int8.onnx"
    quantize_dynamic(source / "model.onnx", quantized, weight_type=QuantType.QInt8)
    quantized.replace(source / "model.onnx")
    manifest["sha256"] = digest(source / "model.onnx")
    (source / "manifest.json").write_text(json.dumps(manifest))
    old_graph = onnx.load(source / "model.onnx")
    assert any(n.op_type == "MatMulInteger" for n in old_graph.graph.node)
    m, feature, width, extractor = expose(source, tmp_path / "extraction")
    head = phone_head(width, 3)
    with torch.no_grad():
        head.weight[:, :2] = torch.tensor([[.1, .2], [.3, -.1], [0., .2]])
    path = graft(source, head, tmp_path / "export", "relative")
    features = np.random.default_rng(3).normal(-10., 2., (99, 80)).astype(np.float32)
    original, _ = stream(session(source / "model.onnx"), m, features)
    lp, x = stream(session(extractor), m, features, feature)
    np.testing.assert_array_equal(original, lp)
    with torch.no_grad():
        expected = log_probs(head, dict(features=torch.tensor(x), shipped=torch.tensor(lp))).numpy()
    got, _ = stream(session(path, optimize=True), m, features)
    np.testing.assert_allclose(np.exp(got), np.exp(expected), atol=2e-6, rtol=2e-6)
    new_weights = {t.name: t.SerializeToString() for t in onnx.load(path).graph.initializer}
    assert all(new_weights[t.name] == t.SerializeToString() for t in old_graph.graph.initializer)


def test_cache_to_selected_complete_bundles_uses_only_grouped_training_rows(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace
    import student_articulation_train as training
    monkeypatch.setattr(training, "TOKENS", "ab")
    monkeypatch.setattr(training, "EPOCHS", 1)
    source = tmp_path / "source"
    source_bundle(source)
    prepared = tmp_path / "prepared"
    m, name, width, extraction = expose(source, prepared)
    fbank = np.random.default_rng(7).normal(-10, 2, (45, 80)).astype(np.float32)
    lp, features = stream(session(extraction), m, fbank, name)
    cache = prepared / "view.npz"
    np.savez_compressed(cache, features=features, shipped=lp, target=np.array([1, 2]), fbank=fbank)
    view = dict(file=cache.name, sha256=digest(cache), samples=8000)
    keys = [d + "-" + a for d in training.DISTRIBUTIONS for a in training.ARMS]
    rows = [dict(speaker=s, source=corpus, common=[view] * 3, arms={k: [view] * 3 for k in keys})
            for s in "abcde" for corpus in ("help-none", "transcribed")]
    info = dict(source=str(source), sourceSha256=m["sha256"], tokens="ab", width=width,
                startedAt=time.time(), rows=rows, excludedSpeakers=[],
                groups=training.speaker_partition([r["speaker"] for r in rows]),
                distributions=training.DISTRIBUTIONS,
                auxiliaryHashes=training.insertion_assets(source))
    training.write_json(prepared / "prepared.json", info)
    out = tmp_path / "fit"
    training.fit(SimpleNamespace(prepared=str(prepared), out=str(out)))
    report = json.loads((out / "training-report.json").read_text())
    selected = report["selectedRelative"]
    assert selected in ("mild-relative", "wide-relative")
    assert report["matchedControls"] == [selected.replace("relative", a) for a in ("identity", "global")]
    assert set(report["arms"]) == set(keys)
    for key in keys:
        manifest = json.loads((out / key / "manifest.json").read_text())
        assert digest(out / key / "model.onnx") == manifest["sha256"]
        assert report["arms"][key]["parityRecordings"] == 2
        assert digest(out / key / "teacher.onnx.data") == digest(source / "teacher.onnx.data")
    # Each side's identical data/order/updates gives the same selection metrics.
    assert len({report["arms"][key]["selectedObjective"] for key in keys}) == 1
    cache.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Cached evidence changed"):
        training.load_prepared(prepared)


def test_training_budget_fails_before_another_recording_update():
    import time
    from student_articulation_train import within_budget, DAY
    with pytest.raises(RuntimeError, match="One-day"):
        within_budget(time.time() - DAY - 1)

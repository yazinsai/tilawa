"""Tiny CPU fixtures; actual frozen-model parity must also pass on Modal."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_ONNX = all(importlib.util.find_spec(m) is not None for m in ("torch", "onnx", "onnxruntime"))


@unittest.skipUnless(HAS_TORCH, "requires the Modal training image's torch")
class HeadRefitTest(unittest.TestCase):
    def example(self):
        import torch
        return dict(features=torch.tensor([[1., 0.], [.5, .2], [.2, .7], [0., 1.], [.1, .1]]),
                    shipped=torch.tensor([[3., 0., 0.]]).repeat(5, 1).log_softmax(-1),
                    target=torch.tensor([1, 2]), mask=torch.tensor([True, True, False, False, False]),
                    ordinary=torch.tensor([[.8, .1, .1]]).repeat(5, 1),
                    marginal=torch.tensor([[.2, .6, .2]]).repeat(5, 1))

    def test_only_head_receives_gradients_and_arms_share_ctc_preservation(self):
        import torch
        from student_future_train import phone_head, log_probs, objectives
        head, v = phone_head(2, 3), self.example()
        for k in ("features", "shipped", "ordinary", "marginal"):
            v[k].requires_grad_(True)
        lp = log_probs(head, v)
        torch.testing.assert_close(lp, v["shipped"])
        ordinary, marginal = objectives(lp, v, "ordinary"), objectives(lp, v, "marginal")
        self.assertTrue(torch.equal(ordinary[0], marginal[0]))
        self.assertTrue(torch.equal(ordinary[2], marginal[2]))
        self.assertFalse(torch.equal(ordinary[1], marginal[1]))
        sum(marginal).backward()
        self.assertTrue(all(p.grad is not None for p in head.parameters()))
        self.assertTrue(all(v[k].grad is None for k in ("features", "shipped", "ordinary", "marginal")))
        changed = {**v, "marginal": v["marginal"].detach().clone()}
        changed["marginal"][2:] = torch.tensor([.9, .05, .05])
        self.assertTrue(torch.equal(objectives(lp, changed, "marginal")[1], marginal[1]))

    def test_selection_is_speaker_balanced_and_never_updates_head(self):
        import torch
        from student_future_train import phone_head, validate
        head, v = phone_head(2, 3), self.example()
        second = {**v, "target": torch.tensor([2, 1])}
        rows = [dict(speaker="one", data=v), dict(speaker="two", data=second)]
        before = copy.deepcopy(head.state_dict())
        a, b = validate(head, rows), validate(head, [rows[0]] * 9 + [rows[1]])
        np.testing.assert_allclose([a["objective"], a["per"]], [b["objective"], b["per"]])
        self.assertTrue(all(torch.equal(value, head.state_dict()[key]) for key, value in before.items()))
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        with self.assertRaises(ValueError):
            validate(head, [])


def source_bundle(path):
    import onnx
    from onnx import helper as h, TensorProto as T, numpy_helper as n
    from student_future_export import digest
    path.mkdir()
    nodes = [h.make_node("ReduceMean", ["x"], ["mean"], axes=[1], keepdims=1),
             h.make_node("Expand", ["mean", "shape"], ["feature"]),
             h.make_node("MatMul", ["feature", "weight"], ["scores"], name="ctc_output/MatMul"),
             h.make_node("LogSoftmax", ["scores"], ["log_probs"], axis=-1),
             h.make_node("Add", ["state", "one"], ["new_state"])]
    graph = h.make_graph(nodes, "fixture", [h.make_tensor_value_info("x", T.FLOAT, [1, 45, 80]),
        h.make_tensor_value_info("state", T.INT64, [1])],
        [h.make_tensor_value_info("log_probs", T.FLOAT, [1, 8, 3]), h.make_tensor_value_info("new_state", T.INT64, [1])],
        [n.from_array(np.array([1, 8, 80], dtype=np.int64), "shape"),
         n.from_array(np.random.default_rng(0).normal(0, .01, (80, 3)).astype(np.float32), "weight"),
         n.from_array(np.ones(1, dtype=np.int64), "one")])
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=9), path / "model.onnx")
    teacher = h.make_model(h.make_graph([], "teacher", [], [],
        [n.from_array(np.ones(4, dtype=np.float32), "teacher_weight")]), ir_version=9)
    onnx.save_model(teacher, path / "teacher.onnx", save_as_external_data=True,
                   all_tensors_to_one_file=True, location="teacher.onnx.data", size_threshold=0)
    manifest = dict(file="model.onnx", sha256=digest(path / "model.onnx"), id="unit", tokens="ab", blank=0,
                    frameMs=40, sampleRate=16000, window=45, shift=32, fbank=dict(mels=80, padValue=-23.0259),
                    states=[dict(name="state", shape=[1], dtype="int64")])
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@unittest.skipUnless(HAS_ONNX, "requires Modal's torch, onnx and onnxruntime")
class ExportTest(unittest.TestCase):
    def test_int8_encoder_caches_assets_and_head_parity(self):
        import onnx
        import torch
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from student_future_export import digest, expose, graft, session, stream
        from student_future_train import phone_head, log_probs
        with tempfile.TemporaryDirectory() as directory:
            root, source = Path(directory), Path(directory) / "source"
            m = source_bundle(source)
            quantized = source / "int8.onnx"
            quantize_dynamic(source / "model.onnx", quantized, weight_type=QuantType.QInt8)
            quantized.replace(source / "model.onnx")
            m["sha256"] = digest(source / "model.onnx")
            (source / "manifest.json").write_text(json.dumps(m))
            weights = {t.name: t.SerializeToString() for t in onnx.load(source / "model.onnx").graph.initializer}
            manifest, feature, width, extractor = expose(source, root / "extraction")
            head = phone_head(width, 3)
            with torch.no_grad():
                head.bias.copy_(torch.tensor([.2, -.1, .3]))
                head.weight[:, :2] = torch.tensor([[.1, .2], [.3, -.1], [0., .2]])
            exported = graft(source, head, root / "export", "marginal")
            new = json.loads((exported.parent / "manifest.json").read_text())
            fb = np.random.default_rng(1).normal(-10, 2, (99, 80)).astype(np.float32)
            original, _ = stream(session(source / "model.onnx"), m, fb)
            lp, x = stream(session(extractor), manifest, fb, feature)
            np.testing.assert_array_equal(original, lp)
            got, _ = stream(session(exported, optimize=True), new, fb)
            with torch.no_grad():
                want = log_probs(head, dict(features=torch.tensor(x), shipped=torch.tensor(lp))).numpy()
            np.testing.assert_allclose(np.exp(got), np.exp(want), atol=2e-6, rtol=2e-6)
            for key in ("states", "window", "shift", "fbank", "tokens", "blank", "frameMs", "sampleRate"):
                self.assertEqual(m[key], new[key])
            for name in ("teacher.onnx", "teacher.onnx.data"):
                self.assertEqual(digest(source / name), digest(exported.parent / name))
            updated = {t.name: t.SerializeToString() for t in onnx.load(exported).graph.initializer}
            self.assertTrue(all(updated[key] == value for key, value in weights.items()))
            feeds = dict(x=fb[:45][None], state=np.array([3], dtype=np.int64))
            np.testing.assert_array_equal(session(source / "model.onnx").run(["new_state"], feeds)[0],
                                          session(exported).run(["new_state"], feeds)[0])
            self.assertEqual(digest(exported), new["sha256"])
            self.assertEqual(digest(source / "model.onnx"), m["sha256"])
            (source / "teacher.onnx.data").unlink()
            with self.assertRaises(ValueError):
                graft(source, head, root / "missing-assets", "marginal")


if __name__ == "__main__":
    unittest.main()

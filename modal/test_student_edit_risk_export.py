"""Synthetic native-head export contracts; no benchmark data."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None
from student_edit_risk import phone_head
TORCH = importlib.util.find_spec("torch") is not None
ONNX = np is not None and importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None

@unittest.skipUnless(ONNX, "ONNX and ONNX Runtime required on supervisor")
class ExportTests(unittest.TestCase):
    def graph(self, quantized=False):
        import onnx
        from onnx import helper as h, numpy_helper as nh
        w = np.array([[.2, -.1, .3], [.4, .2, -.2]], dtype=np.float32)
        b = np.array([.1, .2, -.1], dtype=np.float32)
        nodes = [h.make_node("Identity", ["x"], ["feature"], name="encoder"),
                 h.make_node("Add", ["state", "one"], ["new_state"], name="state_update")]
        tensors = [nh.from_array(np.array([1.], np.float32), "one"), nh.from_array(b, "bias")]
        if quantized:
            tensors.extend([nh.from_array(np.rint(w / .1).astype(np.int8), "weight_quantized"),
                            nh.from_array(np.array(.1, np.float32), "weight_scale"),
                            nh.from_array(np.array(0, np.int8), "weight_zero_point")])
            nodes.extend([h.make_node("DynamicQuantizeLinear", ["feature"], ["q_x", "x_scale", "x_zero"]),
                          h.make_node("MatMulInteger", ["q_x", "weight_quantized", "x_zero", "weight_zero_point"], ["raw"], name="ctc_output/MatMulInteger"),
                          h.make_node("Cast", ["raw"], ["cast"], to=onnx.TensorProto.FLOAT),
                          h.make_node("Mul", ["x_scale", "weight_scale"], ["scale"]),
                          h.make_node("Mul", ["cast", "scale"], ["linear"])])
        else:
            tensors.append(nh.from_array(w, "weight"))
            nodes.append(h.make_node("MatMul", ["feature", "weight"], ["linear"], name="ctc_output/MatMul"))
        nodes.extend([h.make_node("Add", ["linear", "bias"], ["logits"]),
                      h.make_node("LogSoftmax", ["logits"], ["log_probs"], axis=-1)])
        graph = h.make_graph(nodes, "synthetic", [h.make_tensor_value_info("x", 1, [1, None, 2]),
            h.make_tensor_value_info("state", 1, [1])], [h.make_tensor_value_info("log_probs", 1, [1, None, 3]),
            h.make_tensor_value_info("new_state", 1, [1])], tensors,
            value_info=[h.make_tensor_value_info("feature", 1, [1, None, 2])])
        return h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=8), w.T, b

    def test_existing_float_and_quantized_weights_are_extracted(self):
        from student_edit_risk_export import head_parameters
        for quantized in (False, True):
            graph, w, b = self.graph(quantized)
            got_w, got_b = head_parameters(graph)
            np.testing.assert_allclose(got_w, w, atol=1e-7)
            np.testing.assert_array_equal(got_b, b)

    def test_replacement_preserves_encoder_and_states_and_removes_old_head(self):
        import onnxruntime as ort
        from student_edit_risk_export import replace_head
        for quantized in (False, True):
            graph, w, b = self.graph(quantized)
            original = copy.deepcopy(graph)
            replaced = replace_head(graph, w + .1, b + .2)
            names = [n.name for n in replaced.graph.node]
            self.assertIn("encoder", names)
            self.assertIn("state_update", names)
            self.assertFalse(any("ctc_output" in name for name in names))
            x = np.array([[[1., 2.], [.2, -.1]]], dtype=np.float32)
            feeds = dict(x=x, state=np.array([5.], np.float32))
            got = ort.InferenceSession(replaced.SerializeToString()).run(None, feeds)
            old = ort.InferenceSession(original.SerializeToString()).run(None, feeds)
            scores = x @ (w + .1).T + b + .2
            expected = scores - np.log(np.exp(scores).sum(-1, keepdims=True))
            np.testing.assert_allclose(got[0], expected, atol=1e-6)
            np.testing.assert_array_equal(got[1], old[1])

    def test_incomplete_teacher_bundle_rejected(self):
        import onnx
        from student_edit_risk_export import insertion_assets
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "lacks"):
                insertion_assets(root)
            graph, _, _ = self.graph()
            onnx.save(graph, str(Path(root) / "teacher.onnx"))
            with self.assertRaisesRegex(ValueError, "data file"):
                insertion_assets(root)

    @unittest.skipUnless(TORCH, "Torch required for bundle test")
    def test_full_bundle_copies_teacher_and_preserves_manifest_contract(self):
        import onnx
        from student_edit_risk_export import digest, graft, insertion_assets
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source"
            source.mkdir()
            graph, w, b = self.graph(True)
            onnx.save(graph, str(source / "model.onnx"))
            teacher, _, _ = self.graph()
            onnx.save_model(teacher, str(source / "teacher.onnx"), save_as_external_data=True,
                            all_tensors_to_one_file=True, location="teacher.onnx.data", size_threshold=0)
            manifest = dict(id="synthetic", file="model.onnx", sha256=digest(source / "model.onnx"),
                tokens="ab", blank=0, frameMs=40, sampleRate=16000, shift=32, window=45,
                fbank=dict(mels=80, padValue=-23.), states=[dict(name="state", shape=[1], dtype="float32")])
            (source / "manifest.json").write_text(json.dumps(manifest))
            before = insertion_assets(source)
            target = root / "export"
            graft(source, phone_head(w + .1, b), target, "risk")
            exported = json.loads((target / "manifest.json").read_text())
            self.assertEqual(exported["sha256"], digest(target / "model.onnx"))
            for key in ("tokens", "blank", "frameMs", "sampleRate", "shift", "window", "fbank", "states"):
                self.assertEqual(exported[key], manifest[key])
            self.assertEqual(insertion_assets(target), before)
            self.assertEqual(digest(source / "model.onnx"), manifest["sha256"])

    @unittest.skipUnless(TORCH, "Torch required for paired refit")
    def test_paired_fit_exports_both_bundles_and_validates_streaming_parity(self):
        import argparse
        import hashlib
        import torch
        import onnx
        from onnx import helper as h, numpy_helper as nh
        from unittest.mock import patch
        import student_edit_risk_train as runner
        from student_edit_risk import speaker_groups
        from student_edit_risk_export import digest, expose, insertion_assets, session, stream
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, cache, output = root / "source", root / "cache", root / "output"
            source.mkdir()
            cache.mkdir()
            graph, _, _ = self.graph()
            vocabulary = len(runner.TOKENS) + 1
            for tensor in graph.graph.initializer:
                if tensor.name == "weight":
                    weight = np.zeros((2, vocabulary), np.float32)
                    weight[:, :3] = nh.to_array(tensor)
                    tensor.CopyFrom(nh.from_array(weight, "weight"))
                elif tensor.name == "bias":
                    bias = np.full(vocabulary, -20., np.float32)
                    bias[:3] = nh.to_array(tensor)
                    tensor.CopyFrom(nh.from_array(bias, "bias"))
            graph.graph.output[0].type.tensor_type.shape.dim[2].dim_value = vocabulary
            # Tiny streaming encoder: emit eight steps from a 45-frame input.
            graph.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 80
            del graph.graph.node[0]
            graph.graph.node.insert(0, h.make_node("Slice", ["x", "starts", "ends", "axes"],
                ["feature"], name="encoder"))
            graph.graph.initializer.extend([nh.from_array(np.array([0, 0], np.int64), "starts"),
                nh.from_array(np.array([8, 2], np.int64), "ends"),
                nh.from_array(np.array([1, 2], np.int64), "axes")])
            onnx.save(graph, str(source / "model.onnx"))
            teacher, _, _ = self.graph()
            onnx.save_model(teacher, str(source / "teacher.onnx"), save_as_external_data=True,
                all_tensors_to_one_file=True, location="teacher.onnx.data", size_threshold=0)
            manifest = dict(id="synthetic", file="model.onnx", sha256=digest(source / "model.onnx"),
                tokens=runner.TOKENS, blank=0, frameMs=40, sampleRate=16000, shift=32, window=45,
                fbank=dict(mels=80, padValue=-23.), states=[dict(name="state", shape=[1], dtype="float32")])
            (source / "manifest.json").write_text(json.dumps(manifest))
            _, feature, width, exposed = expose(source, cache)
            runtime = session(exposed)
            # Speaker identities are chosen for synthetic partitions only.
            speakers = [f"synthetic-{i}" for i in range(20)]
            all_groups = speaker_groups(speakers)
            fit = next(s for s in speakers if all_groups[s] == "fit")
            selection = next(s for s in speakers if all_groups[s] == "selection")
            groups = speaker_groups([fit, selection])
            rows = []
            rng = np.random.default_rng(4)
            for i, speaker in enumerate((fit, selection)):
                row = dict(speaker=speaker, source="synthetic", audioSha256="synthetic",
                    transcriptSha256=hashlib.sha256(runner.TOKENS[:2].encode()).hexdigest(), views=[])
                for v in range(3):
                    fbank = rng.normal(0, .2, (53, 80)).astype(np.float32)
                    shipped, features = stream(runtime, manifest, fbank, feature)
                    file = f"view-{i}-{v}.npz"
                    np.savez_compressed(cache / file, features=features, shipped=shipped,
                        target=np.array([1, 2], np.int64), fbank=fbank)
                    row["views"].append(file)
                row["cacheFiles"] = list(row["views"])
                row["viewHashes"] = [digest(cache / f) for f in row["views"]]
                rows.append(row)
            info = dict(source=str(source), sourceSha256=manifest["sha256"], width=width,
                tokens=runner.TOKENS, rows=rows, groups=groups, excludedSpeakers=[],
                auxiliaryHashes=insertion_assets(source), cpuSeconds=0., wallSeconds=0.)
            (cache / "prepared.json").write_text(json.dumps(info))
            limits = root / "limits.json"
            limits.write_text(json.dumps(dict(preparedSha256=digest(cache / "prepared.json"),
                sourceSha256=manifest["sha256"], groups=groups, excludedSpeakers=[],
                beam=8, retain=8, curves=[], cpuSeconds=0., wallSeconds=0.)))
            with patch.object(runner, "EPOCHS", 1):
                runner.fit(argparse.Namespace(prepared=str(cache), limits=str(limits), out=str(output)))
            report = json.loads((output / "training-report.json").read_text())
            self.assertEqual(set(report["arms"]), {"risk", "ctc"})
            self.assertEqual(report["trainingUpdates"], 1)
            for arm in report["arms"]:
                self.assertEqual(report["arms"][arm]["parityViews"], 3)
                self.assertEqual(insertion_assets(output / arm), insertion_assets(source))
                self.assertTrue((output / arm / "model.onnx").is_file())
            self.assertEqual(digest(source / "model.onnx"), manifest["sha256"])


if __name__ == "__main__":
    unittest.main()

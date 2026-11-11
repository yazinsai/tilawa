"""Analytic timing and attenuation checks, independent of recitation labels."""
import importlib.util
import hashlib
import json
import shutil
import unittest
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

spec = importlib.util.spec_from_file_location("antialias", Path(__file__).parents[1] / "src/antialias_graph.py")
aa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aa)


class AntialiasTest(unittest.TestCase):
    def filter(self, data, axis, stride):
        nodes, constants, output = aa.centered_filter("x", data.ndim, axis, stride, "test")
        graph = helper.make_graph(nodes, "filter", [helper.make_tensor_value_info("x", TensorProto.FLOAT, data.shape)],
            [helper.make_tensor_value_info(output, TensorProto.FLOAT, data.shape)], constants)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
        return ort.InferenceSession(model.SerializeToString()).run(None, {"x": data})[0]

    def test_dc_and_channels(self):
        x = np.ones((1, 3, 31, 7), np.float32) * np.arange(1, 4, dtype=np.float32)[None, :, None, None]
        np.testing.assert_array_equal(self.filter(x, 2, 2), x)

    def test_timing_and_nyquist(self):
        for stride in (2, 4, 8):
            x = np.zeros((65, 1, 2), np.float32)
            x[32, 0, 0] = 1
            y = self.filter(x, 0, stride)
            self.assertEqual(int(y[:, 0, 0].argmax()), 32)
            self.assertEqual(float(np.dot(np.arange(65), y[:, 0, 0])), 32)
            np.testing.assert_array_equal(y[:, 0, 1], 0)
            x[:, 0, 0] = (-1.) ** np.arange(65)
            y = self.filter(x, 0, stride)
            np.testing.assert_array_equal(y[2 * stride:-2 * stride, 0, 0], 0)

    def test_tiny_stream_preserves_weights_and_io(self):
        source = Path(__file__).parent / "fixtures/tiny/model.onnx"
        original = onnx.load(source)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "antialias.onnx"
            report = aa.build(source, target)
            self.assertEqual([s["stride"] for s in report["sites"]], [2, 2, 4, 8, 4, 2, 2])
            modified = onnx.load(target)
            for a, b in zip(original.graph.input, modified.graph.input):
                self.assertEqual(a.SerializeToString(), b.SerializeToString())
            for a, b in zip(original.graph.output, modified.graph.output):
                self.assertEqual(a.SerializeToString(), b.SerializeToString())
            unchanged = {w.name: w.SerializeToString() for w in modified.graph.initializer}
            for w in original.graph.initializer:
                self.assertEqual(w.SerializeToString(), unchanged[w.name])
            session = ort.InferenceSession(str(target))
            manifest = json.loads((source.parent / "manifest.json").read_text())
            state = {s["name"]: np.zeros(s["shape"], np.int64 if s["dtype"] == "int64" else np.float32)
                     for s in manifest["states"]}
            random = np.random.default_rng(0)
            for _ in range(3):
                output = dict(zip([o.name for o in session.get_outputs()], session.run(None,
                    {"x": random.normal(size=(1, manifest["window"], 80)).astype(np.float32), **state})))
                self.assertEqual(output["log_probs"].shape, (1, manifest["shift"] // 4, len(manifest["tokens"]) + 1))
                self.assertTrue(np.isfinite(output["log_probs"]).all())
                for name, data in state.items():
                    self.assertEqual(output["new_" + name].shape, data.shape)
                state = {name: output["new_" + name] for name in state}

    def test_bundle_preserves_primary_and_existing_auxiliary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = Path(tmp) / "primary", Path(tmp) / "side"
            shutil.copytree(Path(__file__).parent / "fixtures/tiny", source)
            (source / "existing-auxiliary.data").write_bytes(b"existing listener asset")
            report = aa.bundle(source, destination)
            for p in source.iterdir():
                self.assertEqual(p.read_bytes(), (destination / p.name).read_bytes())
            self.assertEqual(json.loads((destination / "antialias.json").read_text()), report)
            self.assertEqual(report["sha256"], hashlib.sha256((destination / "antialias.onnx").read_bytes()).hexdigest())
            self.assertEqual(report["sourceSha256"], json.loads((source / "manifest.json").read_text())["sha256"])


if __name__ == "__main__":
    unittest.main()

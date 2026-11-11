import unittest

import numpy as np

from student_robust_biasnorm import build_bundle, calibration, m_energy, training_partition, transform, sites, stream, RECIPE


class RobustScaleTests(unittest.TestCase):
    def test_sparse_outlier_has_bounded_influence(self):
        rng = np.random.default_rng(8)
        ordinary = rng.normal(size=(8, 256)) ** 2
        corrupted = ordinary.copy()
        corrupted[:, 0] = 1e10
        robust = m_energy(corrupted, 1, 2)
        np.testing.assert_allclose(robust, m_energy(ordinary, 1, 2), rtol=0.06)
        self.assertGreater(float(corrupted.mean() / robust.mean()), 1e7)

    def test_scale_equivariance_and_channel_permutation(self):
        energy = np.random.default_rng(1).normal(size=(5, 1, 192)) ** 2
        baseline = m_energy(energy, 2, 2)
        np.testing.assert_allclose(m_energy(energy * 9, 2, 2), baseline * 9)
        np.testing.assert_allclose(m_energy(energy[:, :, ::-1], 2, 2), baseline)
        self.assertTrue(np.isfinite(m_energy(np.zeros_like(energy), 2, 2)).all())

    def test_training_admission_and_speaker_separation(self):
        row = dict(split="train", kind="none", speaker="speaker")
        self.assertIn(training_partition(row), ("fit", "validation"))
        for change in [dict(split="eval"), dict(kind="wrong_word"), dict(held_out=True),
                       dict(exclude=True), dict(events=[{}]), dict(non_errors=[{}]), dict(speaker="")]:
            self.assertIsNone(training_partition({**row, **change}))
        records = []
        for i in range(32):
            records.append({**row, "speaker": str(i), "sites": {
                "norm": {"energy": np.ones((2, 16)), "axis": 1}}})
        result = calibration(records, 2)
        self.assertFalse(set(result["fitSpeakers"]) & set(result["validationSpeakers"]))
        # Changing validation acoustics does not change restoration or envelope.
        for r in records:
            if training_partition(r) == "validation":
                r["sites"]["norm"]["energy"] *= 1e6
        again = calibration(records, 2)
        self.assertEqual(result, again)


def fixture(channels=8):
    import onnx
    from onnx import helper, numpy_helper
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 1, channels])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 1, channels])
    constants = [numpy_helper.from_array(np.arange(channels, dtype=np.float32) / channels, "bias"),
                 numpy_helper.from_array(np.asarray(2, np.float32), "two"),
                 numpy_helper.from_array(np.asarray(-0.5, np.float32), "power"),
                 numpy_helper.from_array(np.asarray(1.7, np.float32), "learned_gain")]
    nodes = [helper.make_node("Sub", ["x", "bias"], ["centered"]),
             helper.make_node("Pow", ["centered", "two"], ["energy"]),
             helper.make_node("ReduceMean", ["energy"], ["mean"], axes=[2], keepdims=1, name="norm/mean"),
             helper.make_node("Pow", ["mean", "power"], ["inverse"]),
             helper.make_node("Mul", ["inverse", "learned_gain"], ["scale"]),
             helper.make_node("Mul", ["x", "scale"], ["y"], name="norm/output")]
    graph = helper.make_graph(nodes, "biased-rms", [x], [y], constants)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


class GraphTests(unittest.TestCase):
    def test_retains_bias_original_numerator_and_learned_gain(self):
        import onnxruntime as ort
        for channels in (8, 9):
            model = fixture(channels)
            cal = {"recipe": RECIPE, "cutoff": 2, "gains": {"norm/mean": 0.8}}
            x = np.random.default_rng(4).normal(size=(2, 1, channels)).astype(np.float32)
            x[0, 0, 0] = 100
            robust = transform(model, cal)
            actual = ort.InferenceSession(robust.SerializeToString()).run(None, {"x": x})[0]
            bias = np.arange(channels, dtype=np.float32) / channels
            expected = x * 1.7 * 0.8 / np.sqrt(m_energy((x - bias) ** 2, 2, 2))
            np.testing.assert_allclose(actual, expected, rtol=2e-6)
            before = {i.name: i.SerializeToString() for i in model.graph.initializer}
            after = {i.name: i.SerializeToString() for i in robust.graph.initializer}
            self.assertTrue(all(after[name] == value for name, value in before.items()))
            self.assertEqual(model.graph.input[0], robust.graph.input[0])
            self.assertEqual(model.graph.output[0], robust.graph.output[0])
            control = transform(model, cal, gain_only=True)
            baseline = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]
            controlled = ort.InferenceSession(control.SerializeToString()).run(None, {"x": x})[0]
            np.testing.assert_allclose(controlled, baseline * 0.8, rtol=2e-6)
            self.assertFalse(any(n.op_type == "TopK" for n in control.graph.node))

    def test_refuses_incomplete_calibration(self):
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            transform(fixture(), {"recipe": RECIPE, "cutoff": 2, "gains": {}})

    def test_actual_tiny_streaming_export_preserves_clock_and_caches(self):
        import json
        from pathlib import Path
        import onnx
        import onnxruntime as ort
        root = Path(__file__).resolve().parents[1] / "packages/sdk/test/fixtures/tiny"
        model = onnx.load(root / "model.onnx")
        found = sites(model)
        self.assertGreater(len(found), 1)
        cal = {"recipe": RECIPE, "cutoff": 2, "gains": {s["name"]: 1 for s in found}}
        side = transform(model, cal)
        self.assertEqual(list(model.graph.input), list(side.graph.input))
        self.assertEqual(list(model.graph.output), list(side.graph.output))
        manifest = json.loads((root / "manifest.json").read_text())
        features = np.random.default_rng(3).normal(size=(140, 80)).astype(np.float32)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        actual = stream(ort.InferenceSession(side.SerializeToString(), options), manifest, features)
        expected = stream(ort.InferenceSession(model.SerializeToString(), options), manifest, features)
        self.assertEqual(expected["log_probs"].shape, actual["log_probs"].shape)
        self.assertTrue(np.isfinite(actual["log_probs"]).all())
        for s in manifest["states"]:
            for cache in actual["new_" + s["name"]]:
                self.assertEqual(list(cache.shape), s["shape"])
                self.assertTrue(np.isfinite(cache).all())

    def test_complete_bundle_calibration_export_and_training_isolation(self):
        import hashlib
        import json
        import shutil
        import tempfile
        from pathlib import Path
        import onnx

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / ".autolab") as temporary:
            source = Path(temporary) / "source"
            shutil.copytree(root / "packages/sdk/test/fixtures/tiny", source)
            # Inert stand-ins test copying, without loading the independent teacher.
            for name in ("teacher.onnx", "teacher.onnx.data"):
                (source / name).write_bytes(b"teacher-copy-fixture")
            features = np.random.default_rng(7).normal(size=(80, 80)).astype(np.float32)
            row = dict(split="train", kind="none", phoneIds=[1, 2], features=features)
            speakers = [next(str(i) for i in range(100) if training_partition({**row, "speaker": str(i)}) == part)
                        for part in ("fit", "validation")]
            records = [{**row, "speaker": speaker} for speaker in speakers]
            records.extend([{**row, "speaker": "held-out"},
                            {**row, "speaker": "held-out", "split": "eval"}])
            destination = Path(temporary) / "bundle"
            declaration = build_bundle(source, destination, records)
            self.assertNotIn("held-out", declaration["fitSpeakers"] + declaration["validationSpeakers"])
            self.assertEqual(len(declaration["validationTrials"]), 3)
            self.assertEqual((source / "manifest.json").read_bytes(), (destination / "manifest.json").read_bytes())
            for name in ("model.onnx", "teacher.onnx", "teacher.onnx.data"):
                self.assertEqual((source / name).read_bytes(), (destination / name).read_bytes())
            for name, key in (("robust-norm.onnx", "sha256"), ("robust-norm-gain.onnx", "gainSha256")):
                data = (destination / name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), declaration[key])
                onnx.checker.check_model(onnx.load(destination / name))
            self.assertEqual(json.loads((destination / "robust-norm.json").read_text()), declaration)

    def test_bundle_refuses_missing_existing_insertion_assets(self):
        import shutil
        import tempfile
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / ".autolab") as temporary:
            source = Path(temporary) / "source"
            shutil.copytree(root / "packages/sdk/test/fixtures/tiny", source)
            with self.assertRaisesRegex(FileNotFoundError, "insertion asset"):
                build_bundle(source, Path(temporary) / "bundle", [])


if __name__ == "__main__":
    unittest.main()

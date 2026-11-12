import unittest
import hashlib
import json
import tempfile
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from student_paired_gradient import StreamingHead, gradient_subspaces, project_update, shuffled_pairs, speaker_partition, train_heads


class PairedGradientTests(unittest.TestCase):
    def test_reproducible_direction_beats_independent_noise(self):
        rng = np.random.default_rng(3)
        shared = rng.normal(size=300)
        a = np.column_stack([shared, rng.normal(size=(300, 3)) * 4])
        b = np.column_stack([shared, rng.normal(size=(300, 3)) * 4])
        paired, shuffled, report = gradient_subspaces(a, b, max_rank=2, sketch_dim=4)
        self.assertEqual(report["rank"], 1)
        self.assertEqual(paired.shape, shuffled.shape)
        self.assertGreater(abs(paired[0, 0]), .99)
        # Projection retains the repeatable distinction and rejects view noise.
        update = project_update(np.array([1., 5., 5., 5.]), paired)
        self.assertLess(np.linalg.norm(update[1:]), .15)
        np.testing.assert_allclose(paired.T @ paired, np.eye(1), atol=1e-12)

    def test_identical_views_and_zero_signal(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(100, 4))
        paired, _, report = gradient_subspaces(a, a, max_rank=3, sketch_dim=4)
        self.assertEqual(report["rank"], 3)
        v = np.arange(4.)
        p = project_update(v, paired)
        np.testing.assert_allclose(project_update(p, paired), p, atol=1e-12)
        self.assertLessEqual(np.linalg.norm(p), np.linalg.norm(v) + 1e-12)
        paired, shuffled, report = gradient_subspaces(a, -a, max_rank=3, sketch_dim=4)
        self.assertEqual(report["rank"], 0)
        np.testing.assert_array_equal(project_update(v, paired), np.zeros(4))
        self.assertEqual(shuffled.shape, (4, 0))

    def test_recording_shuffling_never_reuses_true_pair(self):
        p = shuffled_pairs(51)
        self.assertEqual(set(p), set(range(51)))
        self.assertTrue(np.all(p != np.arange(51)))

    def test_speaker_split_is_grouped_and_order_independent(self):
        speakers = [str(i % 10) for i in range(40)]
        split = speaker_partition(speakers)
        for speaker in speakers:
            self.assertEqual(len(set(split[np.array(speakers) == speaker])), 1)
        np.testing.assert_array_equal(speaker_partition(speakers[::-1]), split[::-1])
        with self.assertRaises(ValueError):
            speaker_partition(["only-one"] * 10)

    def test_invalid_gradients_fail_before_selection(self):
        with self.assertRaises(ValueError):
            gradient_subspaces(np.ones((4, 3)), np.ones((3, 3)))
        with self.assertRaises(ValueError):
            gradient_subspaces(np.full((4, 3), np.nan), np.ones((4, 3)))


class StreamingExportTests(unittest.TestCase):
    def test_integer_parent_parity_and_export_interface(self):
        import onnx
        import onnxruntime as ort
        from onnx import helper, numpy_helper

        rng = np.random.default_rng(9)
        wq = rng.integers(-80, 80, size=(4, 3), dtype=np.int8)
        init = [numpy_helper.from_array(wq, "wq"),
                numpy_helper.from_array(np.array(0, dtype=np.int8), "wzero"),
                numpy_helper.from_array(np.array(.01, dtype=np.float32), "wscale"),
                numpy_helper.from_array(np.array([.1, -.2, .3], dtype=np.float32), "bias")]
        nodes = [helper.make_node("DynamicQuantizeLinear", ["x"], ["xq", "xscale", "xzero"]),
                 helper.make_node("MatMulInteger", ["xq", "wq", "xzero", "wzero"], ["integer"]),
                 helper.make_node("Cast", ["integer"], ["float"], to=onnx.TensorProto.FLOAT),
                 helper.make_node("Mul", ["xscale", "wscale"], ["scale"]),
                 helper.make_node("Mul", ["float", "scale"], ["scaled"]),
                 helper.make_node("Add", ["scaled", "bias"], ["logits"]),
                 helper.make_node("LogSoftmax", ["logits"], ["log_probs"], axis=-1)]
        graph = helper.make_graph(nodes, "test", [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 8, 4])],
                                  [helper.make_tensor_value_info("log_probs", onnx.TensorProto.FLOAT, [1, 8, 3])], init)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            parent = root / "parent"
            parent.mkdir()
            onnx.save(model, parent / "model.onnx")
            manifest = {"file": "model.onnx", "sha256": hashlib.sha256((parent / "model.onnx").read_bytes()).hexdigest(),
                        "tokens": "ab", "chunk": 16, "window": 8, "shift": 8, "states": [],
                        "fbank": {"padValue": -23.0259}, "frameMs": 40, "format": "onnx-int8"}
            (parent / "manifest.json").write_text(json.dumps(manifest))
            (parent / "auxiliary.bin").write_bytes(b"preserved auxiliary")
            head = StreamingHead(parent)
            x = rng.normal(size=(1, 8, 4)).astype(np.float32)
            original, features = ort.InferenceSession(head.feature_graph).run(None, {"x": x})
            logits = features @ head.weight + head.bias
            lp = logits - np.logaddexp.reduce(logits, axis=-1)[..., None]
            np.testing.assert_allclose(lp, original, atol=2e-5)
            output = root / "refit"
            exported = head.export(output, head.weight, head.bias)
            session = ort.InferenceSession(str(output / "model.onnx"))
            self.assertEqual([o.name for o in session.get_outputs()], ["log_probs"])
            np.testing.assert_allclose(session.run(None, {"x": x})[0], original, atol=2e-5)
            self.assertEqual(exported["frameMs"], 40)
            self.assertEqual(exported["tokens"], "ab")
            self.assertEqual((output / "auxiliary.bin").read_bytes(), b"preserved auxiliary")
            self.assertEqual(hashlib.sha256((output / "model.onnx").read_bytes()).hexdigest(), exported["sha256"])
            with self.assertRaises(ValueError):
                head.export(parent, head.weight, head.bias)


class TrainingIntegrationTests(unittest.TestCase):
    def test_matched_refits_remain_in_the_frozen_selected_subspaces(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from student import TOKENS, VOCAB
        rng = np.random.default_rng(43)
        weight = rng.normal(0, .1, size=(4, VOCAB)).astype(np.float32)
        bias = np.zeros(VOCAB, dtype=np.float32)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            cache = Path(temporary) / "cache"
            cache.mkdir()
            (cache / "identity.json").write_text(json.dumps({"source": "synthetic"}))
            np.savez(cache / "parent.npz", weight=weight, bias=bias)
            parent_bytes = (cache / "parent.npz").read_bytes()
            metadata = []
            speakers = [f"speaker-{i // 2}" for i in range(20)]
            split = speaker_partition(speakers)
            for i, speaker in enumerate(speakers):
                shared = rng.normal(size=(12, 4)).astype(np.float32)
                views = {}
                for view in range(3):
                    features = shared + rng.normal(0, .005, size=shared.shape).astype(np.float32)
                    logits = features @ weight + bias
                    views[f"features{view}"] = features
                    views[f"parent{view}"] = logits - np.logaddexp.reduce(logits, axis=1)[:, None]
                np.savez(cache / f"{i}.npz", **views)
                metadata.append({"speaker": speaker, "target": TOKENS[:2], "file": f"{i}.npz",
                                 "validation": bool(split[i]), "source": "synthetic"})
            (cache / "metadata-0.json").write_text(json.dumps(metadata))
            output = Path(temporary) / "heads"
            train_heads(SimpleNamespace(cache=cache, output=output, epochs=2))
            report = json.loads((output / "selection.json").read_text())
            self.assertFalse(set(report["training_speakers"]) & set(report["validation_speakers"]))
            self.assertEqual(report["refit_gradient_evaluations_per_arm"], 2 * sum(~split) * 3)
            bases = np.load(output / "subspaces.npz")
            initial = np.concatenate([weight.reshape(-1), bias])
            for arm in ("paired", "shuffled"):
                selected = np.load(output / f"{arm}.npz")
                delta = np.concatenate([selected["weight"].reshape(-1), selected["bias"]]) - initial
                np.testing.assert_allclose(project_update(delta, bases[arm]), delta, atol=2e-7)
            self.assertEqual((cache / "parent.npz").read_bytes(), parent_bytes)


if __name__ == "__main__":
    unittest.main()

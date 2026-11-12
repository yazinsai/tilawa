"""Numerical tests: python -m unittest discover -s tools -p test_student_phase_consistency.py."""

import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal"))
from student_phase_consistency import (authorized_none, consistency_loss, occurrence_distributions,
                                       occurrence_weights, origins, output_frames, paired_loss, speaker_partition)


class PhaseGeometryTests(unittest.TestCase):
    def test_half_chunk_clock_preserves_every_valid_output(self):
        m = dict(shift=32, window=45, frameMs=40, fbank=dict(frameShiftMs=10))
        self.assertEqual(origins(m), (0, 16))
        for n in range(13, 200):
            self.assertEqual(output_frames(n + 16) - 4, output_frames(n))
        with self.assertRaises(ValueError):
            origins({**m, "shift": 30})

    def test_speakers_are_grouped_and_selection_is_order_independent(self):
        speakers = ["alice", "bob", "charlie", "dina", "elias", "farah"]
        split = speaker_partition(speakers)
        self.assertEqual(split, speaker_partition(speakers[::-1] + speakers))
        self.assertEqual(list(split.values()).count("validation"), 2)

    def test_only_reliable_training_none_takes(self):
        row = dict(split="train", kind="none", speaker="learner", ayahs=["1:1"])
        self.assertTrue(authorized_none(row))
        for change in (dict(split="eval"), dict(kind="repeat"), dict(kind="vowel"),
                       dict(excluded="review"), dict(events=[{}]), dict(non_errors=[{}]),
                       dict(unsupported=[{}])):
            self.assertFalse(authorized_none({**row, **change}))

    def test_catalog_rejects_held_out_speaker_even_with_a_train_take(self):
        import json
        import tempfile
        from student_phase_head import training_rows
        from student import phoneme_ids

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            labels = [dict(id="clean", speaker="safe", split="train", kind="none", ayahs=["1:1", "1:2"], wav="clean.wav"),
                      dict(id="leak", speaker="held-out", split="train", kind="none", ayahs=["1:1"], wav="leak.wav"),
                      dict(id="eval", speaker="held-out", split="eval", kind="none", ayahs=["1:1"], wav="eval.wav")]
            (root / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in labels))
            reference = dict(ayahs=[dict(s=1, a=1, p="ببَ"), dict(s=1, a=2, p="َم")])
            rows = training_rows(root, reference)
            self.assertEqual([r["id"] for r in rows], ["clean"])
            self.assertEqual(rows[0]["target"], phoneme_ids("بََم"))


class OccurrenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def test_forward_backward_matches_exhaustive_paths_with_equal_phones(self):
        torch = self.torch
        torch.manual_seed(11)
        lp = torch.randn(5, 3, dtype=torch.double).log_softmax(-1)
        # [blank, 1, blank, 1, blank]: equal adjacent occurrences cannot skip blank.
        labels = [0, 1, 0, 1, 0]
        masses = torch.zeros(2, 5, dtype=torch.double)
        z = 0.0
        for path in itertools.product(range(5), repeat=5):
            if path[0] not in (0, 1) or path[-1] not in (3, 4):
                continue
            if any(b - a not in (0, 1) for a, b in zip(path, path[1:])):
                continue
            weight = sum(lp[t, labels[s]] for t, s in enumerate(path)).exp().item()
            z += weight
            for t, s in enumerate(path):
                if s % 2:
                    masses[s // 2, t] += weight
        self.assertGreater(z, 0)
        expected = masses / masses.sum(-1, keepdim=True)
        torch.testing.assert_close(occurrence_weights(lp, [1, 1]), expected)
        self.assertFalse(torch.allclose(expected[0], expected[1]))

    def test_distinct_phones_allow_skip_transitions(self):
        torch = self.torch
        torch.manual_seed(19)
        lp = torch.randn(4, 3, dtype=torch.double).log_softmax(-1)
        labels = [0, 1, 0, 2, 0]
        masses = torch.zeros(2, 4, dtype=torch.double)
        for path in itertools.product(range(5), repeat=4):
            if path[0] not in (0, 1) or path[-1] not in (3, 4):
                continue
            if any(not (b == a or b == a + 1 or (b == a + 2 and labels[b] != 0
                                                and labels[b] != labels[a]))
                   for a, b in zip(path, path[1:])):
                continue
            weight = sum(lp[t, labels[s]] for t, s in enumerate(path)).exp().item()
            for t, s in enumerate(path):
                if s % 2:
                    masses[s // 2, t] += weight
        torch.testing.assert_close(occurrence_weights(lp, [1, 2]), masses / masses.sum(-1, keepdim=True))

    def test_timing_shift_has_no_identity_penalty(self):
        torch = self.torch
        # Same phone distribution appears at different frames amid blank padding.
        a = torch.tensor([[1., 0., 0.], [.1, .7, .2], [1., 0., 0.]], dtype=torch.double).log()
        b = torch.tensor([[1., 0., 0.], [1., 0., 0.], [.1, .7, .2],
                          [1., 0., 0.], [1., 0., 0.]], dtype=torch.double).log()
        torch.testing.assert_close(occurrence_distributions(a, [1]), occurrence_distributions(b, [1]))
        self.assertAlmostEqual(consistency_loss(a, b, [1]).item(), 0)

    def test_consistency_is_not_vacuous_and_reaches_head(self):
        torch = self.torch
        a = torch.tensor([[0., 3., 0.], [2., 0., 0.]], requires_grad=True)
        b = torch.tensor([[0., 2., 2.], [2., 0., 0.]], requires_grad=True)
        loss = consistency_loss(a.log_softmax(-1), b.log_softmax(-1), [1])
        self.assertGreater(loss.item(), .01)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(a.grad.abs().sum().item(), 0)
        self.assertGreater(b.grad.abs().sum().item(), 0)

    def test_empty_and_impossible_transcripts(self):
        torch = self.torch
        lp = torch.randn(2, 3).log_softmax(-1)
        self.assertEqual(consistency_loss(lp, lp, []).item(), 0)
        with self.assertRaises(ValueError):
            occurrence_weights(lp, [1, 1])
        with self.assertRaises(ValueError):
            occurrence_weights(lp, [0])

    def test_matched_refit_differs_only_by_consistency(self):
        torch = self.torch
        torch.manual_seed(4)
        head = torch.nn.Linear(3, 3)
        views = [dict(features=torch.randn(5, 3), log_probs=torch.randn(5, 3).log_softmax(-1))
                 for _ in range(2)]
        control = paired_loss(head, views, [1, 2], 0)
        consistent = paired_loss(head, views, [1, 2], 1)
        torch.testing.assert_close(control["ctc"], consistent["ctc"])
        torch.testing.assert_close(control["preservation"], consistent["preservation"])
        torch.testing.assert_close(consistent["loss"] - control["loss"], consistent["consistency"])
        # Changing a blank-dominated pause changes preservation in both arms.
        changed = [{**v, "log_probs": torch.tensor([[8., 0., 0.]]).expand(5, 3).log_softmax(-1)} for v in views]
        self.assertNotEqual(paired_loss(head, changed, [1, 2], 0)["preservation"].item(),
                            control["preservation"].item())

    def test_trainer_selects_a_common_update_count(self):
        import argparse
        import json
        import tempfile
        import time
        import numpy as np
        from student_phase_head import digest, train

        torch = self.torch
        torch.manual_seed(31)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            torch.save(dict(weight=np.zeros((3, 3), np.float32), bias=np.zeros(3, np.float32)), root / "initial.pt")
            rows = []
            for i, part in enumerate(("train", "train", "validation")):
                views = [dict(features=torch.randn(5, 3).numpy(),
                              log_probs=torch.randn(5, 3).log_softmax(-1).numpy()) for _ in range(2)]
                filename = f"record-{i}.pt"
                torch.save(dict(target=[1, 2], views=views), root / filename)
                rows.append(dict(file=filename, partition=part, cache_sha256=digest(root / filename)))
            index = dict(recordings=rows, source_sha256="test-source", started_at=time.time())
            (root / "index.json").write_text(json.dumps(index))
            fit = root / "fit"
            train(argparse.Namespace(cache=str(root), out=str(fit), threads=1, seed=0, lr=1e-4, hours=1, epochs=2))
            control = torch.load(fit / "control.pt", weights_only=False)
            consistent = torch.load(fit / "consistent.pt", weights_only=False)
            self.assertEqual(control["selected_updates"], consistent["selected_updates"])
            self.assertEqual(control["selected_epoch"], consistent["selected_epoch"])
            history = json.loads((fit / "training.json").read_text())["history"]
            self.assertEqual([r["updates"] for r in history], [2, 4])


class ExportedEncoderTests(unittest.TestCase):
    def test_dynamic_quantized_head_and_original_time_mapping(self):
        import hashlib
        import argparse
        import json
        import tempfile
        import numpy as np
        import onnx
        import torch
        from onnx import helper as h, numpy_helper as nh, TensorProto as t
        from student_phase_head import Encoder, export, readout

        arrays = {
            "starts": np.array([0, 0], np.int64), "ends": np.array([32, 3], np.int64),
            "axes": np.array([1, 2], np.int64), "steps": np.array([4, 1], np.int64),
            "phone_weight_quantized": np.array([[2, -1, 3], [1, 2, -2], [-1, 0, 1]], np.int8),
            "phone_weight_zero_point": np.array(0, np.int8), "phone_weight_scale": np.array(.25, np.float32),
            "bias": np.array([.1, -.2, .3], np.float32), "one": np.array([1], np.int64),
        }
        nodes = [
            h.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["hidden"]),
            h.make_node("DynamicQuantizeLinear", ["hidden"], ["q", "scale", "zero"]),
            h.make_node("MatMulInteger", ["q", "phone_weight_quantized", "zero", "phone_weight_zero_point"], ["integer"]),
            h.make_node("Cast", ["integer"], ["float"], to=t.FLOAT),
            h.make_node("Mul", ["scale", "phone_weight_scale"], ["product_scale"]),
            h.make_node("Mul", ["float", "product_scale"], ["linear"]),
            h.make_node("Add", ["linear", "bias"], ["logits"]),
            h.make_node("LogSoftmax", ["logits"], ["log_probs"], axis=-1),
            h.make_node("Add", ["processed", "one"], ["new_processed"]),
        ]
        graph = h.make_graph(nodes, "head-fixture", [h.make_tensor_value_info("x", t.FLOAT, [1, 45, 80]),
                             h.make_tensor_value_info("processed", t.INT64, [1])],
                            [h.make_tensor_value_info("log_probs", t.FLOAT, [1, 8, 3]),
                             h.make_tensor_value_info("new_processed", t.INT64, [1])],
                            [nh.from_array(value, name) for name, value in arrays.items()])
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
        model.ir_version = 9
        feature, weight, bias, _ = readout(model.graph, 3)
        self.assertEqual(feature, "hidden")
        np.testing.assert_array_equal(weight, arrays["phone_weight_quantized"].T * .25)
        np.testing.assert_array_equal(bias, arrays["bias"])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            data = model.SerializeToString()
            (root / "model.onnx").write_bytes(data)
            manifest = dict(file="model.onnx", sha256=hashlib.sha256(data).hexdigest(), tokens="ab",
                            shift=32, chunk=16, window=45, frameMs=40,
                            fbank=dict(frameShiftMs=10, padValue=-23.0259),
                            states=[dict(name="processed", shape=[1], dtype="int64")])
            (root / "manifest.json").write_text(json.dumps(manifest))
            encoder = Encoder(root, 1)
            f = np.arange(100 * 80, dtype=np.float32).reshape(100, 80) / 8000
            first, second = [encoder.stream(f, phase) for phase in (0, 16)]
            expected = f[:output_frames(len(f)) * 4:4, :3]
            np.testing.assert_array_equal(first["features"], expected)
            np.testing.assert_array_equal(second["features"], expected)
            np.testing.assert_array_equal(first["times_ms"], second["times_ms"])
            # Export and re-open an actually changed head; all encoder states and
            # the interface must survive, including dynamic quantization nodes.
            fit_dir = root / "fit"
            fit_dir.mkdir()
            checkpoint = dict(head={"weight": torch.from_numpy(weight + .02), "bias": torch.from_numpy(bias)},
                              source_sha256=manifest["sha256"], selected_updates=1, selected_epoch=1,
                              arm="consistent", cache_sha256="unit-test")
            torch.save(checkpoint, fit_dir / "consistent.pt")
            bundle = root / "bundle"
            export(argparse.Namespace(model=str(root), fit=str(fit_dir), out=str(bundle),
                                      arm="consistent", threads=1, required_assets=[]))
            got = json.loads((bundle / "manifest.json").read_text())
            for key in ("tokens", "shift", "chunk", "window", "frameMs", "fbank", "states"):
                self.assertEqual(got[key], manifest[key])
            self.assertEqual(got["sha256"], hashlib.sha256((bundle / "model.onnx").read_bytes()).hexdigest())
            unchanged = {i.name: nh.to_array(i) for i in onnx.load(bundle / "model.onnx").graph.initializer}
            for name, value in arrays.items():
                np.testing.assert_array_equal(unchanged[name], value)


if __name__ == "__main__":
    unittest.main()

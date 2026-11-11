"""Synthetic numerical and contract tests; no practice recordings or labels.

Run on the supervisor: python3 -m unittest discover -s modal -p test_student_curvature.py -v
NumPy-only tests can also run locally; dependency skips are reported explicitly.
"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from student_curvature import (KroneckerPreconditioner, view_factors, speaker_partition,
    admitted_none, phone_head, log_probs, objectives, estimate_factors, matched_refit)

TORCH = importlib.util.find_spec("torch") is not None
ONNX = importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None


class CurvatureTests(unittest.TestCase):
    def test_matches_dense_damped_kronecker_solve_including_bias(self):
        a = np.array([[4., 1., .2], [1., 2., .3], [.2, .3, 1.]])
        g = np.array([[.4, -.2], [-.2, .4]])
        grad = np.array([[1., 3., -2.], [-1., 2., .7]])
        solver = KroneckerPreconditioner(a, g, .1)
        dense = np.kron(g, a) + .1 * solver.scale * np.eye(6)
        expected = np.linalg.solve(dense, grad.ravel()).reshape(grad.shape) * solver.scale
        np.testing.assert_allclose(solver.apply(grad), expected, atol=1e-12)

    def test_singular_softmax_and_correlated_feature_directions_stay_finite(self):
        a = np.array([[1., 1.], [1., 1.]])
        g = np.array([[.25, -.25], [-.25, .25]])
        gradient = np.array([[2., -2.], [2., -2.]])
        solver = KroneckerPreconditioner(a, g, .2)
        np.testing.assert_allclose(solver.apply(gradient), gradient / .2, atol=1e-12)

    def test_preconditioner_recovers_weak_feature_direction(self):
        solver = KroneckerPreconditioner(np.diag([100., 1.]), np.eye(2), .1)
        result = solver.apply(np.ones((2, 2)))
        self.assertGreater(result[0, 1] / result[0, 0], 10.)

    def test_invalid_curvature_rejected(self):
        for a, g, ratio in [(np.eye(2), np.eye(2), 0.), (np.eye(2), np.eye(2), float("nan")),
                            (np.diag([-1., 1.]), np.eye(2), 1.),
                            (np.array([[1., .2], [0., 1.]]), np.eye(2), 1.),
                            (np.eye(2), np.zeros((2, 2)), 1.)]:
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                KroneckerPreconditioner(a, g, ratio)

    def test_factor_bias_cross_terms_and_categorical_covariance(self):
        a, g = view_factors([[1., 2.], [3., 4.]], [[.25, .75], [.5, .5]])
        np.testing.assert_allclose(a[-1], [2., 3., 1.])
        np.testing.assert_allclose(g, [[.21875, -.21875], [-.21875, .21875]])
        np.testing.assert_allclose(g.sum(0), 0.)

    def test_invalid_probabilities_rejected(self):
        for p in ([[.2, .2]], [[-1., 2.]], [[float("nan"), 1.]]):
            with self.assertRaises(ValueError):
                view_factors([[1.]], p)

    def test_bad_gradient_rejected(self):
        solver = KroneckerPreconditioner(np.eye(2), np.eye(3), 1.)
        for grad in (np.zeros((2, 3)), np.full((3, 2), np.inf)):
            with self.assertRaises(ValueError):
                solver.apply(grad)


class EnrollmentTests(unittest.TestCase):
    def test_speaker_split_is_order_invariant_and_groups_all_takes(self):
        speakers = ["synthetic-" + str(i) for i in range(10)]
        a = speaker_partition(speakers)
        self.assertEqual(a, speaker_partition(list(reversed(speakers)) + speakers))
        self.assertEqual(list(a.values()).count("selection"), 2)
        with self.assertRaises(ValueError):
            speaker_partition(["only"])

    def test_holdout_repeat_and_ambiguous_none_are_excluded(self):
        good = dict(id="synthetic", split="train", kind="none", speaker="fit", scored=True,
                    events=[], non_errors=[], unsupported=[], wav="x.wav", ayahs=["synthetic"])
        bad = [dict(good, id="repeat", non_errors=[{"kind": "repeat"}]),
               dict(good, id="script", kind="skipped_word"),
               dict(good, id="unscored", scored=False),
               dict(good, id="excluded", excluded=True)]
        self.assertEqual(admitted_none([good, *bad]), [good])
        self.assertEqual(admitted_none([good, dict(good, split="held-out")]), [])
        self.assertEqual(admitted_none([good], ["fit"]), [])

    def test_duplicate_none_rejected(self):
        row = dict(id="same", split="train", kind="none", speaker="fit", scored=True,
                   events=[], non_errors=[], unsupported=[], wav="x.wav", ayahs=["synthetic"])
        with self.assertRaises(ValueError):
            admitted_none([row, row])


@unittest.skipUnless(TORCH, "Torch required on supervisor")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        self.head = phone_head(np.array([[.1, .2], [-.2, .3], [.3, -.1]], np.float32), np.zeros(3, np.float32))

    def view(self, n=8, offset=0.):
        t = self.torch
        x = t.arange(n * 2, dtype=t.float32).reshape(n, 2) / 10 + offset
        frozen = log_probs(self.head, x).detach()
        return dict(features=x, shipped=frozen, teacher=frozen.clone(), target=t.tensor([1, 2]))

    def test_no_encoder_or_teacher_gradient_and_preservation_is_zero_at_origin(self):
        v = self.view()
        v["features"].requires_grad_()
        v["teacher"].requires_grad_()
        ctc, kd, preserve = objectives(log_probs(self.head, v["features"]), v["target"], v["teacher"], v["shipped"])
        self.assertAlmostEqual(float(preserve.detach()), 0., places=7)
        (ctc + kd + preserve).backward()
        self.assertIsNone(v["features"].grad)
        self.assertIsNone(v["teacher"].grad)
        self.assertGreater(float(self.head.bias.grad.abs().sum()), 0.)

    def test_impossible_repeated_target_is_rejected_before_ctc(self):
        v = self.view(n=2)
        with self.assertRaises(ValueError):
            objectives(v["shipped"], self.torch.tensor([1, 1]), v["teacher"], v["shipped"])

    def test_curvature_ignores_selection_and_uses_equal_recording_weight(self):
        rows = [dict(speaker="fit", views=[self.view(n=8)]),
                dict(speaker="fit", views=[self.view(n=80, offset=1.)]),
                dict(speaker="selection", views=[self.view(offset=1000.)])]
        groups = dict(fit="fit", selection="selection")
        got = estimate_factors(self.head, rows, groups)
        expected = [view_factors(r["views"][0]["features"], r["views"][0]["shipped"].exp()) for r in rows[:2]]
        for i in range(2):
            np.testing.assert_allclose(got[i], np.mean([e[i] for e in expected], axis=0))
        rows[-1]["views"] = []
        altered = estimate_factors(self.head, rows, groups)
        np.testing.assert_array_equal(altered[0], got[0])

    def test_both_exports_share_update_count_examples_and_objective(self):
        import student_curvature as module
        rows = [dict(speaker="fit", views=[self.view(offset=i / 10)]) for i in range(2)]
        rows.append(dict(speaker="selection", views=[self.view()]))
        groups = dict(fit="fit", selection="selection")
        calls, validation_calls = [], []
        original = module.recording_loss
        def record(head, views):
            calls.append(id(views))
            return original(head, views)
        def validate(head, selection):
            self.assertEqual([r["speaker"] for r in selection], ["selection"])
            # Force the shared selected epoch to 2; training still completes 3.
            epoch = len(validation_calls) // 3
            validation_calls.append(epoch)
            return dict(objective=[10., 5., 1., 4.][epoch], per=0.)
        with patch.object(module, "recording_loss", side_effect=record):
            heads, report, _ = matched_refit(self.head, rows, groups, validate, (.1, 1.), 3, .001, 42)
        self.assertEqual(report["selectedEpoch"], 2)
        self.assertEqual(report["selectedUpdates"], 4)
        self.assertEqual(report["trainingUpdates"], 6)
        self.assertEqual(len(calls), 18)
        for epoch in range(3):
            sequences = [calls[epoch * 6 + arm * 2:epoch * 6 + arm * 2 + 2] for arm in range(3)]
            self.assertEqual(sequences[0], sequences[1])
            self.assertEqual(sequences[0], sequences[2])
        self.assertEqual(set(heads), {"ordinary", "curvature"})

    def test_additional_transcript_holdout_excludes_none_speaker_globally(self):
        from student_curvature_train import recordings
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            none = dict(id="none", split="train", kind="none", speaker="held", scored=True,
                        events=[], non_errors=[], unsupported=[], wav="a.wav", ayahs=["synthetic:ayah"])
            (root / "labels.jsonl").write_text(json.dumps(none))
            (root / "transcribed.jsonl").write_text(json.dumps(dict(speaker="held", split="release")))
            with self.assertRaisesRegex(ValueError, "Both authorized"):
                recordings(root / "labels.jsonl", root / "transcribed.jsonl", [],
                           {"ayahs": [{"s": "synthetic", "a": "ayah", "p": "a"}]})


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
        from student_curvature_export import head_parameters
        for quantized in (False, True):
            graph, w, b = self.graph(quantized)
            got_w, got_b = head_parameters(graph)
            np.testing.assert_allclose(got_w, w, atol=1e-7)
            np.testing.assert_array_equal(got_b, b)

    def test_replacement_preserves_encoder_and_states_and_removes_old_head(self):
        import onnxruntime as ort
        from student_curvature_export import replace_head
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
        from student_curvature_export import insertion_assets
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
        from student_curvature_export import digest, graft, insertion_assets
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
            graft(source, phone_head(w + .1, b), target, "curvature")
            exported = json.loads((target / "manifest.json").read_text())
            self.assertEqual(exported["sha256"], digest(target / "model.onnx"))
            for key in ("tokens", "blank", "frameMs", "sampleRate", "shift", "window", "fbank", "states"):
                self.assertEqual(exported[key], manifest[key])
            self.assertEqual(insertion_assets(target), before)
            self.assertEqual(digest(source / "model.onnx"), manifest["sha256"])


if __name__ == "__main__":
    unittest.main()

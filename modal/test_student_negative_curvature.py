"""Tiny generated curvature, control and export checks; no evaluation audio."""
import copy
import itertools
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from student_negative_curvature import (admitted_none, speaker_partition, phone_head,
    log_probs, ctc_statistics, corpus_evaluate, smallest_curvature, trust_candidates,
    matched_refit, first_order_start, CorpusOracle)


def collapse(path):
    return [x for i, x in enumerate(path) if x and (i == 0 or x != path[i - 1])]


def fixture():
    head = phone_head(np.array([[.1, -.1], [.4, .2], [-.2, .3]]), np.array([.1, 0., -.1]))
    x = np.array([[1., 0.], [.3, .5], [-.2, 1.], [.4, -.3]])
    lp = log_probs(head, x)
    view = dict(features=x, target=np.array([1, 2]), teacher=lp.copy(), shipped=lp.copy())
    rows = [dict(speaker="fit", views=[copy.deepcopy(view), copy.deepcopy(view)]),
            dict(speaker="selection", views=[copy.deepcopy(view)])]
    return head, rows


class CurvatureTests(unittest.TestCase):
    def test_path_mass_and_sensitivity_match_exhaustive_paths(self):
        rng = np.random.default_rng(7)
        logits, perturbation = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
        lp = logits - logsumexp(logits, axis=1, keepdims=True)
        tangent = perturbation - (np.exp(lp) * perturbation).sum(1, keepdims=True)
        for target in ([], [1], [1, 1], [1, 2], [2, 1, 2]):
            paths = [p for p in itertools.product(range(3), repeat=4) if collapse(p) == target]
            masses = np.array([np.exp(sum(lp[t, c] for t, c in enumerate(p))) for p in paths])
            scores = np.array([sum(tangent[t, c] for t, c in enumerate(p)) for p in paths])
            probabilities = masses / masses.sum()
            expected, derivative = np.zeros_like(lp), np.zeros_like(lp)
            mean_score = probabilities @ scores
            for path, mass, score in zip(paths, probabilities, scores):
                for t, c in enumerate(path):
                    expected[t, c] += mass
                    derivative[t, c] += mass * (score - mean_score)
            z, gamma, dz, dgamma = ctc_statistics(lp, np.array(target, dtype=np.int64), tangent)
            self.assertAlmostEqual(z, np.log(masses.sum()), places=12)
            self.assertAlmostEqual(dz, mean_score, places=12)
            np.testing.assert_allclose(gamma, expected, atol=1e-12)
            np.testing.assert_allclose(dgamma, derivative, atol=1e-12)

    def test_invalid_and_impossible_transcripts_fail(self):
        lp = np.full((2, 3), -np.log(3))
        for target in ([1, 1], [0], [3], [1.5]):
            with self.assertRaises(ValueError):
                ctc_statistics(lp, np.array(target))
        with self.assertRaisesRegex(ValueError, "tangent"):
            ctc_statistics(lp, np.array([1]), np.zeros((1, 3)))

    def test_peaky_long_path_has_finite_second_derivatives(self):
        logits = np.tile([0., -400., -800.], (300, 1))
        lp = logits - logsumexp(logits, axis=1, keepdims=True)
        tangent = np.tile([0., .2, -.3], (300, 1))
        z, q, dz, dq = ctc_statistics(lp, np.array([1, 2, 1, 1]), tangent)
        self.assertTrue(np.isfinite(z) and np.isfinite(dz) and np.isfinite(dq).all())
        np.testing.assert_allclose(q.sum(1), 1., atol=1e-9)
        np.testing.assert_allclose(dq.sum(1), 0., atol=1e-9)

    def test_complete_objective_gradient_and_hvp_match_finite_differences(self):
        head, rows = fixture()
        vector = head.vector()
        direction = np.random.default_rng(9).normal(size=len(vector))
        value, gradient, hv = corpus_evaluate(head, head, rows, .01, direction)
        eps = 1e-5
        plus = corpus_evaluate(head.with_vector(vector + eps * direction), head, rows, .01)
        minus = corpus_evaluate(head.with_vector(vector - eps * direction), head, rows, .01)
        self.assertAlmostEqual(gradient @ direction, (plus[0] - minus[0]) / (2 * eps), places=8)
        np.testing.assert_allclose(hv, (plus[1] - minus[1]) / (2 * eps), atol=1e-9, rtol=1e-8)
        self.assertTrue(np.isfinite(value))

    def test_hvp_is_symmetric_and_retains_nonconvex_path_covariance(self):
        # Two possible placements of one phone make the uniform CTC head a
        # saddle along an antisymmetric acoustic direction, with zero gradient.
        head = phone_head(np.zeros((2, 1)), np.zeros(2))
        features = np.array([[-1.], [1.]])
        lp = log_probs(head, features)
        direction = np.array([0., 1., 0., 0.])
        dlogits = features @ head.with_vector(direction).weight.T
        tangent = dlogits - (np.exp(lp) * dlogits).sum(1, keepdims=True)
        _, gamma, _, dg = ctc_statistics(lp, np.array([1]), tangent)
        gradient = ((np.exp(lp) - gamma).T @ features).ravel()
        hv = ((np.exp(lp) * tangent - dg).T @ features).ravel()
        self.assertAlmostEqual(gradient[1], 0., places=12)
        self.assertLess(hv[1], 0.)
        # On the complete objective (including preservation), verify Hessian
        # symmetry independently across all weight and bias coordinates.
        head, rows = fixture()
        identity = np.eye(len(head.vector()))
        hessian = np.stack([corpus_evaluate(head, head, rows, .01, v)[2] for v in identity], axis=1)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)

    def test_equal_recording_and_view_weights_and_no_cache_mutation(self):
        head, rows = fixture()
        snapshots = copy.deepcopy(rows)
        duplicates = copy.deepcopy(rows)
        duplicates[0]["views"] *= 3
        duplicates *= 2
        direction = np.arange(len(head.vector()), dtype=float)
        a = corpus_evaluate(head, head, rows, .1, direction)
        b = corpus_evaluate(head, head, duplicates, .1, direction)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, atol=1e-12)
        for row, old in zip(rows, snapshots):
            for view, previous in zip(row["views"], old["views"]):
                for key in view:
                    np.testing.assert_array_equal(view[key], previous[key])

    def test_lanczos_finds_negative_direction_and_pads_breakdown_compute(self):
        for matrix in (np.diag([-3., .1, 5.]), np.eye(3)):
            calls = []
            def hvp(v):
                calls.append(v.copy())
                return matrix @ v
            direction, report = smallest_curvature(hvp, 3, 7, iterations=8)
            self.assertEqual(len(calls), 9)
            self.assertAlmostEqual(report["rayleigh"], np.linalg.eigvalsh(matrix).min(), places=10)
            self.assertLess(report["residualNorm"], 1e-10)
            self.assertEqual(report["negative"], bool(matrix[0, 0] < 0))
            self.assertAlmostEqual(np.linalg.norm(direction), 1.)

    def test_trust_escape_at_zero_gradient_only_accepts_actual_decrease(self):
        calls = []
        def evaluate(v):
            calls.append(v.copy())
            x = v[0]
            return float(-x * x + x ** 4), np.array([-2 * x + 4 * x ** 3])
        got, report = trust_candidates(np.zeros(1), np.ones(1), 1., evaluate)
        self.assertEqual(len(calls), 9)
        self.assertTrue(report["accepted"])
        self.assertLess(evaluate(got)[0], 0.)
        self.assertEqual(report["candidateNorms"], [1., .5, .25, .125])
        # A positive-curvature minimum rejects every nonzero step. First-order
        # at the saddle evaluates zero steps and cannot recover by chance.
        for fn, direction in ((lambda v: (float(v @ v), 2 * v), np.ones(1)),
                              (evaluate, np.zeros(1))):
            got, report = trust_candidates(np.zeros(1), direction, 1., fn)
            self.assertFalse(report["accepted"])
            np.testing.assert_array_equal(got, np.zeros(1))

    def test_deadline_and_budget_do_not_claim_stationarity(self):
        head, rows = fixture()
        with self.assertRaises(TimeoutError):
            corpus_evaluate(head, head, rows, .01, deadline=time.monotonic() - 1)
        oracle = CorpusOracle(head, head, rows, .01, time.monotonic() + 30)
        _, report = first_order_start(head, oracle, evaluations=1)
        self.assertEqual(report["evaluations"], 1)
        self.assertFalse(report["converged"])
        for seconds in (0., float("nan"), 86401.):
            with self.assertRaises(ValueError):
                matched_refit(head, rows, dict(fit="fit", selection="selection"),
                    lambda *args: dict(objective=0.), seconds=seconds)

    def test_complete_ctc_saddle_escapes_despite_stationary_first_order_control(self):
        from scipy.optimize import brentq
        features = np.array([[-1.], [-1.], [-1.], [1.], [1.], [1.]])
        target = np.array([1])
        def bias_gradient(bias):
            head = phone_head(np.zeros((2, 1)), np.array([0., bias]))
            lp = log_probs(head, features)
            return (np.exp(lp) - ctc_statistics(lp, target)[1])[:, 1].sum()
        bias = brentq(bias_gradient, -10., 0.)
        head = phone_head(np.zeros((2, 1)), np.array([0., bias]))
        lp = log_probs(head, features)
        view = dict(features=features, target=target, teacher=lp.copy(), shipped=lp.copy())
        rows = [dict(speaker=s, views=[copy.deepcopy(view)]) for s in ("fit", "selection")]
        def validate(h, rs, initial, reg):
            return dict(objective=corpus_evaluate(h, initial, rs, reg)[0])
        heads, report = matched_refit(head, rows, dict(fit="fit", selection="selection"),
            validate, rounds=1, lanczos_iterations=4, warm_evaluations=4)
        self.assertTrue(report["warmStart"]["converged"])
        phase = report["history"][-1]["diagnostics"]
        self.assertEqual(phase["curvature"]["direction"], "negative-curvature")
        self.assertTrue(phase["curvature"]["curvature"]["negative"])
        self.assertTrue(phase["curvature"]["step"]["accepted"])
        self.assertFalse(phase["first-order"]["step"]["accepted"])
        self.assertLess(validate(heads["curvature"], rows[:1], head, .01)["objective"],
                        validate(heads["first-order"], rows[:1], head, .01)["objective"])

    def test_controls_match_compute_norms_and_exclude_selection_from_fit(self):
        head, rows = fixture()
        seen = []
        original = corpus_evaluate
        def observed(h, initial, rs, *args, **kwargs):
            seen.append([r["speaker"] for r in rs])
            return original(h, initial, rs, *args, **kwargs)
        def validate(h, rs, initial, reg):
            self.assertEqual([r["speaker"] for r in rs], ["selection"])
            return dict(objective=original(h, initial, rs, reg)[0])
        with patch("student_negative_curvature.corpus_evaluate", observed):
            heads, report = matched_refit(head, rows, dict(fit="fit", selection="selection"),
                validate, rounds=2, lanczos_iterations=3, warm_evaluations=5)
        self.assertEqual(set(heads), {"curvature", "first-order", "random"})
        self.assertTrue(all(s == ["fit"] for s in seen))
        for phase in report["history"][2:]:
            diagnostics = list(phase["diagnostics"].values())
            self.assertEqual(len({d["evaluations"] for d in diagnostics}), 1)
            self.assertEqual(len({d["hvps"] for d in diagnostics}), 1)
            self.assertTrue(all(d["step"]["candidateNorms"] == diagnostics[0]["step"]["candidateNorms"]
                                for d in diagnostics))
            self.assertEqual(phase["diagnostics"]["first-order"]["direction"], "first-order")
            self.assertEqual(phase["diagnostics"]["random"]["direction"], "random")
        self.assertIn(report["selected"]["phase"], range(-1, 3))


class EnrollmentTests(unittest.TestCase):
    def test_runner_accepts_explicit_enrollment_inputs(self):
        from student_negative_curvature_train import argument_parser
        parsed = argument_parser().parse_args(["prepare", "--help-labels", "labels.jsonl",
            "--model", "bundle", "--transcribed", "performed.jsonl", "--exclusions", "excluded.json",
            "--augmentation", "banks", "--out", "cache"])
        self.assertEqual(parsed.help, "labels.jsonl")
        self.assertEqual(parsed.transcribed, "performed.jsonl")
        parsed = argument_parser().parse_args(["fit", "--prepared", "cache", "--out", "selected", "--seconds", "2700"])
        self.assertEqual(parsed.seconds, 2700.)

    def test_speakers_are_grouped_independently_of_row_order(self):
        speakers = [f"generated-{i}" for i in range(10)]
        self.assertEqual(speaker_partition(speakers), speaker_partition(speakers[::-1] * 3))
        self.assertEqual(list(speaker_partition(speakers).values()).count("selection"), 2)

    def test_only_training_none_takes_are_admitted_with_global_holdout_exclusion(self):
        row = dict(id="generated", split="train", kind="none", speaker="learner", scored=True,
            events=[], non_errors=[], unsupported=[], wav="generated.wav", ayahs=["generated:ayah"])
        self.assertEqual(admitted_none([row]), [row])
        for change in (dict(kind="repeat"), dict(events=[{}]), dict(split="release"),
                       dict(non_errors=[{}]), dict(scored=False)):
            self.assertEqual(admitted_none([dict(row, **change)]), [])
        self.assertEqual(admitted_none([row, dict(row, id="other", split="release")]), [])
        self.assertEqual(admitted_none([row], ["learner"]), [])

    def test_cross_corpus_holdouts_exclude_none_take_before_enrollment(self):
        from student_head_data import recordings
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            none = dict(id="none", split="train", kind="none", speaker="held", scored=True,
                events=[], non_errors=[], unsupported=[], wav="a.wav", ayahs=["synthetic:ayah"])
            (root / "labels.jsonl").write_text(json.dumps(none))
            (root / "transcribed.jsonl").write_text(json.dumps(dict(speaker="held", split="release")))
            with self.assertRaisesRegex(ValueError, "Both authorized"):
                recordings(root / "labels.jsonl", root / "transcribed.jsonl", [],
                    {"ayahs": [dict(s="synthetic", a="ayah", p="a")]})

    def test_duration_notation_collapses_within_ayah_without_losing_boundary_repetition(self):
        from student_head_data import articulation_phones
        self.assertEqual(articulation_phones("aabbbcc"), "abc")
        self.assertEqual(articulation_phones("abbb") + articulation_phones("bccc"), "abbc")


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
        from student_head_export import head_parameters
        for quantized in (False, True):
            graph, w, b = self.graph(quantized)
            got_w, got_b = head_parameters(graph)
            np.testing.assert_allclose(got_w, w, atol=1e-7)
            np.testing.assert_array_equal(got_b, b)

    def test_replacement_preserves_encoder_and_states_and_removes_old_head(self):
        import onnxruntime as ort
        from student_head_export import replace_head
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
        from student_head_export import insertion_assets
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "lacks"):
                insertion_assets(root)
            graph, _, _ = self.graph()
            onnx.save(graph, str(Path(root) / "teacher.onnx"))
            with self.assertRaisesRegex(ValueError, "data file"):
                insertion_assets(root)

    def test_full_bundle_copies_teacher_and_preserves_manifest_contract(self):
        import onnx
        from student_head_export import digest, graft, insertion_assets
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
            graft(source, phone_head(w + .1, b), target, "negative-curvature")
            exported = json.loads((target / "manifest.json").read_text())
            self.assertEqual(exported["sha256"], digest(target / "model.onnx"))
            for key in ("tokens", "blank", "frameMs", "sampleRate", "shift", "window", "fbank", "states"):
                self.assertEqual(exported[key], manifest[key])
            self.assertEqual(insertion_assets(target), before)
            self.assertEqual(digest(source / "model.onnx"), manifest["sha256"])


if __name__ == "__main__":
    unittest.main()

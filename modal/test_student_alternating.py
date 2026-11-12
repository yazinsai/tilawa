"""Tiny generated solver/contract tests; no corpus or practice access."""
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

from student_alternating import (admitted_none, assignments, corpus_value_gradient,
    forward_backward, log_probs, matched_refit, minimize_head, phone_head, speaker_partition)


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


class PathTests(unittest.TestCase):
    def test_posterior_matches_all_paths_including_repeats_and_blank(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(4, 3))
        lp = logits - logsumexp(logits, axis=1, keepdims=True)
        for y in ([], [1], [1, 1], [1, 2], [2, 1, 2]):
            paths = [p for p in itertools.product(range(3), repeat=4) if collapse(p) == y]
            mass = np.array([np.exp(sum(lp[t, x] for t, x in enumerate(p))) for p in paths])
            expected = np.zeros_like(lp)
            for p, m in zip(paths, mass / mass.sum()):
                for t, x in enumerate(p):
                    expected[t, x] += m
            z, got = forward_backward(lp, np.array(y, dtype=np.int64))
            self.assertAlmostEqual(z, np.log(mass.sum()), places=12)
            np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_impossible_and_invalid_targets_fail(self):
        lp = np.full((2, 3), -np.log(3))
        for y in ([1, 1], [0], [3]):
            with self.assertRaises(ValueError):
                forward_backward(lp, np.array(y))

    def test_long_peaky_lattice_stays_normalized(self):
        logits = np.tile([0., -400., -800.], (300, 1))
        lp = logits - logsumexp(logits, axis=1, keepdims=True)
        z, q = forward_backward(lp, np.array([1, 2, 1, 1]))
        self.assertTrue(np.isfinite(z))
        np.testing.assert_allclose(q.sum(1), 1., atol=1e-9)


class FitTests(unittest.TestCase):
    def test_ctc_and_surrogate_gradients_match_finite_differences(self):
        head, rows = fixture()
        for frozen in (None, assignments(head, rows)):
            value, grad = corpus_value_gradient(head, head, rows, .01, frozen)
            vector = head.vector()
            for i in range(len(vector)):
                delta = np.zeros_like(vector)
                delta[i] = 1e-5
                a = corpus_value_gradient(head.with_vector(vector + delta), head, rows, .01, frozen)[0]
                b = corpus_value_gradient(head.with_vector(vector - delta), head, rows, .01, frozen)[0]
                self.assertAlmostEqual(grad[i], (a - b) / 2e-5, places=8)
        self.assertTrue(np.isfinite(value))

    def test_fixed_assignments_match_initial_ctc_gradient(self):
        head, rows = fixture()
        frozen = assignments(head, rows)
        exact = corpus_value_gradient(head, head, rows, .01)[1]
        surrogate = corpus_value_gradient(head, head, rows, .01, frozen)[1]
        np.testing.assert_allclose(exact, surrogate, atol=1e-12)
        with self.assertRaises(ValueError):
            frozen[0][0][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "whole"):
            corpus_value_gradient(head, head, rows, .01, frozen[:1])

    def test_converged_mstep_keeps_paths_fixed_and_decreases_observed_loss(self):
        head, rows = fixture()
        frozen = assignments(head, rows)
        snapshots = [[q.copy() for q in r] for r in frozen]
        before = corpus_value_gradient(head, head, rows, .1)[0]
        def evaluate(z):
            return corpus_value_gradient(head.with_vector(z), head, rows, .1, frozen)
        vector, report = minimize_head(head.vector(), evaluate, 500, time.monotonic() + 30)
        self.assertTrue(report["converged"], report)
        self.assertLess(corpus_value_gradient(head.with_vector(vector), head, rows, .1)[0], before)
        self.assertLess(report["objective"], evaluate(head.vector())[0])
        for old, new in zip(snapshots, frozen):
            for a, b in zip(old, new):
                np.testing.assert_array_equal(a, b)

    def test_budget_is_not_convergence_and_deadline_is_enforced(self):
        vector, report = minimize_head(np.array([2.]), lambda z: (float(z @ z), 2 * z),
                                       1, time.monotonic() + 30)
        self.assertFalse(report["converged"])
        self.assertEqual(report["evaluations"], 1)
        with self.assertRaises(TimeoutError):
            minimize_head(vector, lambda z: (0., z), 10, time.monotonic() - 1)

    def test_equal_recording_and_view_weights(self):
        head, rows = fixture()
        duplicate = copy.deepcopy(rows)
        duplicate[0]["views"] *= 3
        duplicate *= 2
        a = corpus_value_gradient(head, head, rows, .1)
        b = corpus_value_gradient(head, head, duplicate, .1)
        self.assertAlmostEqual(a[0], b[0], places=12)
        np.testing.assert_allclose(a[1], b[1], atol=1e-12)

    def test_whole_fit_phase_and_matched_control_exclude_selection_paths(self):
        head, rows = fixture()
        seen = []
        original = assignments
        def expectation(h, rs, deadline):
            seen.append([r["speaker"] for r in rs])
            return original(h, rs, deadline)
        def validate(h, rs, initial, reg):
            self.assertEqual([r["speaker"] for r in rs], ["selection"])
            return dict(objective=corpus_value_gradient(h, initial, rs, reg)[0])
        with patch("student_alternating.assignments", expectation):
            heads, report = matched_refit(head, rows, dict(fit="fit", selection="selection"),
                                         validate, (.1,), rounds=2)
        self.assertEqual(seen, [["fit"], ["fit"]])
        self.assertEqual(set(heads), {"ordinary", "alternating"})
        for phase in report["history"][1:]:
            self.assertTrue(phase["diagnostics"]["alternating"]["converged"])
            self.assertLess(phase["unusedRoundingWork"], report["workUnits"]["ordinary"])
            self.assertLessEqual(phase["ordinaryEvaluations"], phase["ordinaryAllowance"])

    def test_unconverged_mstep_cannot_refresh_or_export(self):
        head, rows = fixture()
        with patch("student_alternating.minimize_head", return_value=(head.vector(),
            dict(converged=False, termination="test budget"))):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                matched_refit(head, rows, dict(fit="fit", selection="selection"),
                    lambda *a: dict(objective=1.), (.1,), rounds=2)


class EnrollmentTests(unittest.TestCase):
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
        from student_alternating_train import recordings
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
        from student_alternating_train import articulation_phones
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
            graft(source, phone_head(w + .1, b), target, "alternating")
            exported = json.loads((target / "manifest.json").read_text())
            self.assertEqual(exported["sha256"], digest(target / "model.onnx"))
            for key in ("tokens", "blank", "frameMs", "sampleRate", "shift", "window", "fbank", "states"):
                self.assertEqual(exported[key], manifest[key])
            self.assertEqual(insertion_assets(target), before)
            self.assertEqual(digest(source / "model.onnx"), manifest["sha256"])


if __name__ == "__main__":
    unittest.main()

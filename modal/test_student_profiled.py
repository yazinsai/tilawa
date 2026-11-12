"""Synthetic acoustic-objective, variable-projection and bundle contract tests."""
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

from student_profiled import (admitted_none, corpus_value_gradient, ctc_statistics,
    initialize, log_probs, matched_fit, minimize, profile, speaker_partition)


def collapse(path):
    return [x for i, x in enumerate(path) if x and (i == 0 or x != path[i - 1])]


def fixture():
    model = initialize(np.array([[.1, -.1], [.4, .2], [-.2, .3]]), np.array([.1, 0., -.1]), hidden=2)
    x = np.array([[1., 0.], [.3, .5], [-.2, 1.], [.4, -.3]])
    lp = log_probs(model, x)
    view = dict(features=x, target=np.array([1, 2]), teacher=lp.copy(), shipped=lp.copy())
    rows = [dict(speaker=f"generated-{i}", views=[copy.deepcopy(view), copy.deepcopy(view)]) for i in range(5)]
    return model, rows


class PathTests(unittest.TestCase):
    def test_exact_likelihood_and_occupancy_against_exhaustive_paths(self):
        logits = np.random.default_rng(7).normal(size=(4, 3))
        lp = logits - logsumexp(logits, axis=1, keepdims=True)
        for y in ([], [1], [1, 1], [1, 2], [2, 1, 2]):
            paths = [p for p in itertools.product(range(3), repeat=4) if collapse(p) == y]
            mass = np.array([np.exp(sum(lp[t, x] for t, x in enumerate(p))) for p in paths])
            expected = np.zeros_like(lp)
            for p, m in zip(paths, mass / mass.sum()):
                for t, x in enumerate(p):
                    expected[t, x] += m
            z, got = ctc_statistics(lp, np.array(y, dtype=np.int64))
            self.assertAlmostEqual(z, np.log(mass.sum()), places=12)
            np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_impossible_targets_are_rejected_and_long_lattices_stay_finite(self):
        lp = np.full((2, 3), -np.log(3))
        for y in ([1, 1], [0], [3]):
            with self.assertRaises(ValueError):
                ctc_statistics(lp, np.array(y))
        logits = np.tile([0., -400., -800.], (300, 1))
        z, q = ctc_statistics(logits - logsumexp(logits, axis=1, keepdims=True), np.array([1, 2, 1, 1]))
        self.assertTrue(np.isfinite(z))
        np.testing.assert_allclose(q.sum(1), 1., atol=1e-9)


class FitTests(unittest.TestCase):
    def test_zero_residual_starts_at_native_affine_head(self):
        model, rows = fixture()
        x = rows[0]["views"][0]["features"]
        scores = x @ model.weight.T + model.bias
        np.testing.assert_allclose(log_probs(model, x), scores - logsumexp(scores, axis=1, keepdims=True), atol=1e-12)

    def test_all_parameter_gradients_against_finite_differences(self):
        initial, rows = fixture()
        vector = initial.vector() + np.random.default_rng(9).normal(0., .1, len(initial.vector()))
        model = initial.with_vector(vector)
        _, gradient = corpus_value_gradient(model, initial, rows[:1], .1)
        for i, want in enumerate(gradient.vector()):
            delta = np.zeros_like(vector)
            delta[i] = 1e-5
            a = corpus_value_gradient(model.with_vector(vector + delta), initial, rows[:1], .1)[0]
            b = corpus_value_gradient(model.with_vector(vector - delta), initial, rows[:1], .1)[0]
            self.assertAlmostEqual(want, (a - b) / 2e-5, places=8)

    def test_profiled_gradient_matches_reoptimized_finite_differences(self):
        initial, rows = fixture()
        model = copy.deepcopy(initial)
        model.up[:] = [[.1, -.2], [.2, .05]]
        before = model.vector("adapter").copy()
        # 100x tighter than the training stationarity floor, while allowing
        # float64 objective roundoff near the reoptimized head minimum.
        fitted, value, gradient, report = profile(model, initial, rows[:1], .1, 512, time.monotonic() + 30, 1e-7)
        self.assertTrue(report["converged"])
        self.assertLess(value, corpus_value_gradient(model, initial, rows[:1], .1)[0])
        np.testing.assert_array_equal(fitted.vector("adapter"), before)
        for i, want in enumerate(gradient):
            delta = np.zeros_like(before)
            delta[i] = 1e-4
            a = profile(fitted.with_vector(before + delta, "adapter"), initial, rows[:1], .1, 512, time.monotonic() + 30, 1e-7)[1]
            b = profile(fitted.with_vector(before - delta, "adapter"), initial, rows[:1], .1, 512, time.monotonic() + 30, 1e-7)[1]
            self.assertAlmostEqual(want, (a - b) / 2e-4, places=6)

    def test_nonstationary_head_is_not_used_as_profiled_gradient(self):
        model, rows = fixture()
        with self.assertRaisesRegex(RuntimeError, "stationarity"):
            profile(model, model, rows[:1], .1, 1, time.monotonic() + 30)

    def test_budget_and_deadline_fail_without_claiming_convergence(self):
        _, report = minimize(np.array([2.]), lambda z: (float(z @ z), 2 * z), 1, time.monotonic() + 30)
        self.assertFalse(report["converged"])
        self.assertEqual(report["evaluations"], 1)
        with self.assertRaises(TimeoutError):
            minimize(np.array([2.]), lambda z: (float(z @ z), 2 * z), 4, time.monotonic() - 1)
        model, rows = fixture()
        with self.assertRaises(TimeoutError):
            corpus_value_gradient(model, model, rows, .1, time.monotonic() - 1)

    def test_equal_recording_and_view_weights_and_immutable_cache(self):
        model, rows = fixture()
        before = copy.deepcopy(rows)
        duplicate = copy.deepcopy(rows)
        duplicate[0]["views"] *= 3
        duplicate *= 2
        a = corpus_value_gradient(model, model, rows, .1)
        b = corpus_value_gradient(model, model, duplicate, .1)
        self.assertAlmostEqual(a[0], b[0], places=12)
        np.testing.assert_allclose(a[1].vector(), b[1].vector(), atol=1e-12)
        for r, snapshot in zip(rows, before):
            for v, saved in zip(r["views"], snapshot["views"]):
                for k in v:
                    np.testing.assert_array_equal(v[k], saved[k])

    def test_pair_has_same_capacity_compute_allowance_and_training_only_selection(self):
        initial, rows = fixture()
        groups = speaker_partition([r["speaker"] for r in rows])
        fit_speakers = {s for s, g in groups.items() if g == "fit"}
        selection_speakers = set(groups) - fit_speakers
        calls = []
        original = corpus_value_gradient
        def checked(model, init, rs, reg, deadline=float("inf")):
            calls.append({r["speaker"] for r in rs})
            return original(model, init, rs, reg, deadline)
        def validate(model, rs, init, reg):
            self.assertEqual({r["speaker"] for r in rs}, selection_speakers)
            return dict(objective=original(model, init, rs, reg)[0])
        with patch("student_profiled.corpus_value_gradient", checked):
            models, report = matched_fit(initial, rows, groups, validate, (.1,), rounds=2, outer_evaluations=3)
        self.assertTrue(all(s == fit_speakers for s in calls))
        self.assertEqual(set(models), {"profiled", "joint"})
        self.assertEqual(len(models["profiled"].vector()), len(models["joint"].vector()))
        self.assertGreater(np.linalg.norm(models["profiled"].up), 0.)
        for phase in report["history"][1:]:
            count = phase["corpusEvaluations"]
            self.assertLessEqual(count["joint"], count["profiled"])
            self.assertTrue(all(r["converged"] for r in phase["diagnostics"]["profiled"]["headFits"]))


class EnrollmentTests(unittest.TestCase):
    def test_speaker_partition_is_stable_and_grouped(self):
        speakers = [f"generated-{i}" for i in range(10)]
        self.assertEqual(speaker_partition(speakers), speaker_partition(speakers[::-1] * 3))
        self.assertEqual(list(speaker_partition(speakers).values()).count("selection"), 2)

    def test_none_only_with_global_holdout_exclusion(self):
        row = dict(id="generated", split="train", kind="none", speaker="learner", scored=True,
            events=[], non_errors=[], unsupported=[], wav="generated.wav", ayahs=["generated:ayah"])
        self.assertEqual(admitted_none([row]), [row])
        for change in (dict(kind="repeat"), dict(events=[{}]), dict(split="release"), dict(non_errors=[{}]), dict(scored=False)):
            self.assertEqual(admitted_none([dict(row, **change)]), [])
        self.assertEqual(admitted_none([row, dict(row, id="other", split="release")]), [])
        self.assertEqual(admitted_none([row], ["learner"]), [])

    def test_cross_corpus_holdout_excludes_none_take(self):
        from student_profiled_train import recordings
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            none = dict(id="none", split="train", kind="none", speaker="held", scored=True,
                events=[], non_errors=[], unsupported=[], wav="a.wav", ayahs=["synthetic:ayah"])
            (root / "labels.jsonl").write_text(json.dumps(none))
            (root / "transcribed.jsonl").write_text(json.dumps(dict(speaker="held", split="release")))
            with self.assertRaisesRegex(ValueError, "Both authorized"):
                recordings(root / "labels.jsonl", root / "transcribed.jsonl", [], {"ayahs": [dict(s="synthetic", a="ayah", p="a")]})

    def test_duration_notation_retains_boundary_repetitions(self):
        from student_profiled_train import articulation_phones
        self.assertEqual(articulation_phones("aabbbcc"), "abc")
        self.assertEqual(articulation_phones("abbb") + articulation_phones("bccc"), "abbc")

    def test_enrolled_audio_identity_and_corpus_containment_are_required(self):
        from student_profiled_train import recordings, contained
        from student_profiled_export import digest
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "none.wav").write_bytes(b"generated none recording")
            (root / "transcribed.wav").write_bytes(b"generated authorized recording")
            none = dict(id="none", split="train", kind="none", speaker="one", scored=True,
                events=[], non_errors=[], unsupported=[], wav="none.wav", ayahs=["synthetic:ayah"])
            enrolled = dict(speaker="two", split="train", authorized=True, verified=True,
                            phones="بت", wav="transcribed.wav", sha256=digest(root / "transcribed.wav"))
            (root / "labels.jsonl").write_text(json.dumps(none))
            (root / "transcribed.jsonl").write_text(json.dumps(enrolled))
            reference = {"ayahs": [dict(s="synthetic", a="ayah", p="ببتت")]}
            rows, excluded = recordings(root / "labels.jsonl", root / "transcribed.jsonl", [], reference)
            self.assertEqual({r["source"] for r in rows}, {"help-none", "transcribed"})
            self.assertEqual(excluded, [])
            (root / "transcribed.wav").write_bytes(b"changed recording")
            with self.assertRaisesRegex(ValueError, "identity"):
                recordings(root / "labels.jsonl", root / "transcribed.jsonl", [], reference)
            with self.assertRaisesRegex(ValueError, "leaves"):
                contained(root, "../escape.wav")

    def test_prepared_cache_identity_and_partition_are_checked(self):
        from student_profiled_train import load_prepared, TOKENS
        from student_profiled_export import digest
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source"
            source.mkdir()
            (source / "manifest.json").write_text("{}")
            model, rows = fixture()
            view = rows[0]["views"][0]
            files = [f"view-{i}.npz" for i in range(3)]
            for file in files:
                np.savez(root / file, **view)
            metadata = dict(rows=[dict(speaker=r["speaker"], views=files, viewHashes=[digest(root / f) for f in files]) for r in rows],
                groups=speaker_partition([r["speaker"] for r in rows]), excludedSpeakers=[], source=str(source),
                sourceSha256="generated", width=2, tokens=TOKENS, auxiliaryHashes={}, sourceManifestSha256=digest(source / "manifest.json"))
            (root / "prepared.json").write_text(json.dumps(metadata))
            with patch("student_profiled_train.load_source", return_value=({"sha256": "generated"}, None, "feature", 2)), \
                    patch("student_profiled_train.insertion_assets", return_value={}):
                _, loaded = load_prepared(root)
                self.assertEqual(len(loaded), 5)
                (root / files[0]).write_bytes(b"changed cache")
                with self.assertRaisesRegex(ValueError, "Cached training evidence changed"):
                    load_prepared(root)


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
        from student_profiled_export import head_parameters
        for quantized in (False, True):
            graph, w, b = self.graph(quantized)
            got_w, got_b = head_parameters(graph)
            np.testing.assert_allclose(got_w, w, atol=1e-7)
            np.testing.assert_array_equal(got_b, b)

    def test_replacement_preserves_encoder_and_states_and_removes_old_head(self):
        import onnxruntime as ort
        from student_profiled_export import replace_head
        for quantized in (False, True):
            graph, w, b = self.graph(quantized)
            original = copy.deepcopy(graph)
            model = initialize(w + .1, b + .2, hidden=2)
            model.up[:] = [[.1, -.2], [.2, .05]]
            replaced = replace_head(graph, model)
            names = [n.name for n in replaced.graph.node]
            self.assertIn("encoder", names)
            self.assertIn("state_update", names)
            self.assertFalse(any("ctc_output" in name for name in names))
            x = np.array([[[1., 2.], [.2, -.1]]], dtype=np.float32)
            feeds = dict(x=x, state=np.array([5.], np.float32))
            got = ort.InferenceSession(replaced.SerializeToString()).run(None, feeds)
            old = ort.InferenceSession(original.SerializeToString()).run(None, feeds)
            expected = log_probs(model, x)
            np.testing.assert_allclose(got[0], expected, atol=1e-6)
            np.testing.assert_array_equal(got[1], old[1])

    def test_incomplete_teacher_bundle_rejected(self):
        import onnx
        from student_profiled_export import insertion_assets
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "lacks"):
                insertion_assets(root)
            graph, _, _ = self.graph()
            onnx.save(graph, str(Path(root) / "teacher.onnx"))
            with self.assertRaisesRegex(ValueError, "data file"):
                insertion_assets(root)

    def test_full_bundle_copies_teacher_and_preserves_manifest_contract(self):
        import onnx
        from student_profiled_export import digest, graft, insertion_assets
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
            graft(source, initialize(w + .1, b, hidden=2), target, "profiled")
            exported = json.loads((target / "manifest.json").read_text())
            self.assertEqual(exported["sha256"], digest(target / "model.onnx"))
            for key in ("tokens", "blank", "frameMs", "sampleRate", "shift", "window", "fbank", "states"):
                self.assertEqual(exported[key], manifest[key])
            self.assertEqual(insertion_assets(target), before)
            self.assertEqual(digest(source / "model.onnx"), manifest["sha256"])


if __name__ == "__main__":
    unittest.main()

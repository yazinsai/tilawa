"""Synthetic paired-path, data-admission and export contracts; no corpus access."""
import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from student_paired import (admitted_none, correspondence, corpus_value_gradient,
    forward_backward, log_probs, matched_refit, paired_paths, phone_head, speaker_partition)


def collapse(path):
    return [x for i, x in enumerate(path) if x and (not i or x != path[i - 1])]


def fixture(delay=0):
    head = phone_head(np.array([[.1, -.1], [.4, .2], [-.2, .3]]), np.array([.1, 0., -.1]))
    x = np.array([[1., 0.], [.3, .5], [-.2, 1.], [.4, -.3]])
    views = [dict(features=x.copy(), shipped=log_probs(head, x), times=np.arange(4) * 640),
             dict(features=x[::-1].copy(), shipped=log_probs(head, x[::-1]), times=np.arange(4) * 640 - delay)]
    rows = [dict(speaker="fit", views=views, target=np.array([1, 2])),
            dict(speaker="selection", views=copy.deepcopy(views), target=np.array([1, 2]))]
    return head, rows


class PathTests(unittest.TestCase):
    def test_single_path_sum_and_occupancies_against_exhaustive_paths(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(4, 3))
        lp = logits - logsumexp(logits, axis=1, keepdims=True)
        for y in ([], [1], [1, 1], [1, 2], [2, 1, 2]):
            paths = [p for p in itertools.product(range(3), repeat=4) if collapse(p) == y]
            for e in (lp, 2 * lp):
                mass = np.array([np.exp(sum(e[t, x] for t, x in enumerate(p))) for p in paths])
                expected = np.zeros_like(lp)
                for p, m in zip(paths, mass / mass.sum()):
                    for t, x in enumerate(p):
                        expected[t, x] += m
                z, got = forward_backward(e, np.array(y, dtype=np.int64))
                self.assertAlmostEqual(z, np.log(mass.sum()), places=12)
                np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_joint_sums_same_paths_not_independent_paths_or_normalized_products(self):
        rng = np.random.default_rng(3)
        lps = []
        for _ in range(2):
            logits = rng.normal(size=(4, 3))
            lps.append(logits - logsumexp(logits, axis=1, keepdims=True))
        target = np.array([1, 2])
        result = paired_paths(lps, target, [np.arange(4) * 640] * 2)
        paths = [p for p in itertools.product(range(3), repeat=4) if collapse(p) == target.tolist()]
        weights = [[np.exp(sum(lp[t, x] for t, x in enumerate(p))) for p in paths] for lp in lps]
        self.assertAlmostEqual(result["shared"][0], np.log(np.dot(*weights)), places=12)
        self.assertAlmostEqual(result["independent"][0], np.log(np.sum(weights[0]) * np.sum(weights[1])), places=12)
        self.assertLess(result["shared"][0], result["independent"][0])
        product = lps[0] + lps[1]
        wrong_z = forward_backward(product - logsumexp(product, axis=1, keepdims=True), target)[0]
        self.assertGreater(abs(wrong_z - result["shared"][0]), .1)
        np.testing.assert_array_equal(result["shared"][1][0], result["shared"][1][1])

    def test_clear_view_disambiguates_weak_consonant_assignment(self):
        clear = np.log([[.02, .96, .02], [.97, .02, .01], [.97, .02, .01]])
        weak = np.log([[.70, .29, .01], [.40, .59, .01], [.97, .02, .01]])
        result = paired_paths([clear, weak], np.array([1]), [np.arange(3) * 640] * 2)
        shared, independent = result["shared"][1][1], result["independent"][1][1]
        self.assertGreater(shared[0, 1], independent[0, 1])
        self.assertLess(shared[1, 1], independent[1, 1])

    def test_exact_delay_and_partial_overlap_without_duplicate_emissions(self):
        head, rows = fixture(320)
        views = rows[0]["views"]
        grid, maps = correspondence([v["times"] for v in views])
        np.testing.assert_array_equal(grid, [-320, 0, 320, 640, 960, 1280, 1600, 1920])
        for v, indices in zip(views, maps):
            np.testing.assert_array_equal(grid[indices], v["times"])
        lps = [log_probs(head, v["features"]) for v in views]
        result = paired_paths(lps, rows[0]["target"], [v["times"] for v in views])
        # Oracle enumerates the union path and scores only each view's actual cells.
        paths = [p for p in itertools.product(range(3), repeat=len(grid)) if collapse(p) == [1, 2]]
        scores = [sum(lp[t, p[k]] for lp, indices in zip(lps, maps) for t, k in enumerate(indices)) for p in paths]
        self.assertAlmostEqual(result["shared"][0], logsumexp(scores), places=12)
        for gamma in result["shared"][1]:
            np.testing.assert_allclose(gamma.sum(1), 1., atol=1e-12)

    def test_integer_delay_pairs_coincident_rows_on_original_clock(self):
        grid, maps = correspondence([np.arange(4) * 640, np.arange(4) * 640 - 640])
        np.testing.assert_array_equal(grid, [-640, 0, 640, 1280, 1920])
        np.testing.assert_array_equal(maps[0][:-1], maps[1][1:])

    def test_impossible_repeats_invalid_targets_and_bad_correspondence_fail(self):
        lp = np.full((2, 3), -np.log(3))
        for y in ([1, 1], [0], [3]):
            with self.assertRaises(ValueError):
                forward_backward(lp, np.array(y))
        for times in ([np.array([0, 1]), np.array([2, 3])], [np.array([0, 0])] * 2,
                      [np.array([0., 1.])] * 2, [np.array([], dtype=int)] * 2):
            with self.assertRaises(ValueError):
                correspondence(times)
        with self.assertRaises(ValueError):
            paired_paths([lp, lp * 2], np.array([1]), [np.arange(2)] * 2)

    def test_long_peaky_joint_lattice_remains_finite(self):
        logits = np.tile([0., -400., -800.], (300, 1))
        lp = logits - logsumexp(logits, axis=1, keepdims=True)
        z, q = forward_backward(2 * lp, np.array([1, 2, 1, 1]))
        self.assertTrue(np.isfinite(z))
        np.testing.assert_allclose(q.sum(1), 1., atol=1e-9)


class FitTests(unittest.TestCase):
    def test_gradients_match_finite_differences_in_both_arms_with_delays(self):
        for delay in (0, 320, 640):
            head, rows = fixture(delay)
            initial = head.with_vector(head.vector() + .1)
            for arm in ("shared", "independent"):
                value, gradient = corpus_value_gradient(head, initial, rows, .1, arm)
                for i in range(len(head.vector())):
                    delta = np.zeros_like(head.vector())
                    delta[i] = 1e-5
                    a = corpus_value_gradient(head.with_vector(head.vector() + delta), initial, rows, .1, arm)[0]
                    b = corpus_value_gradient(head.with_vector(head.vector() - delta), initial, rows, .1, arm)[0]
                    self.assertAlmostEqual(gradient[i], (a - b) / 2e-5, places=8)
                self.assertTrue(np.isfinite(value))

    def test_shipped_preservation_covers_pause_rows(self):
        head, rows = fixture()
        before = corpus_value_gradient(head, head, rows, .1, "shared")
        changed = copy.deepcopy(rows)
        # A real-pause posterior enters preservation even without a phone label.
        for r in changed:
            r["views"][1]["shipped"][-1] = np.log([.98, .01, .01])
        after = corpus_value_gradient(head, head, changed, .1, "shared")
        self.assertGreater(after[0], before[0])
        self.assertGreater(np.linalg.norm(after[1] - before[1]), .01)

    def test_recording_duplication_and_view_swap_do_not_change_weights(self):
        head, rows = fixture(320)
        duplicate = copy.deepcopy(rows) * 3
        for row in duplicate:
            row["views"] = row["views"][::-1]
        for arm in ("shared", "independent"):
            a = corpus_value_gradient(head, head, rows, .1, arm)
            b = corpus_value_gradient(head, head, duplicate, .1, arm)
            self.assertAlmostEqual(a[0], b[0], places=12)
            np.testing.assert_allclose(a[1], b[1], atol=1e-12)

    def test_matched_updates_and_lattice_work_exclude_selection_from_fit(self):
        head, rows = fixture()
        seen, calls = [], []
        original = corpus_value_gradient
        def objective(h, initial, rs, regularization, arm, deadline):
            seen.append(([r["speaker"] for r in rs], arm))
            return original(h, initial, rs, regularization, arm, deadline)
        def validate(h, rs):
            self.assertEqual([r["speaker"] for r in rs], ["selection"])
            return dict(per=corpus_value_gradient(h, head, rs, .1, "independent")[0])
        with patch("student_paired.corpus_value_gradient", objective), patch("student_paired.forward_backward", wraps=forward_backward) as fb:
            heads, report = matched_refit(head, rows, dict(fit="fit", selection="selection"), validate,
                (.1,), updates=20, emit=calls.append)
            # Each update has 3 FB passes per arm; selection also evaluates both.
            self.assertEqual(fb.call_count, 20 * 2 * 3 + 3 * 2 * 3)
        self.assertEqual(seen, [(["fit"], arm) for _ in range(20) for arm in ("shared", "independent")])
        self.assertEqual(report["updatesPerArm"], 20)
        self.assertEqual(set(heads), {"shared", "independent"})
        self.assertGreater(report["selected"]["step"], 0)
        for arm in heads:
            self.assertLess(original(heads[arm], head, rows[:1], .1, arm)[0], original(head, head, rows[:1], .1, arm)[0])

    def test_invalid_partition_and_expired_budget_fail(self):
        head, rows = fixture()
        with self.assertRaises(ValueError):
            matched_refit(head, rows, dict(fit="fit", selection="fit"), lambda *_: dict(per=0.))
        with patch("student_paired.time.monotonic", side_effect=[0., 2.]):
            with self.assertRaises(TimeoutError):
                matched_refit(head, rows, dict(fit="fit", selection="selection"), lambda *_: dict(per=0.), seconds=1.)
        with self.assertRaises(TimeoutError):
            corpus_value_gradient(head, head, rows, .1, "shared", deadline=-1.)


class EnrollmentTests(unittest.TestCase):
    def test_whole_speakers_are_partitioned_independent_of_row_order(self):
        speakers = [f"synthetic-{i}" for i in range(10)]
        self.assertEqual(speaker_partition(speakers), speaker_partition(speakers[::-1] * 3))
        self.assertEqual(list(speaker_partition(speakers).values()).count("selection"), 2)

    def test_only_none_training_takes_without_other_labels_are_admitted(self):
        row = dict(id="synthetic", split="train", kind="none", speaker="learner", scored=True,
            events=[], non_errors=[], unsupported=[], wav="synthetic.wav", ayahs=["synthetic:ayah"])
        self.assertEqual(admitted_none([row]), [row])
        for change in (dict(kind="repeat"), dict(events=[{}]), dict(split="release"),
                       dict(non_errors=[{}]), dict(scored=False), dict(excluded=True)):
            self.assertEqual(admitted_none([dict(row, **change)]), [])
        self.assertEqual(admitted_none([row, dict(row, id="other", split="release")]), [])
        self.assertEqual(admitted_none([row], ["learner"]), [])

    def test_cross_corpus_holdout_discovery_precedes_admission(self):
        from student_paired_train import recordings
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            none = dict(id="none", split="train", kind="none", speaker="held", scored=True,
                events=[], non_errors=[], unsupported=[], wav="missing.wav", ayahs=["synthetic:ayah"])
            (root / "labels.jsonl").write_text(json.dumps(none))
            (root / "additional.jsonl").write_text(json.dumps(dict(speaker="held", split="release")))
            rows, excluded = recordings(root / "labels.jsonl", {"ayahs": [dict(s="synthetic", a="ayah", p="ب")]}, root / "additional.jsonl")
            self.assertEqual(rows, [])
            self.assertEqual(excluded, ["held"])

    def test_none_transcripts_preserve_repeated_articulation_at_ayah_boundaries(self):
        from student_paired_train import recordings
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "a.wav").write_bytes(b"synthetic audio identity")
            row = dict(id="none", split="train", kind="none", speaker="fit", scored=True,
                events=[], non_errors=[], unsupported=[], wav="a.wav", ayahs=["x:a", "x:b"])
            (root / "labels.jsonl").write_text(json.dumps(row))
            rows, _ = recordings(root / "labels.jsonl", {"ayahs": [dict(s="x", a="a", p="ببتت"), dict(s="x", a="b", p="تتثث")]})
            self.assertEqual(rows[0]["phones"], "بتتث")
            row["wav"] = "../outside.wav"
            (root / "labels.jsonl").write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                recordings(root / "labels.jsonl", {"ayahs": [dict(s="x", a="a", p="ببتت"), dict(s="x", a="b", p="تتثث")]})

    def test_phone_augmentation_keeps_samples_and_declares_direct_path_delay(self):
        from student_paired_train import phone_view, sample_times
        wave = np.sin(np.arange(1600) * .1).astype("float32")
        banks = dict(rir=dict(data=np.array([0., 0., 1.]), offsets=np.array([0, 3])),
                     noise=dict(data=np.sin(np.arange(100) * .5), offsets=np.array([0, 100])))
        phone, delay = phone_view(wave, banks, 1729)
        self.assertEqual(len(phone), len(wave))
        self.assertEqual(delay, 0)
        np.testing.assert_array_equal(phone, phone_view(wave, banks, 1729)[0])
        np.testing.assert_array_equal(sample_times(3, 16000, 40, 321), [-321, 319, 959])
        with self.assertRaises(ValueError):
            sample_times(3, 16000, 40, .5)


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
            graft(source, phone_head(w + .1, b), target, "shared")
            exported = json.loads((target / "manifest.json").read_text())
            self.assertEqual(exported["sha256"], digest(target / "model.onnx"))
            for key in ("tokens", "blank", "frameMs", "sampleRate", "shift", "window", "fbank", "states"):
                self.assertEqual(exported[key], manifest[key])
            self.assertEqual(insertion_assets(target), before)
            self.assertEqual(digest(source / "model.onnx"), manifest["sha256"])


if __name__ == "__main__":
    unittest.main()

"""Exact path enumeration, gradient, enrollment and native export contracts.

Run with Torch, ONNX and ONNX Runtime on the supplied supervisor:
python3 -m unittest discover -s modal -p test_student_margin.py -v
"""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from student_margin import (best_paths, path_margin, MAX_FRAMES, MAX_TARGET,
    reference_articulations, speaker_partition, admitted_none, phone_head,
    log_probs, objectives, matched_refit)


TORCH = importlib.util.find_spec("torch") is not None
ONNX = importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None


def collapsed(path):
    result, previous = [], None
    for label in path:
        if label != previous and label != 0:
            result.append(label)
        previous = label
    return result


class EnrollmentTests(unittest.TestCase):
    def test_held_notation_does_not_erase_genuine_gemination(self):
        self.assertEqual(reference_articulations("بباااۥۥںںںننننممم"), "بباۥںننمم")
        self.assertEqual(reference_articulations("ننممبب"), "ننممبب")

    def test_speaker_split_is_order_invariant_and_groups_all_takes(self):
        speakers = ["synthetic-" + str(i) for i in range(10)]
        self.assertEqual(speaker_partition(speakers), speaker_partition(list(reversed(speakers)) + speakers))
        self.assertEqual(list(speaker_partition(speakers).values()).count("selection"), 2)
        with self.assertRaises(ValueError):
            speaker_partition(["only"])

    def test_holdout_repeat_and_ambiguous_none_are_excluded(self):
        good = dict(id="synthetic", split="train", kind="none", speaker="fit", scored=True,
                    events=[], non_errors=[], unsupported=[], wav="x.wav", ayahs=["synthetic"])
        bad = [dict(good, id="repeat", non_errors=[{"kind": "repeat"}]),
               dict(good, id="script", kind="skipped_word"),
               dict(good, id="unscored", scored=False), dict(good, id="excluded", excluded=True)]
        self.assertEqual(admitted_none([good, *bad]), [good])
        self.assertEqual(admitted_none([good, dict(good, split="held-out")]), [])
        self.assertEqual(admitted_none([good], ["fit"]), [])
        with self.assertRaises(ValueError):
            admitted_none([good, good])


class PathTests(unittest.TestCase):
    def test_both_max_paths_match_exhaustive_enumeration(self):
        rng = np.random.default_rng(19)
        for blank in (0, 2):
            phones = [k for k in range(3) if k != blank]
            for n in range(1, 5):
                scores = rng.normal(size=(n, 3))
                paths = list(itertools.product(range(3), repeat=n))
                def collapse(path):
                    return [k for i, k in enumerate(path) if k != blank and (i == 0 or k != path[i - 1])]
                def score(path):
                    return sum(scores[t, k] for t, k in enumerate(path))
                for length in range(n + 1):
                    for target in itertools.product(phones, repeat=length):
                        correct = [p for p in paths if collapse(p) == list(target)]
                        if not correct:
                            with self.assertRaises(ValueError):
                                best_paths(scores, np.array(target, dtype=np.int64), blank)
                            continue
                        wrong = [p for p in paths if collapse(p) != list(target)]
                        got_good, got_bad = best_paths(scores, np.array(target, dtype=np.int64), blank)
                        self.assertEqual(collapse(got_good), list(target))
                        self.assertNotEqual(collapse(got_bad), list(target))
                        self.assertAlmostEqual(score(got_good), max(map(score, correct)), places=12)
                        self.assertAlmostEqual(score(got_bad), max(map(score, wrong)), places=12)
                        # Unit loss augmentation preserves the maximum wrong
                        # path, while including correct paths yields the hinge.
                        augmented = max(score(p) + (collapse(p) != list(target)) for p in paths)
                        self.assertAlmostEqual(max(score(got_good), score(got_bad) + 1), augmented, places=12)

    def test_incomplete_prefix_is_incorrect_even_without_a_mismatch(self):
        scores = np.array([[0., 8., -8.], [8., 0., -8.]])
        good, bad = best_paths(scores, np.array([1, 2]))
        self.assertEqual(collapsed(good), [1, 2])
        self.assertEqual(collapsed(bad), [1])

    def test_repetitions_need_blank_and_end_after_completion_can_be_wrong(self):
        scores = np.array([[0., 8., -8.], [0., 8., -8.], [0., 8., -8.]])
        good, bad = best_paths(scores, np.array([1, 1]))
        self.assertEqual(list(good), [1, 0, 1])
        self.assertEqual(collapsed(bad), [1])
        scores = np.array([[0., 8., -8.], [8., 0., -8.], [0., 8., -8.]])
        good, bad = best_paths(scores, np.array([1]))
        self.assertEqual(collapsed(good), [1])
        self.assertEqual(collapsed(bad), [1, 1])

    def test_silence_and_two_class_vocabulary(self):
        scores = np.array([[0., 8.], [8., 0.]])
        good, bad = best_paths(scores, np.array([], dtype=np.int64))
        self.assertEqual(list(good), [0, 0])
        self.assertEqual(collapsed(bad), [1])
        good, bad = best_paths(scores, np.array([1]))
        self.assertEqual(collapsed(good), [1])
        self.assertEqual(list(bad), [0, 0])

    def test_bounds_and_invalid_inputs_fail_without_approximating(self):
        target = np.array([1])
        for scores in (np.zeros((0, 3)), np.zeros((MAX_FRAMES + 1, 3)),
                       np.zeros((3, 1)), np.array([[np.nan, 1., 2.]])):
            with self.assertRaises(ValueError):
                best_paths(scores, target)
        for target in (np.array([0]), np.array([3]), np.array([1.5]),
                       np.ones(MAX_TARGET + 1, dtype=np.int64)):
            with self.assertRaises(ValueError):
                best_paths(np.zeros((5, 3)), target)


@unittest.skipUnless(TORCH, "Torch required on supervisor")
class GradientTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.t = torch

    def test_subgradient_matches_enumerated_hinge_finite_difference(self):
        t = self.t
        # Avoid tied maximizing paths: a central finite difference averages
        # subgradients at a tie, while Viterbi intentionally selects one path.
        logits = t.tensor([[.43, -.7, .3], [.8, -.3, -.9], [-.5, -.2, .1]],
                          dtype=t.float64, requires_grad=True)
        target = t.tensor([1, 2])
        loss = path_margin(logits.log_softmax(-1), target)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.)
        def enumerated(x):
            good, bad = [], []
            for path in itertools.product(range(3), repeat=len(x)):
                score = sum(x[i, k] for i, k in enumerate(path))
                (good if collapsed(path) == target.tolist() else bad).append(score)
            return max(0., 1. + max(bad) - max(good)) / len(target)
        x = logits.detach().numpy()
        for index in np.ndindex(x.shape):
            a, b = x.copy(), x.copy()
            a[index] += 1e-5
            b[index] -= 1e-5
            want = (enumerated(a) - enumerated(b)) / 2e-5
            self.assertAlmostEqual(float(logits.grad[index]), want, places=8)

    def test_correct_greedy_path_with_unit_margin_has_zero_loss(self):
        t = self.t
        logits = t.tensor([[0., 8., -8.], [8., 0., -8.], [0., -8., 8.]], requires_grad=True)
        loss = path_margin(logits.log_softmax(-1), t.tensor([1, 2]))
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0.)

    def test_high_aggregate_transcript_mass_does_not_defeat_greedy_blank(self):
        t = self.t
        p = t.tensor([[.6, .4]] * 2, dtype=t.float64)
        target = t.tensor([1])
        # .64 total mass collapses to [1], but greedy blank has .36;
        # the best correct frame path has only .24.
        mass = sum(np.prod([float(p[i, k]) for i, k in enumerate(path)])
                   for path in itertools.product(range(2), repeat=2) if collapsed(path) == [1])
        greedy_mass = float(p[:, 0].prod())
        self.assertAlmostEqual(mass, .64)
        self.assertGreater(mass, greedy_mass)
        logits = p.log().requires_grad_()
        loss = path_margin(logits.log_softmax(-1), target)
        self.assertAlmostEqual(float(loss.detach()), 1. + np.log(.36 / .24))
        loss.backward()
        self.assertLess(float(logits.grad[:, 1].sum()), 0.)
        self.assertGreater(float(logits.grad[:, 0].sum()), 0.)

    def test_padding_and_softmax_gauge_do_not_change_path_loss(self):
        t = self.t
        logits = t.tensor([[.4, -.7, .3], [.8, -.3, -.9], [-.5, -.2, .1], [100., -50., 80.]],
                          dtype=t.float64, requires_grad=True)
        target = t.tensor([1, 2])
        loss = path_margin(logits[:3], target)
        normalized = path_margin(logits[:3].log_softmax(-1), target)
        self.assertAlmostEqual(float(loss.detach()), float(normalized.detach()), places=12)
        loss.backward()
        self.assertEqual(float(logits.grad[3].abs().sum()), 0.)
        t.testing.assert_close(logits.grad.sum(-1), t.zeros(4, dtype=t.float64))

    def test_silence_retains_all_blank_ctc_and_penalizes_emissions(self):
        t = self.t
        lp = t.tensor([[.1, .9]] * 3).log()
        ctc, margin, preserve = objectives(lp, t.tensor([], dtype=t.int64), lp)
        self.assertAlmostEqual(float(ctc), -np.log(.1), places=6)
        self.assertGreater(float(margin), 0.)
        self.assertAlmostEqual(float(preserve), 0., places=6)


@unittest.skipUnless(TORCH, "Torch required on supervisor")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.t = torch
        self.head = phone_head(np.array([[.1, .2], [-.2, .3], [.3, -.1]], np.float32), np.zeros(3, np.float32))

    def view(self, n=8, offset=0., silence=False):
        t = self.t
        x = t.arange(n * 2, dtype=t.float32).reshape(n, 2) / 10 + offset
        frozen = log_probs(self.head, x).detach()
        return dict(features=x, shipped=frozen, target=t.tensor([] if silence else [1, 2], dtype=t.int64))

    def test_frozen_features_and_preservation_have_no_training_gradient(self):
        v = self.view()
        v["features"].requires_grad_()
        v["shipped"].requires_grad_()
        ctc, margin, preserve = objectives(log_probs(self.head, v["features"]), v["target"], v["shipped"])
        self.assertAlmostEqual(float(preserve.detach()), 0., places=7)
        (ctc + margin + preserve).backward()
        self.assertIsNone(v["features"].grad)
        self.assertIsNone(v["shipped"].grad)
        self.assertGreater(float(self.head.bias.grad.abs().sum()), 0.)

    def test_impossible_repeated_target_is_rejected(self):
        v = self.view(n=2)
        with self.assertRaises(ValueError):
            objectives(v["shipped"], self.t.tensor([1, 1]), v["shipped"])

    def test_ordinary_control_computes_paths_but_has_only_ctc_and_preservation_gradient(self):
        import student_margin as module
        t = self.t
        view = self.view()
        with patch.object(module, "best_paths", wraps=module.best_paths) as search:
            control, _ = module.recording_loss(self.head, [view], 0.)
            gradient = t.autograd.grad(control, tuple(self.head.parameters()))
            search.assert_called_once()
        lp = log_probs(self.head, view["features"])
        ctc, _, preserve = objectives(lp, view["target"], view["shipped"])
        expected = t.autograd.grad(ctc + preserve, tuple(self.head.parameters()))
        for a, b in zip(gradient, expected):
            t.testing.assert_close(a, b, rtol=0, atol=0)

    def test_budget_expiry_cannot_select_an_unmatched_partial_run(self):
        rows = [dict(speaker=s, views=[self.view()]) for s in ("fit", "selection")]
        with self.assertRaises(TimeoutError):
            matched_refit(self.head, rows, dict(fit="fit", selection="selection"),
                          lambda h, r: dict(objective=1.), 1., 1, .01, 42, deadline=0.)

    def test_both_arms_share_examples_order_initialization_and_selected_updates(self):
        import student_margin as module
        rows = [dict(speaker="fit", views=[self.view(offset=i / 10)]) for i in range(2)]
        rows.append(dict(speaker="selection", views=[self.view(silence=True)]))
        groups = dict(fit="fit", selection="selection")
        calls, validation_calls, origins = [], [], []
        original = module.recording_loss
        def record(head, views, strength):
            calls.append((strength, id(views)))
            return original(head, views, strength)
        def validate(head, selection):
            self.assertEqual([r["speaker"] for r in selection], ["selection"])
            epoch = len(validation_calls) // 2
            if not epoch:
                origins.append(copy.deepcopy(head.state_dict()))
            validation_calls.append(epoch)
            return dict(objective=[10., 5., 1., 4.][epoch], per=0.)
        with patch.object(module, "recording_loss", side_effect=record):
            heads, report = matched_refit(self.head, rows, groups, validate, 1., 3, .001, 42)
        for state in origins:
            for name in state:
                self.t.testing.assert_close(state[name], self.head.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(report["selectedEpoch"], 2)
        self.assertEqual(report["selectedUpdates"], 4)
        self.assertEqual(report["trainingUpdates"], 6)
        self.assertEqual(len(calls), 12)
        for epoch in range(3):
            sequences = [[i for _, i in calls[epoch * 4 + arm * 2:epoch * 4 + arm * 2 + 2]] for arm in range(2)]
            self.assertEqual(sequences[0], sequences[1])
        self.assertEqual(set(heads), {"ordinary", "margin"})

    def test_additional_holdout_excludes_none_speaker_globally(self):
        from student_margin_train import recordings
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            none = dict(id="none", split="train", kind="none", speaker="held", scored=True,
                        events=[], non_errors=[], unsupported=[], wav="a.wav", ayahs=["synthetic:ayah"])
            (root / "labels.jsonl").write_text(json.dumps(none))
            (root / "transcribed.jsonl").write_text(json.dumps(dict(speaker="held", split="release")))
            with self.assertRaisesRegex(ValueError, "Both authorized"):
                recordings(root / "labels.jsonl", root / "transcribed.jsonl", [],
                           {"ayahs": [{"s": "synthetic", "a": "ayah", "p": "ا"}]})

    def test_enrollment_checks_silence_provenance_and_occurrence_notation(self):
        from student_margin_train import recordings
        from student_head_export import digest
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name in ("none", "speech", "pause"):
                (root / (name + ".wav")).write_bytes(name.encode())
            none = dict(id="none", split="train", kind="none", speaker="n", scored=True,
                        events=[], non_errors=[], unsupported=[], wav="none.wav", ayahs=["synthetic:ayah"])
            (root / "labels.jsonl").write_text(json.dumps(none))
            base = dict(speaker="s", split="train", authorized=True, verified=True,
                        transcriptFormat="ctc-articulations")
            speech = dict(base, phones="بب", wav="speech.wav", sha256=digest(root / "speech.wav"))
            pause = dict(base, kind="silence", phones="", wav="pause.wav", sha256=digest(root / "pause.wav"))
            def enroll(rows):
                (root / "transcribed.jsonl").write_text("\n".join(map(json.dumps, rows)))
                return recordings(root / "labels.jsonl", root / "transcribed.jsonl", [],
                                  {"ayahs": [{"s": "synthetic", "a": "ayah", "p": "بباا"}]})
            rows, _ = enroll([speech, pause])
            self.assertEqual([r["phones"] for r in rows], ["ببا", "بب", ""])
            for bad in (dict(pause, kind="none"), dict(pause, phones="ب"),
                        dict(speech, transcriptFormat="duration-expanded"),
                        dict(speech, sha256="wrong")):
                with self.assertRaises(ValueError):
                    enroll([speech, bad])

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

    @unittest.skipUnless(TORCH, "Torch required for bundle test")
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
            graft(source, phone_head(w + .1, b), target, "margin")
            exported = json.loads((target / "manifest.json").read_text())
            self.assertEqual(exported["sha256"], digest(target / "model.onnx"))
            for key in ("tokens", "blank", "frameMs", "sampleRate", "shift", "window", "fbank", "states"):
                self.assertEqual(exported[key], manifest[key])
            self.assertEqual(insertion_assets(target), before)
            self.assertEqual(digest(source / "model.onnx"), manifest["sha256"])


if __name__ == "__main__":
    unittest.main()

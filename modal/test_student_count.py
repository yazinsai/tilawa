"""Exact path enumeration, gradient, enrollment and native export contracts.

Run with Torch, ONNX and ONNX Runtime on the supplied supervisor:
python3 -m unittest discover -s modal -p test_student_count.py -v
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

from student_count import (expected_counts, transcript_counts, count_loss, reference_articulations, speaker_partition,
    admitted_none, phone_head, log_probs, objectives, matched_refit)

TORCH = importlib.util.find_spec("torch") is not None
ONNX = importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None


def collapsed(path):
    result, previous = [], None
    for label in path:
        if label != previous and label != 0:
            result.append(label)
        previous = label
    return result


def enumerate_counts(p):
    counts = np.zeros(p.shape[-1] - 1)
    for path in itertools.product(range(p.shape[-1]), repeat=len(p)):
        probability = np.prod([p[t, label] for t, label in enumerate(path)])
        for label in collapsed(path):
            counts[label - 1] += probability
    return counts


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


@unittest.skipUnless(TORCH, "Torch required on supervisor")
class CountTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.t = torch

    def test_expectation_matches_enumeration_of_every_ctc_path(self):
        rng = np.random.default_rng(8)
        for n in range(1, 5):
            p = rng.dirichlet([.4, .2, .3], size=n)
            got = expected_counts(self.t.tensor(p, dtype=self.t.float64)).numpy()
            np.testing.assert_allclose(got, enumerate_counts(p), atol=1e-14)

    def test_sustained_labels_count_once_and_blank_separates_articulations(self):
        for path, wanted in [([1, 1, 1, 1], [1., 0.]), ([1, 0, 1, 1], [2., 0.]),
                             ([0, 1, 2, 1], [2., 1.]), ([0, 0, 0], [0., 0.])]:
            p = self.t.nn.functional.one_hot(self.t.tensor(path), 3).double()
            np.testing.assert_array_equal(expected_counts(p).numpy(), wanted)

    def test_valid_lengths_ignore_padding_and_its_gradient(self):
        t = self.t
        logits = t.randn(3, 6, 4, dtype=t.float64, requires_grad=True)
        lengths = t.tensor([6, 2, 0])
        got = expected_counts(logits.softmax(-1), lengths)
        for i, n in enumerate(lengths):
            np.testing.assert_allclose(got[i].detach(), enumerate_counts(logits[i, :n].softmax(-1).detach().numpy()), atol=1e-13)
        got.sum().backward()
        self.assertEqual(float(logits.grad[1, 2:].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[2].abs().sum()), 0.)
        self.assertEqual(expected_counts(t.empty(0, 4, dtype=t.float64)).shape, (3,))

    def test_blank_index_is_not_a_phone(self):
        p = self.t.tensor([[.2, .7, .1], [.3, .4, .3]], dtype=self.t.float64)
        got = expected_counts(p, blank=1)
        np.testing.assert_allclose(got, [.2 + .3 * .8, .1 + .3 * .9], atol=1e-14)

    def test_gradient_matches_finite_difference_of_enumerated_paths(self):
        t = self.t
        logits = t.tensor([[1., .2, -.4], [.3, -.5, .1], [-.4, .2, .8]], dtype=t.float64, requires_grad=True)
        (expected_counts(logits.softmax(-1)) ** 2).sum().backward()
        x = logits.detach().numpy()
        for index in np.ndindex(x.shape):
            a, b = x.copy(), x.copy()
            a[index] += 1e-5
            b[index] -= 1e-5
            softmax = lambda z: np.exp(z) / np.exp(z).sum(-1, keepdims=True)
            want = (sum(enumerate_counts(softmax(a)) ** 2) - sum(enumerate_counts(softmax(b)) ** 2)) / 2e-5
            self.assertAlmostEqual(float(logits.grad[index]), want, places=8)

    def test_transcript_counts_preserve_real_repeated_articulations(self):
        counts = transcript_counts(self.t.tensor([1, 1, 2]), 4)
        np.testing.assert_array_equal(counts, [2, 1, 0])
        np.testing.assert_array_equal(transcript_counts(self.t.tensor([], dtype=self.t.int64), 4), [0, 0, 0])
        for target in ([0], [4], [-1]):
            with self.assertRaises(ValueError):
                transcript_counts(self.t.tensor(target), 4)

    def test_zero_count_silence_penalizes_phone_emissions(self):
        t = self.t
        target = t.tensor([], dtype=t.int64)
        blank = t.tensor([[.999, .0005, .0005]] * 5).log()
        speech = t.tensor([[.5, .25, .25]] * 5).log()
        self.assertGreater(float(count_loss(speech, target)), float(count_loss(blank, target)))
        ctc, count, preserve = objectives(speech, target, speech.clone())
        self.assertAlmostEqual(float(ctc), -np.log(.5), places=6)
        self.assertAlmostEqual(float(preserve), 0., places=7)
        self.assertGreater(float(count), 0.)

    def test_blank_winning_frames_still_receive_missing_phone_count_gradient(self):
        t = self.t
        logits = t.tensor([[.97, .02, .01]] * 4, dtype=t.float64).log().requires_grad_()
        target = t.tensor([1])
        self.assertEqual(collapsed(logits.argmax(-1).tolist()), [])
        self.assertGreater(float(expected_counts(logits.softmax(-1))[0].detach()), 0.)
        count_loss(logits.log_softmax(-1), target).backward()
        self.assertLess(float(logits.grad[:, 1].sum()), 0.)
        self.assertGreater(float(logits.grad[:, 0].sum()), 0.)

    def test_invalid_probability_and_frame_lengths_are_rejected(self):
        for p in ([[.2, .2]], [[-1., 2.]], [[float("nan"), 1.]]):
            with self.assertRaises(ValueError):
                expected_counts(self.t.tensor(p))
        p = self.t.tensor([[.5, .5], [.5, .5]])
        for lengths in (3, -1, 1.5, [1, 2]):
            with self.assertRaises(ValueError):
                expected_counts(p, lengths)


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
        ctc, count, preserve = objectives(log_probs(self.head, v["features"]), v["target"], v["shipped"])
        self.assertAlmostEqual(float(preserve.detach()), 0., places=7)
        (ctc + count + preserve).backward()
        self.assertIsNone(v["features"].grad)
        self.assertIsNone(v["shipped"].grad)
        self.assertGreater(float(self.head.bias.grad.abs().sum()), 0.)

    def test_impossible_repeated_target_is_rejected(self):
        v = self.view(n=2)
        with self.assertRaises(ValueError):
            objectives(v["shipped"], self.t.tensor([1, 1]), v["shipped"])

    def test_both_arms_share_examples_order_initialization_and_selected_updates(self):
        import student_count as module
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
            epoch = len(validation_calls) // 3
            if not epoch:
                origins.append(copy.deepcopy(head.state_dict()))
            validation_calls.append(epoch)
            return dict(objective=[10., 5., 1., 4.][epoch], per=0.)
        with patch.object(module, "recording_loss", side_effect=record):
            heads, report = matched_refit(self.head, rows, groups, validate, (.1, 1.), 3, .001, 42)
        for state in origins:
            for name in state:
                self.t.testing.assert_close(state[name], self.head.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(report["selectedEpoch"], 2)
        self.assertEqual(report["selectedUpdates"], 4)
        self.assertEqual(report["trainingUpdates"], 6)
        self.assertEqual(len(calls), 18)
        for epoch in range(3):
            sequences = [[i for _, i in calls[epoch * 6 + arm * 2:epoch * 6 + arm * 2 + 2]] for arm in range(3)]
            self.assertEqual(sequences[0], sequences[1])
            self.assertEqual(sequences[0], sequences[2])
        self.assertEqual(set(heads), {"ordinary", "count"})

    def test_additional_holdout_excludes_none_speaker_globally(self):
        from student_count_train import recordings
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
        from student_count_train import recordings
        from student_count_export import digest
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
        from student_count_export import head_parameters
        for quantized in (False, True):
            graph, w, b = self.graph(quantized)
            got_w, got_b = head_parameters(graph)
            np.testing.assert_allclose(got_w, w, atol=1e-7)
            np.testing.assert_array_equal(got_b, b)

    def test_replacement_preserves_encoder_and_states_and_removes_old_head(self):
        import onnxruntime as ort
        from student_count_export import replace_head
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
        from student_count_export import insertion_assets
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
        from student_count_export import digest, graft, insertion_assets
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
            graft(source, phone_head(w + .1, b), target, "count")
            exported = json.loads((target / "manifest.json").read_text())
            self.assertEqual(exported["sha256"], digest(target / "model.onnx"))
            for key in ("tokens", "blank", "frameMs", "sampleRate", "shift", "window", "fbank", "states"):
                self.assertEqual(exported[key], manifest[key])
            self.assertEqual(insertion_assets(target), before)
            self.assertEqual(digest(source / "model.onnx"), manifest["sha256"])


if __name__ == "__main__":
    unittest.main()

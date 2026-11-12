"""Synthetic SVRG, enrollment and streaming-export contracts; no lab audio."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import onnx
import onnxruntime

from student_variance import (speaker_partition, admitted_none, phone_head, log_probs,
    objectives, recording_loss, gradient, full_gradient, paired_gradient, matched_refit,
    consonant_instability)

TORCH = ONNX = True

class VarianceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.head = phone_head(np.array([[.1, .2], [-.2, .3], [.3, -.1]], np.float32), np.zeros(3, np.float32))

    def view(self, n=8, offset=0.):
        x = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2) / 10 + offset
        frozen = log_probs(self.head, x).detach()
        mask = torch.ones_like(x)
        mask[::2, 0] = 0
        return dict(features=x, fitFeatures=x * mask / .9, shipped=frozen,
                    teacher=frozen.clone(), target=torch.tensor([1, 2]))

    def rows(self):
        return [dict(speaker="fit", views=[self.view(), self.view(n=15, offset=1)]),
                dict(speaker="fit", views=[self.view(n=32, offset=-1)]),
                dict(speaker="selection", views=[self.view(offset=2)])]

    def test_full_gradient_matches_complete_objective_with_unequal_durations(self):
        rows = self.rows()[:2]
        loss = sum(recording_loss(self.head, r["views"]) for r in rows) / len(rows)
        expected = torch.autograd.grad(loss, tuple(self.head.parameters()))
        for got, want in zip(full_gradient(self.head, rows), expected):
            torch.testing.assert_close(got, want)

    def test_paired_estimator_has_exact_current_gradient_expectation(self):
        rows = self.rows()[:2]
        current = copy.deepcopy(self.head)
        with torch.no_grad():
            current.weight.add_(torch.tensor([[.2, -.3], [-.1, .4], [.5, -.2]]))
            current.bias.add_(.1)
        reference_mean = full_gradient(self.head, rows)
        samples = [paired_gradient(current, self.head, r, reference_mean) for r in rows]
        for i, expected in enumerate(full_gradient(current, rows)):
            torch.testing.assert_close(torch.stack([s[i] for s in samples]).mean(0), expected)

    def test_variance_disappears_at_reference_without_changing_recording_weights(self):
        rows = self.rows()[:2]
        mean = full_gradient(self.head, rows)
        self.assertFalse(torch.allclose(gradient(self.head, rows[0])[0], gradient(self.head, rows[1])[0]))
        for row in rows:
            for got, want in zip(paired_gradient(self.head, self.head, row, mean), mean):
                torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_frozen_features_teacher_and_preservation_receive_no_gradient(self):
        v = self.view()
        for k in ("features", "fitFeatures", "teacher", "shipped"):
            v[k].requires_grad_()
        recording_loss(self.head, [v]).backward()
        for k in ("features", "fitFeatures", "teacher", "shipped"):
            self.assertIsNone(v[k].grad)
        self.assertGreater(float(self.head.bias.grad.abs().sum()), 0.)
        components = objectives(log_probs(self.head, v["features"]), v["target"], v["teacher"], v["shipped"])
        self.assertAlmostEqual(float(components[2].detach()), 0., places=7)

    def test_compute_matching_counts_full_and_paired_gradients_and_excludes_selection(self):
        import student_variance as module
        rows, calls, validation = self.rows(), [], []
        real = module.gradient
        def counted(head, row):
            calls.append(row["speaker"])
            return real(head, row)
        def validate(head, selected):
            self.assertEqual([r["speaker"] for r in selected], ["selection"])
            validation.append(id(head))
            return dict(objective=1.)
        with patch.object(module, "gradient", side_effect=counted):
            heads, report = matched_refit(self.head, rows, dict(fit="fit", selection="selection"),
                validate, (1,), 1, .001)
        self.assertEqual(report["gradientEvaluationsPerArm"], 6)
        self.assertEqual(report["updates"], {"ordinary": 6, "1": 2})
        self.assertEqual(calls, ["fit"] * 12)
        self.assertEqual(len(validation), 4)
        self.assertEqual(report["selectedEvaluations"], 0)
        for head in heads.values():
            for p, q in zip(head.parameters(), self.head.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_refresh_and_checkpoint_selected_only_at_equal_compute(self):
        seen, counts = [], {}
        def validate(head, selected):
            key = id(head)
            i = counts.get(key, 0)
            counts[key] = i + 1
            return dict(objective=[10., 5., 1.][i])
        _, report = matched_refit(self.head, self.rows(), dict(fit="fit", selection="selection"),
            validate, (1, 4), 2, .001, checkpoint=lambda a, n, h: seen.append((a, n)))
        self.assertEqual(report["selectedEvaluations"], 36)
        self.assertEqual(report["gradientEvaluationsPerArm"], 36)
        self.assertEqual(report["updates"], {"ordinary": 36, "1": 12, "4": 16})
        for hist in report["history"].values():
            self.assertEqual([r["evaluations"] for r in hist], [0, 18, 36])
        self.assertEqual([n for a, n in seen if a == "ordinary"], list(range(0, 37, 2)))

    def test_ordinary_premise_stage_is_identical_to_full_comparison_control(self):
        ordinary = []
        args = (self.head, self.rows(), dict(fit="fit", selection="selection"), lambda h, r: dict(objective=1.))
        matched_refit(*args, (1,), 1, .001, checkpoint=lambda a, n, h:
            ordinary.append((n, copy.deepcopy(h.state_dict()))) if a == "ordinary" else None,
            ordinary_only=True)
        repeated = []
        matched_refit(*args, (1,), 1, .001, checkpoint=lambda a, n, h:
            repeated.append((n, copy.deepcopy(h.state_dict()))) if a == "ordinary" else None)
        self.assertEqual([n for n, s in ordinary], [n for n, s in repeated])
        for (_, a), (_, b) in zip(ordinary, repeated):
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_impossible_alignment_is_rejected(self):
        v = self.view(n=2)
        with self.assertRaises(ValueError):
            objectives(v["shipped"], torch.tensor([1, 1]), v["teacher"], v["shipped"])

    def test_instability_requires_missing_supported_phone_and_same_frame_crossing(self):
        x = torch.zeros(2, 2)
        teacher = torch.tensor([[0., 3., 1.], [0., 2., 1.]])
        baseline = torch.tensor([[3., 0., 1.], [3., 0., 1.]])
        heads = [phone_head(np.zeros((3, 2), np.float32), np.array([3., b, 1.], np.float32)) for b in (2., 4., 2.)]
        got = consonant_instability(heads, x, teacher, [(0, 2, 1)], baseline)
        self.assertTrue(got["changesSign"])
        self.assertEqual(got["occurrences"][0]["frame"], 0)
        self.assertEqual(got["occurrences"][0]["margins"], [-1., 1., -1.])
        self.assertFalse(consonant_instability(heads, x, baseline, [(0, 2, 1)], baseline)["changesSign"])
        self.assertFalse(consonant_instability(heads, x, teacher, [(0, 2, 1)], teacher)["changesSign"])
        self.assertFalse(consonant_instability(heads[1:2] * 2, x, teacher, [(0, 2, 1)], baseline)["changesSign"])
        with self.assertRaises(ValueError):
            consonant_instability(heads, x, teacher * float("nan"), [(0, 2, 1)], baseline)

    def test_invalid_compute_schedule_fails_before_training(self):
        for schedule, wall in (((0,), 60), ((1, 1), 60), ((1,), 0), ((1,), 90000)):
            with self.assertRaises(ValueError):
                matched_refit(self.head, self.rows(), dict(fit="fit", selection="selection"),
                              lambda h, r: dict(objective=1.), schedule, wall_seconds=wall)

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

    def test_cross_corpus_release_speaker_is_excluded_globally(self):
        from student_variance_train import recordings
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            row = dict(id="none", split="train", kind="none", speaker="held", scored=True,
                events=[], non_errors=[], unsupported=[], wav="a.wav", ayahs=["synthetic:ayah"])
            (root / "labels.jsonl").write_text(json.dumps(row))
            (root / "enrolled.jsonl").write_text(json.dumps(dict(speaker="held", split="release")))
            with self.assertRaisesRegex(ValueError, "Authorized training-speaker"):
                recordings(root / "labels.jsonl", root / "enrolled.jsonl", [],
                    {"ayahs": [{"s": "synthetic", "a": "ayah", "p": "a"}]})

    def test_audio_enrollment_cannot_leave_its_corpus(self):
        from student_variance_train import contained
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                contained(root, "../outside.wav")

    def test_authorized_none_audio_does_not_require_another_corpus(self):
        from student_variance_train import recordings
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "a.wav").write_bytes(b"synthetic-audio-identity")
            row = dict(id="none", split="train", kind="none", speaker="fit", scored=True,
                events=[], non_errors=[], unsupported=[], wav="a.wav", ayahs=["synthetic:ayah"])
            (root / "labels.jsonl").write_text(json.dumps(row))
            rows, excluded = recordings(root / "labels.jsonl", None, [],
                {"ayahs": [{"s": "synthetic", "a": "ayah", "p": "ب"}]})
            self.assertEqual([r["source"] for r in rows], ["help-none"])
            self.assertEqual(rows[0]["phones"], "ب")
            self.assertEqual(excluded, [])


class AuditTests(unittest.TestCase):
    def test_premise_counts_targets_and_excludes_initialization(self):
        from argparse import Namespace
        from student_variance_train import audit, TOKENS
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            width, vocab, phone = 2, len(TOKENS) + 1, 1
            for evaluations, margin in ((0, 5.), (2, -1.), (4, 1.)):
                bias = np.zeros(vocab, np.float32)
                bias[phone] = margin
                head = phone_head(np.zeros((vocab, width), np.float32), bias)
                torch.save(dict(state=head.state_dict(), sourceSha256="synthetic",
                                evaluations=evaluations), checkpoints / f"{evaluations:09d}.pt")
            teacher, baseline = np.zeros((3, vocab), np.float32), np.zeros((3, vocab), np.float32)
            teacher[:, phone] = 4.
            baseline[:, 0] = 3.
            np.savez(root / "case.npz", features=np.zeros((3, width), np.float32),
                     teacher=teacher, shipped=baseline)
            target = dict(id="synthetic-target", cache="case.npz", consonants=[[0, 1, phone], [1, 3, phone]])
            spec = dict(sourceSha256="synthetic", targets=[target])
            (root / "diagnostics.json").write_text(json.dumps(spec))
            args = Namespace(checkpoints=checkpoints, diagnostics=root / "diagnostics.json", out=root / "result.json")
            audit(args)
            result = json.loads(args.out.read_text())
            self.assertEqual(result["checkpointEvaluations"], [2, 4])
            self.assertEqual(result["targetsWithSignChange"], 1)
            self.assertTrue(result["premiseKill"])
            spec["targets"].append(dict(target, id="synthetic-other"))
            args.diagnostics.write_text(json.dumps(spec))
            audit(args)
            self.assertFalse(json.loads(args.out.read_text())["premiseKill"])
            spec["targets"].append(target)
            args.diagnostics.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "distinct diagnostic"):
                audit(args)


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
        from student_variance_export import head_parameters
        for quantized in (False, True):
            graph, w, b = self.graph(quantized)
            got_w, got_b = head_parameters(graph)
            np.testing.assert_allclose(got_w, w, atol=1e-7)
            np.testing.assert_array_equal(got_b, b)

    def test_replacement_preserves_encoder_and_states_and_removes_old_head(self):
        import onnxruntime as ort
        from student_variance_export import replace_head
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
        from student_variance_export import insertion_assets
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
        from student_variance_export import digest, graft, insertion_assets
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
            graft(source, phone_head(w + .1, b), target, "variance")
            exported = json.loads((target / "manifest.json").read_text())
            self.assertEqual(exported["sha256"], digest(target / "model.onnx"))
            for key in ("tokens", "blank", "frameMs", "sampleRate", "shift", "window", "fbank", "states"):
                self.assertEqual(exported[key], manifest[key])
            self.assertEqual(insertion_assets(target), before)
            self.assertEqual(digest(source / "model.onnx"), manifest["sha256"])

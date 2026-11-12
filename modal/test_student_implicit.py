"""Synthetic numerical/streaming export checks; no practice inputs.

python3 -m unittest discover -s modal -p test_student_implicit.py -v
Requires torch, numpy, onnx and onnxruntime. Missing dependencies fail explicitly.
"""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
import onnx
import onnxruntime
from student_implicit import (speaker_partition, admitted_none, phone_head, log_probs,
    recording_loss, implicit_update, explicit_update, matched_refit)

TORCH = ONNX = True


class ImplicitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.head = phone_head([[.1], [-.2], [.3]], [0., 0., 0.])

    def view(self, n=5):
        x = torch.linspace(-1., 1., n)[:, None]
        return dict(features=x, target=torch.tensor([1, 2]),
                    shipped=log_probs(self.head, x).detach())

    def test_quadratic_matches_implicit_equation_and_differs_from_explicit(self):
        head = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            head.weight.fill_(2.)
        objective = lambda h: 3 * h.weight.square().sum() / 2
        ordinary = copy.deepcopy(head)
        diag = implicit_update(head, objective, .5)
        explicit_update(ordinary, objective, .5)
        self.assertTrue(diag["accepted"])
        self.assertAlmostEqual(float(head.weight.detach()), 2 / (1 + .5 * 3), places=6)
        self.assertAlmostEqual(float(ordinary.weight.detach()), -1., places=6)
        self.assertLess(diag["proposed"], diag["initial"])

    def test_ctc_is_complete_forward_sum_and_has_no_frozen_gradients(self):
        view = self.view(n=4)
        view["features"].requires_grad_()
        view["shipped"].requires_grad_()
        lp = log_probs(self.head, view["features"])
        def collapse(path):
            return [p for i, p in enumerate(path) if p and (not i or p != path[i-1])]
        paths = [sum(lp[t, p] for t, p in enumerate(path))
                 for path in itertools.product(range(3), repeat=4)
                 if collapse(path) == [1, 2]]
        expected = -torch.logsumexp(torch.stack(paths), 0) / 2
        loss = recording_loss(self.head, [view])
        self.assertAlmostEqual(float(loss.detach()), float(expected.detach()), places=10)
        loss.backward()
        self.assertIsNone(view["features"].grad)
        self.assertIsNone(view["shipped"].grad)
        self.assertGreater(float(self.head.weight.grad.abs().sum()), 0.)

    def test_full_ctc_proximal_residual_uses_candidate_parameters(self):
        view, visited = self.view(n=8), []
        previous = [p.detach().clone() for p in self.head.parameters()]
        def objective(head):
            visited.append(head.weight.detach().clone())
            return recording_loss(head, [view])
        diag = implicit_update(self.head, objective, .03)
        self.assertTrue(diag["accepted"])
        self.assertEqual(diag["evaluations"], len(visited))
        self.assertTrue(any(not torch.equal(visited[0], w) for w in visited[1:]))
        self.assertLess(diag["residual"], 1e-6)
        self.head.zero_grad(set_to_none=True)
        recording_loss(self.head, [view]).backward()
        for p, old in zip(self.head.parameters(), previous):
            self.assertLess(float((p.grad + (p.detach() - old) / .03).abs().max()), 1e-6)

    def test_full_ctc_gradient_matches_finite_difference(self):
        view = self.view()
        recording_loss(self.head, [view]).backward()
        analytic = float(self.head.weight.grad[1, 0])
        values = []
        with torch.no_grad():
            original = self.head.weight[1, 0].clone()
            for delta in (-1e-5, 1e-5):
                self.head.weight[1, 0].copy_(original + delta)
                values.append(float(recording_loss(self.head, [view])))
            self.head.weight[1, 0].copy_(original)
        self.assertAlmostEqual(analytic, (values[1] - values[0]) / 2e-5, places=7)

    def test_nonfinite_solve_restores_preceding_parameters(self):
        before = copy.deepcopy(self.head.state_dict())
        calls = 0
        def objective(h):
            nonlocal calls
            calls += 1
            return h.weight.square().sum() * (float("nan") if calls > 1 else 1.)
        with self.assertRaises(ValueError):
            implicit_update(self.head, objective, .1)
        for k, v in before.items():
            torch.testing.assert_close(self.head.state_dict()[k], v, rtol=0., atol=0.)

    def test_unconverged_solve_rolls_back(self):
        before = copy.deepcopy(self.head.state_dict())
        # Suppress the inner optimizer to leave a nonstationary candidate.
        with patch.object(torch.optim.LBFGS, "step", lambda self, closure: closure()):
            diag = implicit_update(self.head, lambda h: recording_loss(h, [self.view()]), .1)
        self.assertFalse(diag["accepted"])
        self.assertEqual(diag["distance"], 0.)
        for k, v in before.items():
            torch.testing.assert_close(self.head.state_dict()[k], v, rtol=0., atol=0.)

    def test_recording_views_are_equal_weight_and_impossible_ctc_rejected(self):
        short, long = self.view(5), self.view(20)
        expected = (recording_loss(self.head, [short]) + recording_loss(self.head, [long])) / 2
        torch.testing.assert_close(recording_loss(self.head, [short, long]), expected)
        invalid = self.view(2)
        invalid["target"] = torch.tensor([1, 1])
        with self.assertRaises(ValueError):
            recording_loss(self.head, [invalid])

    def test_controls_match_recordings_and_actual_compute_and_select_shared_epoch(self):
        rows = [dict(speaker="fit", views=[self.view(5)]), dict(speaker="fit", views=[self.view(6)]),
                dict(speaker="selection", views=[self.view(7)])]
        import student_implicit as module
        seen = []
        original = module.recording_loss
        def record(head, views):
            seen.append(id(views))
            return original(head, views)
        validations = []
        def validate(head, selection):
            self.assertEqual([r["speaker"] for r in selection], ["selection"])
            # Initial validation then six validations per shared epoch.
            epoch = 0 if not validations else (len(validations) - 1) // 6 + 1
            validations.append(epoch)
            return dict(objective=[10., 5., 1., 4.][epoch], per=0.)
        with patch.object(module, "recording_loss", side_effect=record):
            heads, report = matched_refit(self.head, rows, {"fit": "fit", "selection": "selection"},
                validate, (.01, .03), epochs=3)
        self.assertEqual(report["selectedEpoch"], 2)
        self.assertEqual(set(heads), {"implicit", "ordinary", "multi-step"})
        cursor = 0
        for row in report["ledger"]:
            self.assertEqual(row["implicit"], row["multiStep"])
            self.assertEqual(row["ordinary"], 1)
            n = row["implicit"] + 1 + row["multiStep"]
            self.assertEqual(seen[cursor:cursor+n], [id(rows[row["recording"]]["views"])] * n)
            cursor += n
        self.assertEqual(cursor, len(seen))


class EnrollmentTests(unittest.TestCase):
    def test_runner_cli_help_is_available(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("student_implicit_train.py")),
                                 "prepare", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--help-labels", result.stdout)

    def test_speaker_split_is_order_invariant(self):
        speakers = ["synthetic-" + str(i) for i in range(10)]
        a = speaker_partition(speakers)
        self.assertEqual(a, speaker_partition(list(reversed(speakers)) + speakers))
        self.assertEqual(list(a.values()).count("selection"), 2)

    def test_holdout_repeat_and_ambiguous_none_excluded(self):
        good = dict(id="synthetic", split="train", kind="none", speaker="fit", scored=True,
                    events=[], non_errors=[], unsupported=[], wav="x.wav", ayahs=["synthetic"])
        bad = [dict(good, id="repeat", non_errors=[{"kind": "repeat"}]),
               dict(good, id="script", kind="skipped_word"), dict(good, id="unscored", scored=False)]
        self.assertEqual(admitted_none([good, *bad]), [good])
        self.assertEqual(admitted_none([good, dict(good, split="eval")]), [])
        self.assertEqual(admitted_none([good], ["fit"]), [])

    def test_cross_corpus_holdout_excludes_none_speaker(self):
        from student_implicit_train import recordings
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            none = dict(id="none", split="train", kind="none", speaker="held", scored=True,
                        events=[], non_errors=[], unsupported=[], wav="a.wav", ayahs=["synthetic:ayah"])
            (root / "labels.jsonl").write_text(json.dumps(none))
            (root / "transcribed.jsonl").write_text(json.dumps(dict(speaker="held", split="release")))
            with self.assertRaisesRegex(ValueError, "No authorized"):
                recordings(root / "labels.jsonl", root / "transcribed.jsonl", [],
                           {"ayahs": [{"s": "synthetic", "a": "ayah", "p": "a"}]})

    def test_supplied_none_corpus_needs_no_external_transcripts_and_collapses_duration(self):
        from student_implicit_train import recordings
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "a.wav").write_bytes(b"synthetic-audio-identity")
            none = dict(id="none", split="train", kind="none", speaker="fit", scored=True,
                        events=[], non_errors=[], unsupported=[], wav="a.wav", ayahs=["synthetic:ayah"])
            (root / "labels.jsonl").write_text(json.dumps(none))
            rows, excluded = recordings(root / "labels.jsonl", None, [],
                {"ayahs": [{"s": "synthetic", "a": "ayah", "p": "ببِ"}]})
            self.assertEqual(rows[0]["phones"], "بِ")
            self.assertEqual(rows[0]["source"], "help-none")
            self.assertEqual(excluded, [])
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
        from student_implicit_export import head_parameters
        for quantized in (False, True):
            graph, w, b = self.graph(quantized)
            got_w, got_b = head_parameters(graph)
            np.testing.assert_allclose(got_w, w, atol=1e-7)
            np.testing.assert_array_equal(got_b, b)

    def test_replacement_preserves_encoder_and_states_and_removes_old_head(self):
        import onnxruntime as ort
        from student_implicit_export import replace_head
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
        from student_implicit_export import insertion_assets
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
        from student_implicit_export import digest, graft, insertion_assets
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
            graft(source, phone_head(w + .1, b), target, "implicit")
            exported = json.loads((target / "manifest.json").read_text())
            self.assertEqual(exported["sha256"], digest(target / "model.onnx"))
            for key in ("tokens", "blank", "frameMs", "sampleRate", "shift", "window", "fbank", "states"):
                self.assertEqual(exported[key], manifest[key])
            self.assertEqual(insertion_assets(target), before)
            self.assertEqual(digest(source / "model.onnx"), manifest["sha256"])


if __name__ == "__main__":
    unittest.main()

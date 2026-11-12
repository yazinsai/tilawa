"""Synthetic gradient/history tests; no practice recordings or labels.

Run on supervisor: python3 -m unittest discover -s modal -p test_student_optimizer_memory.py -v
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
import torch
import student_optimizer_memory as module
from student_optimizer_memory import (speaker_partition, admitted_none, phone_head,
    log_probs, ctc_loss, occupancy, teacher_distribution, objectives, gradient_parts,
    AcousticMoments, WholeGradientClip, matched_refit)

TORCH = True
ONNX = importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None


class OccupancyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_conditioned_occupancy_matches_exhaustive_ctc_paths(self):
        lp = torch.tensor([[.5, .3, .2], [.1, .6, .3], [.4, .2, .4], [.2, .5, .3]], dtype=torch.float64).log()
        for transcript in ([1, 2], [1, 1], []):
            target = torch.tensor(transcript, dtype=torch.int64)
            total, expected = 0., torch.zeros_like(lp)
            for path in itertools.product(range(3), repeat=len(lp)):
                collapsed = [p for i, p in enumerate(path) if p and (i == 0 or p != path[i-1])]
                if collapsed != transcript:
                    continue
                weight = float(lp.exp()[torch.arange(len(lp)), list(path)].prod())
                total += weight
                expected[torch.arange(len(lp)), list(path)] += weight
            torch.testing.assert_close(occupancy(lp, target), expected / total, atol=1e-12, rtol=1e-12)
            self.assertAlmostEqual(float(ctc_loss(lp, target)), -np.log(total) / len(lp), places=12)

    def test_impossible_repeated_transcript_and_blank_target_rejected(self):
        lp = torch.zeros(2, 3).log_softmax(-1)
        for target in ([1, 1], [0], [3]):
            with self.assertRaises(ValueError):
                occupancy(lp, torch.tensor(target))

    def test_teacher_top_five_and_clamped_end_match_shipped_loss(self):
        teacher = torch.arange(14, dtype=torch.float64).reshape(2, 7).log_softmax(-1)
        q = teacher_distribution(teacher, 4)
        values, ids = teacher.exp()[[0, 1, 1, 1]].topk(5, -1)
        expected = torch.zeros_like(q).scatter_(-1, ids, values / values.sum(-1, keepdim=True))
        torch.testing.assert_close(q, expected)
        self.assertTrue(torch.equal(q[1], q[3]))
        self.assertTrue(torch.equal((q > 0).sum(-1), torch.full((4,), 5)))

    def test_gradient_sum_preserves_original_weights_and_freezes_inputs(self):
        head = phone_head(torch.tensor([[.1, .2], [-.2, .3], [.3, -.1]], dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
        x = torch.arange(16, dtype=torch.float64).reshape(8, 2).div(10).requires_grad_()
        teacher = torch.arange(24, dtype=torch.float64).reshape(8, 3).div(10).log_softmax(-1).requires_grad_()
        views = [dict(features=x, teacher=teacher, target=torch.tensor([1, 2, 1]))]
        parts, masses, _ = gradient_parts(head, views, "بَ")
        ctc, kd = objectives(log_probs(head, x), views[0]["target"], teacher)
        w, b = torch.autograd.grad(ctc + 2 * kd, (head.weight, head.bias))
        torch.testing.assert_close(sum(parts), torch.cat([w, b[:, None]], 1), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(masses.sum(), torch.tensor(1., dtype=masses.dtype))
        self.assertIsNone(x.grad)
        self.assertIsNone(teacher.grad)
        self.assertTrue(all(float(m) > 0 for m in masses))

    def test_blank_only_supervision_has_no_other_regime_gradient(self):
        head = phone_head(torch.zeros(3, 2), torch.zeros(3))
        view = dict(features=torch.ones(5, 2), teacher=torch.zeros(5, 3).log_softmax(-1), target=torch.tensor([], dtype=torch.int64))
        parts, masses, _ = gradient_parts(head, [view], "بَ")
        torch.testing.assert_close(masses, torch.tensor([0., 0., 1.]))
        self.assertFalse(torch.count_nonzero(parts[0]) or torch.count_nonzero(parts[1]))

    def test_per_frame_reduction_and_equal_view_weight(self):
        head = phone_head(torch.zeros(3, 2), torch.zeros(3))
        views = [dict(features=torch.ones(n, 2), teacher=torch.zeros(n, 3).log_softmax(-1), target=torch.tensor([1, 2])) for n in (5, 11)]
        _, _, diag = gradient_parts(head, views, "بَ")
        expected = sum(float(ctc_loss(log_probs(head, v["features"]), v["target"])) for v in views) / 2
        self.assertAlmostEqual(diag["ctc"], expected, places=6)

    def test_nasal_symbols_use_duration_memory_without_relabelling(self):
        head = phone_head(torch.zeros(5, 2), torch.zeros(5))
        target = torch.tensor([1, 2, 3, 4])
        view = dict(features=torch.ones(8, 2), teacher=torch.zeros(8, 5).log_softmax(-1), target=target)
        parts, masses, _ = gradient_parts(head, [view], "منں۾")
        self.assertEqual(float(masses[0]), 0.)
        self.assertFalse(torch.count_nonzero(parts[0]))
        self.assertTrue(torch.equal(view["target"], target))


class MemoryTests(unittest.TestCase):
    def test_shared_matches_torch_adam_oracle_across_history(self):
        moments = AcousticMoments("shared")
        p = torch.nn.Parameter(torch.tensor([[.3, -.7]], dtype=torch.float64))
        optimizer = torch.optim.Adam([p], lr=.01, betas=(.9, .98), eps=1e-8)
        masses = torch.tensor([.2, .3, .5], dtype=torch.float64)
        for gradient in (torch.tensor([[1., -.5]], dtype=torch.float64), torch.tensor([[-2., 3.]], dtype=torch.float64), torch.tensor([[.1, .2]], dtype=torch.float64)):
            before = p.detach().clone()
            p.grad = gradient
            optimizer.step()
            direction = moments.direction([gradient * m for m in masses], masses)
            torch.testing.assert_close(before - .01 * direction, p.detach(), atol=1e-12, rtol=1e-12)

    def test_occupancy_weights_do_not_triple_identical_updates(self):
        shared, separated = AcousticMoments("shared"), AcousticMoments("separated")
        masses = torch.tensor([.2, .3, .5], dtype=torch.float64)
        for scalar in (2., -1., .1):
            parts = [torch.full((2, 3), scalar, dtype=torch.float64) * m for m in masses]
            torch.testing.assert_close(shared.direction(parts, masses), separated.direction(parts, masses))

    def test_absent_consonant_history_does_not_absorb_blank_updates(self):
        opt = AcousticMoments("separated")
        zero, positive = torch.zeros(1, 1), torch.ones(1, 1)
        opt.direction([positive, zero, zero], torch.tensor([1., 0., 0.]))
        saved = copy.deepcopy(opt.state["consonant"])
        for _ in range(30):
            opt.direction([zero, zero, -positive], torch.tensor([0., 0., 1.]))
        for a, b in zip(saved[:2], opt.state["consonant"][:2]):
            torch.testing.assert_close(a, b)
        self.assertEqual(opt.state["consonant"][2], 1)
        result = opt.direction([positive, zero, zero], torch.tensor([1., 0., 0.]))
        self.assertGreater(float(result), 0.)

    def test_memoryless_has_no_history_and_matches_first_separated_step(self):
        opt, sep = AcousticMoments("memoryless"), AcousticMoments("separated")
        masses = torch.tensor([.2, .3, .5])
        parts = [torch.tensor([[.3, -.2]]), torch.tensor([[-.1, .7]]), torch.tensor([[.4, .2]])]
        first = opt.direction(parts, masses)
        torch.testing.assert_close(first, sep.direction(parts, masses))
        opt.direction([-p for p in parts], masses)
        torch.testing.assert_close(first, opt.direction(parts, masses))
        self.assertEqual(opt.state, {})

    def test_invalid_gradient_does_not_partially_change_history(self):
        opt = AcousticMoments("separated")
        for parts, masses in (([torch.ones(1), torch.ones(1), torch.zeros(1)], torch.tensor([1., 0., 0.])),
                              ([torch.ones(1), torch.full((1,), float("nan")), torch.zeros(1)], torch.tensor([.5, .5, 0.]))):
            with self.assertRaises(ValueError):
                opt.direction(parts, masses)
            self.assertEqual(opt.state, {})

    def test_whole_gradient_clip_retains_relative_regime_weights(self):
        head = phone_head(torch.ones(3, 2), torch.ones(3))
        clipper = WholeGradientClip()
        small = torch.ones(3, 3)
        for _ in range(11):
            self.assertEqual(clipper.factor(head, small), 1.)
        parts = [small * 20, small * 30, small * 50]
        factor = clipper.factor(head, sum(parts))
        self.assertAlmostEqual(factor, .04, places=6)
        torch.testing.assert_close(sum(p * factor for p in parts), sum(parts) * factor)

    def test_clip_rms_refresh_follows_norm_measurement(self):
        head = phone_head(torch.ones(3, 2), torch.ones(3))
        clipper, g = WholeGradientClip(), torch.ones(3, 3)
        for _ in range(3):
            clipper.factor(head, g)
        with torch.no_grad():
            head.weight.fill_(2)
            head.bias.fill_(2)
        clipper.factor(head, g)
        self.assertEqual(float(clipper.norms[3]), 3.)
        self.assertEqual(float(clipper.rms[0]), 2.)


class EnrollmentAndScheduleTests(unittest.TestCase):
    def test_speaker_groups_and_global_holdout_exclusion(self):
        speakers = ["synthetic-" + str(i) for i in range(10)]
        self.assertEqual(speaker_partition(speakers), speaker_partition(speakers[::-1] + speakers))
        self.assertEqual(list(speaker_partition(speakers).values()).count("selection"), 2)
        with self.assertRaises(ValueError):
            speaker_partition(["only"])
        row = dict(id="take", split="train", kind="none", speaker="held", scored=True, events=[], non_errors=[], unsupported=[], wav="x.wav", ayahs=["synthetic"])
        self.assertEqual(admitted_none([row]), [row])
        self.assertEqual(admitted_none([row, dict(row, split="release")]), [])
        for changed in (dict(row, non_errors=[{"kind": "repeat"}]), dict(row, kind="skipped_word")):
            self.assertEqual(admitted_none([changed]), [])

    def test_other_corpus_holdout_excludes_none_speaker_before_loading_audio(self):
        from student_optimizer_memory_train import recordings
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            row = dict(id="take", split="train", kind="none", speaker="held", scored=True,
                events=[], non_errors=[], unsupported=[], wav="missing.wav", ayahs=["synthetic:ayah"])
            (root / "labels.jsonl").write_text(json.dumps(row))
            (root / "transcribed.jsonl").write_text(json.dumps(dict(speaker="held", split="release")))
            with self.assertRaisesRegex(ValueError, "Both authorized"):
                recordings(root / "labels.jsonl", root / "transcribed.jsonl", [],
                    {"ayahs": [{"s": "synthetic", "a": "ayah", "p": "ب"}]})

    def test_corpus_paths_cannot_escape_the_enrolled_root(self):
        from student_optimizer_memory_train import contained
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                contained(root, "../outside.wav")

    def test_matched_order_count_and_shared_only_checkpoint_choice(self):
        head = phone_head(torch.zeros(3, 2), torch.zeros(3))
        def view(offset):
            return dict(features=torch.ones(8, 2) + offset, target=torch.tensor([1, 2]), teacher=torch.zeros(8, 3).log_softmax(-1))
        rows = [dict(speaker="fit", views=[view(i / 10)]) for i in range(2)] + [dict(speaker="selection", views=[view(0)])]
        calls, validation = [], []
        original = module.gradient_parts
        def gradients(head, views, tokens):
            calls.append(id(views))
            return original(head, views, tokens)
        def validate(head, selected):
            self.assertEqual([r["speaker"] for r in selected], ["selection"])
            epoch, arm = divmod(len(validation), 3)
            validation.append(epoch)
            return dict(objective=([4., 1., 3.][epoch] if arm == 0 else [4., 3., .1][epoch]), per=0.)
        with patch.object(module, "gradient_parts", side_effect=gradients):
            heads, report = matched_refit(head, rows, dict(fit="fit", selection="selection"), validate, "بَ", 2, .0001, 42)
        self.assertEqual(report["selectedEpoch"], 1)
        self.assertEqual(report["selectedUpdates"], 2)
        self.assertEqual(report["trainingUpdates"], 4)
        self.assertEqual(set(heads), {"shared", "separated", "memoryless"})
        for epoch in range(2):
            sequences = [calls[epoch*6+arm*2:epoch*6+arm*2+2] for arm in range(3)]
            self.assertEqual(sequences[0], sequences[1])
            self.assertEqual(sequences[0], sequences[2])

    def test_one_day_budget_expires_before_an_update(self):
        head = phone_head(torch.zeros(3, 2), torch.zeros(3))
        rows = [dict(speaker=s, views=[]) for s in ("fit", "selection")]
        with self.assertRaises(TimeoutError):
            matched_refit(head, rows, dict(fit="fit", selection="selection"), lambda h, r: dict(objective=1.), "بَ", 1, .001, 42, deadline=0)

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
        from student_optimizer_memory_export import head_parameters
        for quantized in (False, True):
            graph, w, b = self.graph(quantized)
            got_w, got_b = head_parameters(graph)
            np.testing.assert_allclose(got_w, w, atol=1e-7)
            np.testing.assert_array_equal(got_b, b)

    def test_replacement_preserves_encoder_and_states_and_removes_old_head(self):
        import onnxruntime as ort
        from student_optimizer_memory_export import replace_head
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
        from student_optimizer_memory_export import insertion_assets
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
        from student_optimizer_memory_export import digest, graft, insertion_assets
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
            graft(source, phone_head(w + .1, b), target, "separated")
            exported = json.loads((target / "manifest.json").read_text())
            self.assertEqual(exported["sha256"], digest(target / "model.onnx"))
            for key in ("tokens", "blank", "frameMs", "sampleRate", "shift", "window", "fbank", "states"):
                self.assertEqual(exported[key], manifest[key])
            self.assertEqual(insertion_assets(target), before)
            self.assertEqual(digest(source / "model.onnx"), manifest["sha256"])


if __name__ == "__main__":
    unittest.main()

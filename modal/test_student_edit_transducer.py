import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from student_edit_transducer import (COPY, DELETE, HOLD, INSERT, SUBSTITUTE,
    CrossFittedCorruptions, decode, marginal_log_probability, source_events,
    speaker_folds, transitions, build_head)


def scores(frames, vocabulary=3):
    return ([[[0.] * 5 for _ in range(frames + 1)] for _ in range(frames)],
            [[[0.] * vocabulary for _ in range(frames + 1)] for _ in range(frames)])


def enumerate_scripts(actions, phones, source):
    result = []
    def visit(t, i, output, value, script):
        if t == len(actions):
            if i == len(source):
                result.append((output, value, script))
            return
        for nt, ni, phone, score, action in transitions(actions, phones, source, t, i):
            visit(nt, ni, output + (() if phone is None else (phone,)), value + score,
                  script + (action,))
    visit(0, 0, (), 0., ())
    return result


class LatticeTests(unittest.TestCase):
    def test_forward_sum_matches_exhaustive_scripts_and_normalizes(self):
        a, p = scores(2)
        a[0][0] = [.1, .5, -.2, .7, -.3]
        p[0][0] = [.2, -.5, .4]
        source = [[1, 0, 0], [2, 1, 1]]
        all_paths = enumerate_scripts(a, p, source)
        self.assertAlmostEqual(sum(math.exp(x[1]) for x in all_paths), 1.)
        for target in set(x[0] for x in all_paths):
            expected = sum(math.exp(x[1]) for x in all_paths if x[0] == target)
            self.assertAlmostEqual(math.exp(marginal_log_probability(a, p, source, target)), expected)

    def test_equivalent_edit_scripts_are_marginalized(self):
        a, p = scores(1)
        source = [[1, 0, 0]]
        paths = [x for x in enumerate_scripts(a, p, source) if x[0] == (2,)]
        self.assertEqual({x[2] for x in paths}, {(SUBSTITUTE,), (DELETE, INSERT)})
        self.assertGreater(marginal_log_probability(a, p, source, (2,)), max(x[1] for x in paths))

    def test_deletion_and_insertion_have_real_separate_clocks(self):
        a, p = scores(3)
        for row in a:
            for state in row:
                state[:] = [-12.] * 5
        for t, i, action in [(0, 0, COPY), (1, 1, DELETE), (1, 2, INSERT), (2, 2, COPY)]:
            a[t][i][action] = 12.
        p[1][2] = [-12., -12., 12.]
        _, script = decode(a, p, [[1, 0, 0], [2, 1, 1], [1, 2, 2]])
        self.assertEqual([(phone, t) for _, phone, t, _ in script if phone], [(1, 0), (3, 1), (1, 2)])

    def test_adjacent_equal_outputs_are_distinct_events(self):
        a, p = scores(2)
        self.assertTrue(math.isfinite(marginal_log_probability(a, p, [], [1, 1])))
        with self.assertRaisesRegex(ValueError, "no legal"):
            marginal_log_probability(a, p, [], [1, 1, 1])

    def test_viterbi_matches_exhaustive_optimum(self):
        rng = random.Random(7)
        a, p = scores(3)
        for rows in (a, p):
            for row in rows:
                for state in row:
                    state[:] = [rng.uniform(-2, 2) for _ in state]
        source = [[1, 0, 1], [3, 2, 2]]
        result, _ = decode(a, p, source)
        self.assertAlmostEqual(result, max(x[1] for x in enumerate_scripts(a, p, source)))

    def test_source_clock_and_carried_runs(self):
        self.assertEqual(source_events([1, 1, 0, 2, 2, 0, 2]), [[1, 0, 1], [2, 3, 4], [2, 6, 6]])
        self.assertEqual(source_events([1, 1, 0, 2], 1), [[2, 3, 3]])

    def test_acoustically_timed_source_cannot_be_consumed_in_another_span(self):
        a, p = scores(2)
        source = [[1, 1, 1]]
        self.assertEqual({x[4] for x in transitions(a, p, source, 0, 0)}, {HOLD, INSERT})
        self.assertEqual({x[4] for x in transitions(a, p, source, 1, 0)}, {COPY, DELETE, SUBSTITUTE})


class CrossFitTests(unittest.TestCase):
    def test_folds_are_whole_speaker_and_stable(self):
        speakers = [f"speaker-{i}" for i in range(10)]
        folds = speaker_folds(speakers)
        self.assertEqual(folds, speaker_folds(list(reversed(speakers)) * 2))
        self.assertEqual(set(folds.values()), set(range(5)))
        with self.assertRaisesRegex(ValueError, "held-out"):
            speaker_folds(speakers, [speakers[0]])
        with self.assertRaises(ValueError):
            speaker_folds(["", "a"])

    def test_channel_excludes_recipient_fold_and_selection_speakers(self):
        folds = {f"speaker-{i}": i % 5 for i in range(10)}
        rows = [dict(speaker=s, source=[i % 3 + 1], target=[1]) for i, s in enumerate(folds)]
        channels = CrossFittedCorruptions(rows, folds)
        for s, fold in folds.items():
            self.assertNotIn(s, channels.members[fold])
            self.assertTrue(all(folds[k] not in (fold, 0) for k in channels.members[fold]))
            draw = channels.draw([1], [[1, 0, 2]], s, folds, random.Random(3))
            self.assertTrue(all(0 <= e[1] <= e[2] < 3 for e in draw))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is available on the supervisor training image")
class GradientTests(unittest.TestCase):
    def test_gradient_matches_enumerated_finite_difference(self):
        a = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
        p = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
        source, target = [[1, 0, 0]], [2]
        result = marginal_log_probability(a, p, source, target)
        result.backward()
        epsilon = 1e-5
        for variable, index in [(a, (0, 0, DELETE)), (p, (0, 0, 1)), (a, (1, 1, INSERT))]:
            original = float(variable[index].detach())
            values = []
            with torch.no_grad():
                for offset in (epsilon, -epsilon):
                    variable[index] = original + offset
                    values.append(float(marginal_log_probability(a, p, source, target)))
                variable[index] = original
            self.assertAlmostEqual(float(variable.grad[index]), (values[0] - values[1]) / (2 * epsilon), places=6)

    def test_joint_representation_is_trained_by_edit_and_support_heads(self):
        model = build_head(4, 4, hidden=8)
        features = torch.randn(1, 2, 4)
        source = torch.tensor([[1, 0, 0]])
        starts = torch.tensor([[0, 2, 2]])
        actions, phones, support, _ = model(features, source, starts)
        loss = -marginal_log_probability(actions[0], phones[0], [[1, 0, 0]], [2]) - support[0, 0, 2]
        loss.backward()
        self.assertGreater(float(model.correction.weight.grad.norm()), 0)
        self.assertGreater(float(model.embedding.weight.grad.norm()), 0)
        direct = model(features, source, starts, direct=True)
        altered = model(features, torch.zeros_like(source), starts + 7, direct=True)
        for a, b in zip(direct, altered):
            torch.testing.assert_close(a, b)


try:
    import numpy as np
    import onnx
    import onnxruntime
except ImportError:
    onnx = None


@unittest.skipIf(torch is None or onnx is None, "ONNX/Torch available on the training image")
class ExportTests(unittest.TestCase):
    def source(self, path):
        from onnx import helper as H, numpy_helper as N, TensorProto as T
        from student_edit_native import digest
        path.mkdir()
        constants = [N.from_array(np.array([0, 0], np.int64), "begins"),
            N.from_array(np.array([8, 4], np.int64), "ends"),
            N.from_array(np.array([1, 2], np.int64), "axes"),
            N.from_array(np.eye(4, dtype=np.float32) + .01, "weight"),
            N.from_array(np.array([-2, 4, -2, -2], np.float32), "bias"),
            N.from_array(np.array([1.], np.float32), "one")]
        nodes = [H.make_node("Slice", ["x", "begins", "ends", "axes"], ["slice"]),
            H.make_node("Add", ["slice", "native_count"], ["encoder_features"]),
            H.make_node("MatMul", ["encoder_features", "weight"], ["linear"], name="ctc_output/MatMul"),
            H.make_node("Add", ["linear", "bias"], ["logits"]),
            H.make_node("LogSoftmax", ["logits"], ["log_probs"], axis=-1),
            H.make_node("Add", ["native_count", "one"], ["new_native_count"])]
        graph = H.make_graph(nodes, "native", [H.make_tensor_value_info("x", T.FLOAT, [1, 45, 80]),
            H.make_tensor_value_info("native_count", T.FLOAT, [1])],
            [H.make_tensor_value_info("log_probs", T.FLOAT, [1, 8, 4]),
             H.make_tensor_value_info("new_native_count", T.FLOAT, [1])], initializer=constants,
             value_info=[H.make_tensor_value_info("encoder_features", T.FLOAT, [1, 8, 4])])
        model = H.make_model(graph, opset_imports=[H.make_opsetid("", 17)], ir_version=10)
        onnx.save(model, str(path / "model.onnx"))
        teacher = H.make_model(H.make_graph([H.make_node("Identity", ["weights"], ["y"])], "teacher", [],
            [H.make_tensor_value_info("y", T.FLOAT, [1])],
            initializer=[N.from_array(np.array([1.], np.float32), "weights")]),
            opset_imports=[H.make_opsetid("", 17)], ir_version=10)
        onnx.save_model(teacher, str(path / "teacher.onnx"), save_as_external_data=True,
                        all_tensors_to_one_file=True, location="teacher.onnx.data", size_threshold=0)
        m = dict(schema=1, id="synthetic", blank=0, tokens="abc", frameMs=40,
            inventory="synthetic", fbank=dict(mels=80, padValue=-23.0259), sampleRate=16000,
            window=45, shift=32, file="model.onnx", bytes=(path / "model.onnx").stat().st_size,
            sha256=digest(path / "model.onnx"), states=[dict(name="native_count", shape=[1], dtype="float32")])
        (path / "manifest.json").write_text(json.dumps(m))
        return m

    def test_grafted_network_parity_caches_source_runs_and_complete_assets(self):
        from student_edit_export import graft
        from student_edit_native import insertion_assets, session
        torch.set_num_threads(1)
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "native"
            self.source(original)
            native = session(original / "model.onnx")
            head = build_head(4, 4, hidden=128).eval()
            for direct in (False, True):
                out = root / ("control" if direct else "edit")
                runtime = session(graft(original, head, out, direct))
                self.assertEqual(insertion_assets(out), insertion_assets(original))
                state, previous = np.zeros(1, np.float32), np.zeros(1, np.int64)
                for call in range(3):
                    x = np.random.default_rng(call).normal(size=(1, 45, 80)).astype(np.float32)
                    valid = 3 if call == 2 else 8
                    if call == 2:
                        x[:, 3:8, 2] = 20  # a different padding-only source phone
                    native_names = [v.name for v in native.get_outputs()]
                    reference = dict(zip(native_names, native.run(None, {"x": x, "native_count": state})))
                    names = [v.name for v in runtime.get_outputs()]
                    got = dict(zip(names, runtime.run(None, {"x": x, "native_count": state,
                        "edit_source_previous": previous, "edit_valid_frames": np.array([valid], np.int64)})))
                    np.testing.assert_array_equal(got["source_log_probs"], reference["log_probs"])
                    np.testing.assert_array_equal(got["new_native_count"], reference["new_native_count"])
                    ids = reference["log_probs"][0].argmax(-1)
                    events = source_events(ids[:valid], int(previous[0]))
                    source = torch.zeros(1, 9, dtype=torch.int64)
                    starts = torch.full_like(source, 8)
                    for i, (p, start, _) in enumerate(events):
                        source[0, i], starts[0, i] = p, start
                    with torch.no_grad():
                        expected = head(torch.tensor(x[:, :8, :4] + state), source, starts, direct=direct)
                    for name, want in zip(("edit_actions", "edit_phones", "support_log_probs", "direct_log_probs"), expected):
                        np.testing.assert_allclose(got[name], want.numpy(), atol=2e-5, rtol=2e-5)
                    np.testing.assert_array_equal(got["new_edit_source_previous"], ids[valid - 1:valid])
                    state, previous = got["new_native_count"], got["new_edit_source_previous"]
            (original / "teacher.onnx.data").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                graft(original, head, root / "incomplete")

    def test_performed_transcript_timing_and_real_pause_targets(self):
        from student_edit_train import timed_target
        lp = np.full((5, 4), -12., np.float32)
        for t, p in enumerate([0, 1, 0, 1, 2]):
            lp[t, p] = 0.
        self.assertEqual(timed_target(lp, [1, 1, 2]), [[1, 1, 1], [1, 3, 3], [2, 4, 4]])
        self.assertEqual(timed_target(lp, []), [])
        with self.assertRaisesRegex(ValueError, "cannot align"):
            timed_target(lp[:2], [1, 1])

    def test_complete_recording_loss_with_partial_drain_and_real_pause_target(self):
        from student_edit_train import loss
        from student import VOCAB
        shipped = np.full((4, VOCAB), -8., np.float32)
        for t, p in enumerate([0, 1, 0, 3]):
            shipped[t, p] = 0.
        view = dict(features=np.random.default_rng(7).normal(size=(4, 4)).astype(np.float32),
            shipped=shipped, valid=3, target=np.array([1, 2], np.int64),
            timed=np.array([[1, 1, 1], [2, 2, 2]], np.int64))
        for direct in (False, True):
            head = build_head(4, VOCAB, hidden=8)
            for pause in (False, True):
                example = dict(view, target=np.array([], np.int64), timed=np.empty((0, 3), np.int64)) if pause else view
                head.zero_grad(set_to_none=True)
                objective = loss(head, example, 2, "cpu", direct)
                self.assertTrue(torch.isfinite(objective))
                objective.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None))
                self.assertGreater(float(head.correction.weight.grad.norm()), 0.)

    def test_none_enrollment_and_duration_notation(self):
        from student_edit_data import admitted_none, reference_articulations
        good = dict(id="take", speaker="learner", split="train", kind="none", scored=True,
            wav="take.wav", ayahs=["1:1"], events=[], non_errors=[], unsupported=[])
        self.assertEqual(admitted_none([good]), [good])
        for change in [dict(kind="wrong_word"), dict(non_errors=["repeat"]), dict(scored=False), dict(exclude=True)]:
            self.assertEqual(admitted_none([dict(good, **change)]), [])
        self.assertEqual(admitted_none([good, dict(good, id="held", split="test")]), [])
        self.assertEqual(reference_articulations("ااانننمممبب"), "اننممبب")

    def test_cross_corpus_holdouts_and_transcript_identity(self):
        from student_edit_data import recordings
        from student_edit_native import digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("none", "held", "speech", "pause"):
                (root / (name + ".wav")).write_bytes(name.encode())
            labels = [dict(id=name, speaker=speaker, split="train", kind="none", scored=True,
                wav=name + ".wav", ayahs=["1:1"], events=[], non_errors=[], unsupported=[])
                for name, speaker in (("none", "training"), ("held", "release"))]
            additional = [dict(speaker="transcribed", split="train", authorized=True, verified=True,
                transcriptFormat="ctc-articulations", phones="بَ" if name == "speech" else "",
                kind="speech" if name == "speech" else "silence", wav=name + ".wav",
                sha256=digest(root / (name + ".wav"))) for name in ("speech", "pause")]
            additional.append(dict(speaker="release", split="test"))
            help_path, transcript_path = root / "help.jsonl", root / "transcripts.jsonl"
            help_path.write_text("\n".join(map(json.dumps, labels)))
            transcript_path.write_text("\n".join(map(json.dumps, additional)))
            ref = {"ayahs": [dict(s=1, a=1, p="بَ")]}
            rows, excluded = recordings(help_path, transcript_path, [], ref)
            self.assertEqual(set(r["source"] for r in rows), {"help-none", "transcribed", "silence"})
            self.assertIn("release", excluded)
            self.assertNotIn("release", [r["speaker"] for r in rows])
            (root / "speech.wav").write_bytes(b"changed audio")
            with self.assertRaisesRegex(ValueError, "identity"):
                recordings(help_path, transcript_path, [], ref)

    def test_enrollment_refuses_unverified_performed_text_and_path_escape(self):
        from student_edit_data import recordings, contained
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "help.jsonl").write_text("")
            (root / "transcripts.jsonl").write_text(json.dumps(dict(speaker="learner", split="train",
                phones="ب", wav="take.wav", authorized=True, verified=False)))
            with self.assertRaisesRegex(ValueError, "performed transcript"):
                recordings(root / "help.jsonl", root / "transcripts.jsonl", [], {"ayahs": []})
            with self.assertRaisesRegex(ValueError, "leaves"):
                contained(root, "../outside.wav")


if __name__ == "__main__":
    unittest.main()

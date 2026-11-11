"""Tiny CPU unit tests; acoustic training and inference belong on Modal."""

import ast
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from student_tangent import (AcousticResponseProvider, PerturbationGeometry, append_help_none, corpus_sha256,
                             deterministic_student, help_none_rows,
                             localized_direction, paired_audio, preservation_objective,
                             response_objective, state_sha256, training_recipe)


def recipe():
    return dict(schema=1, split="train", training_speakers=["train"], validation_speakers=["valid"],
                excluded_speakers=["excluded"], training_keys=["a:1", "help:clean"], validation_keys=["a:2", "help:valid"],
                speakers_by_key={"a:1": "train", "a:2": "valid", "help:clean": "train", "help:valid": "valid"},
                real_pause_examples=2,
                geometry_validation_examples=2, selection="minimum-validation-objective",
                teacher_sha256="a" * 64, student_sha256="b" * 64, training_data_sha256="c" * 64,
                sample_rate=16000, frame_ms=40,
                geometry=dict(n_fft=32, hop=8, window=24, time_frames=3, frequency_bins=4, epsilon=.05))


def response(bias, slope, eps=.05, curvature=0):
    return [(bias + sign * eps * slope + sign * sign * eps * eps * curvature).log_softmax(-1)
            for sign in (0, 1, -1)]


class GeometryTests(unittest.TestCase):
    def test_no_practice_geometry_or_overlapping_speakers(self):
        record = recipe()
        training_recipe(record)
        for update in (dict(split="practice"), dict(validation_speakers=["train"]),
                       dict(training_speakers=["excluded"]), dict(geometry_validation_examples=0),
                       dict(validation_keys=["a:1"]), dict(teacher_sha256=""), dict(frame_ms=20),
                       dict(real_pause_examples=0), dict(speakers_by_key={})):
            with self.subTest(update=update), self.assertRaises(ValueError):
                training_recipe({**record, **update})

    def test_reject_invalid_geometry(self):
        for update in (dict(epsilon=0), dict(epsilon=float("nan")), dict(frequency_bins=100),
                       dict(hop=24), dict(time_frames=1.5)):
            with self.subTest(update=update), self.assertRaises(ValueError):
                PerturbationGeometry(**{**recipe()["geometry"], **update})

    def test_audio_signs_preserve_original_and_length(self):
        wave = torch.sin(torch.arange(256, dtype=torch.float64) * .1)
        original = wave.clone()
        geometry = training_recipe(recipe())
        base, plus, minus = paired_audio(wave, geometry, 22)
        self.assertIs(base, wave)
        torch.testing.assert_close(wave, original)
        torch.testing.assert_close((plus + minus) / 2, wave)
        self.assertEqual(plus.shape, wave.shape)
        self.assertGreater((plus - wave).abs().max(), 0)

    def test_patch_is_local_and_uses_a_private_deterministic_rng(self):
        wave = torch.sin(torch.arange(2048, dtype=torch.float64) * .17)
        geometry = training_recipe(recipe())
        state = torch.random.get_rng_state()
        a = localized_direction(wave, geometry, 1)
        torch.testing.assert_close(torch.random.get_rng_state(), state)
        torch.testing.assert_close(a, localized_direction(wave, geometry, 1))
        self.assertFalse(torch.equal(a, localized_direction(wave, geometry, 2)))
        # The inverse STFT adds at most one analysis window at either edge.
        active = (a.abs() > 1e-10).nonzero().flatten()
        self.assertLessEqual(int(active[-1] - active[0] + 1),
                             geometry.time_frames * geometry.hop + 2 * geometry.n_fft)

    def test_real_silence_remains_silent(self):
        geometry = training_recipe(recipe())
        for wave in paired_audio(torch.zeros(256), geometry, 2):
            self.assertEqual(int(torch.count_nonzero(wave)), 0)

    def test_short_or_nonfinite_audio_is_not_silently_replaced(self):
        for wave in (torch.zeros(2), torch.full((256,), float("nan")), torch.zeros(2, 256)):
            with self.assertRaises(ValueError):
                paired_audio(wave, training_recipe(recipe()), 2)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.bias = torch.tensor([[[2., 0., -1.], [-1., 0., 1.]]], dtype=torch.float64)
        self.slope = torch.tensor([[[1., 2., -1.], [0., 3., 1.]]], dtype=torch.float64)
        self.valid = torch.ones(1, 2, dtype=torch.bool)

    def test_static_phone_priors_do_not_contribute_to_the_tangent(self):
        s = response(self.bias, self.slope)
        t = response(self.bias + torch.tensor([0., 9., -7.]), self.slope)
        tangent = response_objective(s, t, self.valid, .05, "tangent")
        self.assertLess(float(tangent), 1e-24)
        # Ordinary distillation still sees those different priors.
        self.assertGreater(float(response_objective(s, t, self.valid, .05, "ordinary")), 1.)

    def test_blank_relative_derivative_has_the_analytic_affine_value(self):
        s = response(self.bias, self.slope)
        t = response(self.bias, torch.zeros_like(self.slope))
        expected = torch.tensor([1., -2., 3., 1.], dtype=torch.float64).square().mean()
        torch.testing.assert_close(response_objective(s, t, self.valid, .05, "tangent"), expected)

    def test_even_response_curvature_cancels(self):
        s = response(self.bias, self.slope, curvature=torch.tensor([0., 7., -3.]))
        t = response(self.bias, self.slope)
        self.assertLess(float(response_objective(s, t, self.valid, .05, "tangent")), 1e-24)

    def test_mismatched_sensitivity_has_student_gradients_and_no_teacher_gradients(self):
        slope = torch.zeros_like(self.slope, requires_grad=True)
        teacher_slope = self.slope.clone().requires_grad_()
        loss = response_objective(response(self.bias, slope), response(self.bias, teacher_slope),
                                  self.valid, .05, "tangent")
        loss.backward()
        self.assertGreater(float(slope.grad.abs().sum()), 0)
        self.assertIsNone(teacher_slope.grad)
        improved = response_objective(response(self.bias, slope - .1 * slope.grad),
                                      response(self.bias, teacher_slope), self.valid, .05, "tangent")
        self.assertLess(float(improved.detach()), float(loss.detach()))

    def test_symmetric_difference_converges_for_a_nonlinear_response(self):
        want = torch.tensor(4., dtype=torch.float64)  # d(x^2)/dx at x=2
        errors = []
        for eps in (.2, .1, .05):
            s = [torch.tensor([[[0., (2 + sign * eps) ** 3]]], dtype=torch.float64).log_softmax(-1)
                 for sign in (0, 1, -1)]
            t = response(torch.zeros(1, 1, 2, dtype=torch.float64),
                         torch.tensor([[[0., 3 * want]]], dtype=torch.float64), eps)
            errors.append(float(response_objective(s, t, torch.ones(1, 1, dtype=torch.bool), eps, "tangent")))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_ordinary_control_matches_full_distribution_cross_entropy_on_both_signs(self):
        s, t = response(self.bias, self.slope), response(-self.bias, self.slope)
        want = sum(-(q.exp() * p).sum(-1).mean() for p, q in zip(s[1:], t[1:])) / 2
        torch.testing.assert_close(response_objective(s, t, self.valid, .05, "ordinary"), want)

    def test_padding_including_infinite_sentinels_has_no_loss_or_gradient(self):
        s, t = response(self.bias, self.slope), response(self.bias, 2 * self.slope)
        padded_s = [torch.cat((x, torch.full((1, 1, 3), -torch.inf)), 1).detach().requires_grad_() for x in s]
        padded_t = [torch.cat((x, torch.full((1, 1, 3), -torch.inf)), 1) for x in t]
        valid = torch.tensor([[True, True, False]])
        for mode in ("ordinary", "tangent"):
            got = response_objective(padded_s, padded_t, valid, .05, mode)
            want = response_objective(s, t, self.valid, .05, mode)
            torch.testing.assert_close(got, want)
            grads = torch.autograd.grad(got, padded_s, allow_unused=True)
            for grad in grads:
                if grad is not None:
                    self.assertEqual(float(grad[:, 2].abs().sum()), 0)

    def test_mask_and_clock_errors_fail_instead_of_clamping(self):
        s = response(self.bias, self.slope)
        for target, valid, eps, mode in (([x[:, :1] for x in s], self.valid, .05, "tangent"),
                                         (s, self.valid & False, .05, "tangent"),
                                         (s, self.valid, 0, "tangent"),
                                         (s, self.valid, .05, "unknown")):
            with self.assertRaises(ValueError):
                response_objective(s, target, valid, eps, mode)

    def test_preservation_keeps_real_blank_rows_and_detaches_anchor(self):
        student = self.bias.log_softmax(-1).detach().requires_grad_()
        anchor = (-self.bias).log_softmax(-1).detach().requires_grad_()
        loss = preservation_objective(student, anchor, self.valid)
        loss.backward()
        self.assertGreater(float(student.grad.abs().sum()), 0)
        self.assertIsNone(anchor.grad)
        self.assertAlmostEqual(float(preservation_objective(student, student, self.valid).detach()), 0.)


class HelpIngestionTests(unittest.TestCase):
    def label(self):
        return dict(id="clean", speaker="train", kind="none", split="train", scored=True,
                    events=[], non_errors=[], unsupported=[], ayahs=["2:7"], wav="clean.wav")

    def test_empty_events_does_not_admit_repeat_or_unsupported_or_held_out_takes(self):
        label = self.label()
        self.assertEqual(help_none_rows([label], recipe(), "training"), [("help:clean", label)])
        for update in (dict(kind="repeat"), dict(non_errors=[{"kind": "repeat"}]),
                       dict(unsupported=[{"rule": "unknown"}]), dict(events=[{}]), dict(scored=False),
                       dict(split="eval"), dict(speaker="excluded"), dict(excluded="audit"), dict(ayahs=[])):
            with self.subTest(update=update), self.assertRaises(ValueError):
                help_none_rows([{**label, **update}], recipe(), "training")
        with self.assertRaises(ValueError):
            help_none_rows([], recipe(), "training")
        with self.assertRaises(ValueError):
            help_none_rows([label, label], recipe(), "training")

    def test_append_preserves_full_pcm_and_has_exact_static_teacher_targets(self):
        import numpy as np
        import soundfile as sf

        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=source / ".scratch") as directory, \
                patch.object(sys, "path", [str(source / "tools"), *sys.path]):
            from student import TOKENS, VOCAB

            root = Path(directory)
            label = self.label()
            (root / "labels.jsonl").write_text(json.dumps(label) + "\n")
            wave = np.sin(np.arange(256) * .1).astype(np.float32)
            wave[80:160] = 0  # The original real-pause-shaped region is retained.
            sf.write(root / "clean.wav", wave, 16000, subtype="FLOAT")
            clips = dict(key=["a:1"], ref=[TOKENS[0]], hyp=[TOKENS[0]],
                         n_samples=np.array([256]), n_frames=np.array([4]),
                         flac=np.array([7, 8], dtype=np.uint8), flac_off=np.array([0, 2]),
                         ids=np.zeros(20, dtype=np.uint8), probs=np.full(20, .2, dtype=np.float16),
                         top_off=np.array([0, 20]))
            seen = []
            lp = torch.arange(VOCAB * 4, dtype=torch.float32).reshape(1, 4, VOCAB).log_softmax(-1)

            def teacher(pcm):
                seen.append(pcm.clone())
                return lp

            index = dict(ayahs=[dict(s=2, a=7, p=TOKENS[:2])])
            merged = append_help_none(clips, root, index, recipe(), "training", teacher)
            torch.testing.assert_close(seen[0], torch.from_numpy(wave), rtol=0, atol=0)
            self.assertEqual(merged["key"], ["a:1", "help:clean"])
            self.assertEqual(merged["ref"][-1], TOKENS[:2])
            self.assertEqual(int(merged["n_samples"][-1]), len(wave))
            import io
            restored, sr = sf.read(io.BytesIO(merged["flac"][2:].tobytes()), dtype="float32")
            np.testing.assert_array_equal(restored, wave)
            self.assertEqual(sr, 16000)
            top_p, top_i = lp[0].exp().topk(5, -1)
            np.testing.assert_array_equal(merged["ids"][20:].reshape(4, 5), top_i.numpy())
            np.testing.assert_array_equal(merged["probs"][20:].reshape(4, 5), top_p.numpy().astype(np.float16))
            original_hash = corpus_sha256(merged)
            merged["ref"][-1] = TOKENS[2]
            self.assertNotEqual(corpus_sha256(merged), original_hash)
            merged["ref"][-1] = TOKENS[:2]
            merged["flac"][-1] ^= 1
            self.assertNotEqual(corpus_sha256(merged), original_hash)
            # Source arrays and offsets were not mutated by appending.
            np.testing.assert_array_equal(clips["flac"], np.array([7, 8], dtype=np.uint8))

    def test_help_audio_cannot_escape_authorized_root(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=source / ".scratch") as directory, \
                patch.object(sys, "path", [str(source / "tools"), *sys.path]):
            root = Path(directory)
            (root / "labels.jsonl").write_text(json.dumps({**self.label(), "wav": "../unavailable.wav"}))
            import numpy as np
            clips = dict(key=[], ref=[], hyp=[], n_samples=[], n_frames=[], flac=np.array([], dtype=np.uint8),
                         ids=np.array([], dtype=np.uint8), probs=np.array([], dtype=np.float16),
                         flac_off=np.array([0]), top_off=np.array([0]))
            with self.assertRaisesRegex(ValueError, "authorized corpus"):
                append_help_none(clips, root, dict(ayahs=[]), recipe(), "training", lambda pcm: None)


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.5)
        self.output = torch.nn.Linear(2, 3)

    def forward(self, features, lengths):
        return self.output(self.dropout(features)).log_softmax(-1), lengths


class ProviderTests(unittest.TestCase):
    def test_teacher_identity_changes_with_weights(self):
        teacher = TinyStudent()
        fingerprint = state_sha256(teacher)
        self.assertEqual(fingerprint, state_sha256(copy.deepcopy(teacher)))
        with torch.no_grad():
            teacher.output.weight[0, 0] += .01
        self.assertNotEqual(fingerprint, state_sha256(teacher))

    def test_export_loader_uses_training_selected_weights_and_rejects_missing_selection(self):
        source = Path(__file__).with_name("student_train.py")
        fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "load_student")
        namespace = {"build_model": lambda size: TinyStudent()}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
        last, selected = TinyStudent(), TinyStudent()
        cfg = dict(size="unit", acoustic_response_mode="tangent")
        checkpoints = {"/student/runs/unit-run/ckpt.pt": dict(model=last.state_dict(), cfg=cfg, response_recipe=recipe()),
                       "/student/runs/unit-run/selected.pt": dict(model=selected.state_dict(), cfg=cfg, response_recipe=recipe())}
        with patch("torch.load", side_effect=lambda path, **kw: checkpoints[path]):
            model, _ = namespace["load_student"]("unit-run")
        for got, want in zip(model.parameters(), selected.parameters()):
            torch.testing.assert_close(got, want)
        checkpoints.pop("/student/runs/unit-run/selected.pt")
        with patch("torch.load", side_effect=lambda path, **kw: checkpoints[path]), self.assertRaises(KeyError):
            namespace["load_student"]("unit-run")

    def test_training_flags_restore_after_an_exception(self):
        model = TinyStudent().train()
        model.dropout.eval()
        flags = [m.training for m in model.modules()]
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with deterministic_student(model):
                self.assertFalse(any(m.training for m in model.modules()))
                raise RuntimeError("failed")
        self.assertEqual([m.training for m in model.modules()], flags)

    def test_streaming_context_restores_even_when_configuration_raises(self):
        model = TinyStudent().train()
        model.encoder = torch.nn.Identity()
        model.encoder.chunk_size, model.encoder.left_context_frames = (16, 32), (64, 128)

        def configure(chunk, left):
            model.encoder.chunk_size, model.encoder.left_context_frames = (chunk,), (left,)
            raise RuntimeError("context setup failed")

        model.streaming_eval = configure
        with self.assertRaisesRegex(RuntimeError, "context setup failed"):
            with deterministic_student(model):
                self.fail("Failed configuration must not enter the response pass")
        self.assertEqual(model.encoder.chunk_size, (16, 32))
        self.assertEqual(model.encoder.left_context_frames, (64, 128))
        self.assertTrue(all(m.training for m in model.modules()))

    def test_shared_prefix_uses_original_rows_without_repeating_the_teacher_edge(self):
        model = TinyStudent().eval()
        seen, targets = [], []

        def features(wave):
            return torch.stack((wave[:3], wave[3:6]), -1)

        def teacher(wave):
            seen.append(wave.clone())
            target = torch.stack((wave[:5], wave[5:10], -wave[:5]), -1)[None].log_softmax(-1)
            targets.append(target)
            return target

        provider = AcousticResponseProvider(teacher, features, training_recipe(recipe()), 9)
        actual = provider.loss(model, [torch.sin(torch.arange(256) * .1)], ["training-item"], 7,
                               "tangent", torch.device("cpu"))
        predicted = [model(features(wave)[None], torch.tensor([3]))[0] for wave in seen]
        expected = response_objective(predicted, [x[:, :3] for x in targets],
                                      torch.ones(1, 3, dtype=torch.bool), .05, "tangent")
        torch.testing.assert_close(actual, expected)
        self.assertEqual(provider.edge_rows, 2)

    def test_matched_provider_consumes_same_audio_and_retains_student_gradients(self):
        model = TinyStudent().train()
        seen = []

        def features(wave):
            return torch.stack((wave[:3], wave[3:6]), -1)

        def teacher(wave):
            seen.append(wave.clone())
            f = features(wave)
            return torch.cat((f, -f[:, :1]), -1)[None].log_softmax(-1)

        provider = AcousticResponseProvider(teacher, features, training_recipe(recipe()), 9)
        wave = torch.sin(torch.arange(256) * .1)
        ordinary = provider.loss(model, [wave], ["train-item"], 7, "ordinary", torch.device("cpu"))
        first = [x.clone() for x in seen]
        seen.clear()
        tangent = provider.loss(model, [wave], ["train-item"], 7, "tangent", torch.device("cpu"))
        for a, b in zip(seen, first):
            torch.testing.assert_close(a, b)
        self.assertEqual(len(seen), 3)
        torch.testing.assert_close((seen[1] + seen[2]) / 2, seen[0])
        (ordinary + tangent).backward()
        self.assertIsNotNone(model.output.weight.grad)
        self.assertTrue(model.training)

    def test_actual_training_dispatch_retains_ctc_static_kd_and_preservation(self):
        source = Path(__file__).with_name("student_train.py")
        fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "losses")
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
        model = TinyStudent().eval()
        anchor = copy.deepcopy(model).requires_grad_(False)
        batch = dict(feats=torch.tensor([[[1., 0.], [0., 1.]]]), feat_lens=torch.tensor([2]),
                     targets=torch.tensor([1]), target_lens=torch.tensor([1]), n_teacher=torch.tensor([2]),
                     top_ids=torch.tensor([[[0, 1, 2, 0, 0]] * 2]),
                     top_p=torch.tensor([[[.3, .5, .2, 0., 0.]] * 2]), waves=[torch.zeros(256)], keys=["a:1"])

        class Provider:
            def loss(self, model, waves, keys, step, mode, device):
                # Nonzero and differentiable, so total-loss dispatch is tested.
                return model.output.weight.square().mean() * (2 if mode == "tangent" else 1)

        cfg = dict(ctc_weight=1., kd_weight=.7, kd_delay=0, target="reference",
                   acoustic_response_weight=.2, acoustic_preservation_weight=1.)
        with patch.object(sys, "path", [str(source.parents[1] / "tools"), *sys.path]), \
                patch("torch.autocast", side_effect=lambda *a, **k: nullcontext()):
            plain = namespace["losses"](model, batch, cfg, torch.device("cpu"))
            outputs = [namespace["losses"](model, batch, {**cfg, "acoustic_response_mode": mode},
                                           torch.device("cpu"), Provider(), anchor, 2)
                       for mode in ("ordinary", "tangent")]
            with self.assertRaisesRegex(ValueError, "fresh teacher"):
                namespace["losses"](model, batch, {**cfg, "acoustic_response_mode": "tangent"}, torch.device("cpu"))
        for result in outputs:
            for key in ("ctc", "kd", "frames"):
                torch.testing.assert_close(result[key], plain[key])
            torch.testing.assert_close(result["loss"], plain["loss"] + .2 * result["response"])
            self.assertEqual(float(result["preservation"].detach()), 0.)
        self.assertTrue(all(p.grad is None for p in anchor.parameters()))


if __name__ == "__main__":
    unittest.main()

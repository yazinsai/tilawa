"""python3 -m unittest discover -s modal -p test_student_reallocation.py -v"""

import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_reallocation import (admitted_none, channel_statistics, dormant_channels,
    freeze_stem, preservation, recycle, speaker_partition, transcript_path)
from student_reallocation_train import InvocationBoundary, boundary, contained, refit_state, restore_refit


class ChannelAuditTests(unittest.TestCase):
    def test_negative_preactivations_are_not_dead_channels(self):
        stats = channel_statistics(np.array([[-6., -2., -1.]]), np.eye(3))
        np.testing.assert_allclose(stats["activity"], [.4450454, .1274757, .0517153], atol=1e-7)
        np.testing.assert_array_equal(stats["activity"], stats["contribution"])

    def test_outgoing_norm_matters_even_when_activation_is_small(self):
        a = np.array([[1e-4, 1.]])
        # Direct observations let a quiet channel have a large outgoing effect.
        r = dict(activity=a[0], contribution=np.array([1., 1.]), output_rms=2.)
        self.assertEqual(dormant_channels([r, r]), [])

    def test_intermittently_active_channel_is_never_recycled(self):
        quiet = dict(activity=np.array([1e-8, 1.]), contribution=np.array([1e-8, 1.]), output_rms=1.)
        active = dict(activity=np.array([1., 1.]), contribution=np.array([1., 1.]), output_rms=1.)
        self.assertEqual(dormant_channels([quiet, quiet]), [0])
        self.assertEqual(dormant_channels([quiet, active]), [])

    def test_statistics_reject_empty_padding_and_nonfinite_frames(self):
        for x in (np.empty((0, 2)), np.array([[np.nan, 0.]])):
            with self.assertRaises(ValueError):
                channel_statistics(x, np.eye(2))
        with self.assertRaises(ValueError):
            dormant_channels([dict(activity=np.ones(2))])

    def test_probability_parity_rejects_small_top1_change(self):
        a = np.log([[.5000001, .4999999]])
        b = np.log([[.4999999, .5000001]])
        self.assertFalse(preservation(a, b)["passes"])
        self.assertLess(preservation(a, b)["max_probability_drift"], 1e-6)
        self.assertTrue(preservation(a, a.copy())["passes"])

    def test_probability_parity_rejects_clock_changes(self):
        with self.assertRaises(ValueError):
            preservation(np.zeros((2, 3)), np.zeros((3, 3)))


class EnrollmentTests(unittest.TestCase):
    def test_selection_is_independent_of_record_order_and_speaker_counts(self):
        speakers = ["speaker-a", "speaker-b", "speaker-c", "speaker-d", "speaker-e"]
        result = speaker_partition(speakers)
        self.assertEqual(result, speaker_partition(speakers[::-1] + ["speaker-a"] * 20))
        self.assertEqual(set(result.values()), {"fit", "selection"})

    def test_missing_speaker_provenance_is_rejected(self):
        with self.assertRaises(ValueError):
            speaker_partition(["", "a", "b"])

    @staticmethod
    def clean(speaker):
        return dict(speaker=speaker, kind="none", split="train", scored=True,
                    events=[], non_errors=[], unsupported=[], ayahs=["prompt"], wav="audio.wav")

    def test_held_out_speaker_is_excluded_before_any_clean_take_is_admitted(self):
        rows = [self.clean("a"), dict(self.clean("a"), split="eval"), self.clean("b")]
        self.assertEqual([r["speaker"] for r in admitted_none(rows, [])], ["b"])
        self.assertEqual(admitted_none(rows, ["b"]), [])

    def test_mistake_repeat_and_unaudited_none_are_not_transcripts(self):
        r = self.clean("a")
        for changed in (dict(kind="substitution"), dict(events=[{}]), dict(non_errors=[{}]),
                        dict(scored=False), dict(excluded="audit failed"), dict(unsupported=[{}])):
            self.assertEqual(admitted_none([dict(r, **changed)], []), [])
        self.assertEqual(admitted_none([r], []), [r])


class AlignmentTests(unittest.TestCase):
    @staticmethod
    def collapse(path):
        return [v for i, v in enumerate(path) if v and (i == 0 or v != path[i - 1])]

    def test_viterbi_matches_enumerated_ctc_paths(self):
        lp = np.log([[.2, .5, .3], [.6, .3, .1], [.1, .2, .7], [.5, .1, .4]])
        target = [1, 2]
        possible = [p for p in itertools.product(range(3), repeat=4) if self.collapse(p) == target]
        best = max(sum(lp[t, p[t]] for t in range(4)) for p in possible)
        path = transcript_path(lp, target)
        self.assertEqual(self.collapse(path.tolist()), target)
        self.assertAlmostEqual(sum(lp[t, path[t]] for t in range(4)), best)

    def test_repeated_articulation_requires_a_blank(self):
        lp = np.log([[.1, .9], [.8, .2], [.1, .9]])
        np.testing.assert_array_equal(transcript_path(lp, [1, 1]), [1, 0, 1])
        with self.assertRaises(ValueError):
            transcript_path(lp[:2], [1, 1])

    def test_empty_clock_is_rejected(self):
        with self.assertRaises(ValueError):
            transcript_path(np.empty((0, 2)), [1])


class RunnerBoundaryTests(unittest.TestCase):
    def test_expired_invocation_stops_at_the_calling_durable_boundary(self):
        boundary(float("inf"))
        with self.assertRaises(InvocationBoundary):
            boundary(0.)

    def test_audio_cannot_escape_the_enrolled_corpus(self):
        root = Path(__file__).resolve().parent
        self.assertEqual(contained(root, Path(__file__).name), Path(__file__).resolve())
        with self.assertRaises(ValueError):
            contained(root, "../AGENTS.md")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch validation must run in the supervisor train image")
class RecyclingTests(unittest.TestCase):
    def module(self):
        import torch
        torch.manual_seed(3)
        class Feedforward(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = torch.nn.Linear(3, 4)
                self.out_proj = torch.nn.Linear(4, 3)
            def forward(self, x):
                z = self.in_proj(x)
                return self.out_proj(torch.nn.functional.softplus(z - 4) - .08 * z - .035)
        return Feedforward()

    def test_zero_contribution_channel_recycles_without_changing_function(self):
        import torch
        m = self.module()
        with torch.no_grad():
            m.out_proj.weight[:, 1] = 0
        x = torch.randn(20, 3)
        before = m(x).detach()
        shape = {k: tuple(v.shape) for k, v in m.state_dict().items()}
        incoming = m.in_proj.weight.detach().clone()
        recycle(m, [1], np.array([[1., 2., -3.]]))
        torch.testing.assert_close(before, m(x), rtol=0, atol=0)
        self.assertEqual(shape, {k: tuple(v.shape) for k, v in m.state_dict().items()})
        self.assertFalse(torch.equal(incoming[1], m.in_proj.weight[1]))
        torch.testing.assert_close(incoming[[0, 2, 3]], m.in_proj.weight[[0, 2, 3]])

    def test_zeroing_nonzero_outgoing_weight_does_not_preserve_function(self):
        import torch
        m = self.module()
        x = torch.randn(20, 3)
        before = m(x).detach().log_softmax(-1).numpy()
        recycle(m, [1], np.array([[1., 2., 3.]]))
        after = m(x).detach().log_softmax(-1).numpy()
        self.assertFalse(preservation(before, after)["passes"])

    def test_recycled_channel_can_learn_outgoing_weights(self):
        import torch
        m = self.module()
        recycle(m, [1], np.array([[1., 2., 3.]]))
        target = torch.ones(1, 3)
        (m(torch.tensor([[1., 2., 3.]])) - target).square().sum().backward()
        self.assertGreater(float(m.out_proj.weight.grad[:, 1].abs().sum()), 0)

    def test_refit_keeps_stem_and_lower_stack_weights_fixed(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_embed = torch.nn.Linear(3, 3)
                self.encoder = torch.nn.Module()
                self.encoder.encoders = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(3)])
                self.encoder.downsample_output = torch.nn.Linear(3, 3)
                self.ctc_output = torch.nn.Linear(3, 3)
            def forward(self, x):
                x = self.encoder_embed(x)
                for stack in self.encoder.encoders:
                    x = stack(x)
                return self.ctc_output(self.encoder.downsample_output(x))
        model = Model()
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        freeze_stem(model)
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        model(torch.ones(1, 3)).square().sum().backward()
        opt.step()
        for name, parameter in model.named_parameters():
            if name.startswith(("encoder_embed.", "encoder.encoders.0.")):
                torch.testing.assert_close(before[name], parameter, rtol=0, atol=0)
                self.assertIsNone(parameter.grad)
        self.assertFalse(torch.equal(before["ctc_output.weight"], model.ctc_output.weight))
        saved = refit_state(model)
        self.assertNotIn("encoder_embed.weight", saved)
        restore_refit(model, saved)
        with self.assertRaises(ValueError):
            restore_refit(model, dict(saved, unexpected=torch.zeros(1)))


if __name__ == "__main__":
    unittest.main()

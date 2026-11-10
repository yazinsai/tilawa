"""python -m unittest discover -s modal -p test_student_temporal.py

Structural tests need only Python. Loss, exhaustive-partition and gradient
tests need torch and must also run on the supplied training image.
"""

import importlib.util
import itertools
import math
import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from student_temporal import DelayBounds, calibrated_bounds, correspondence_rows, ctc_occurrences, derive_calibration, input_deadline_samples, matched_emissions, restrict_teacher_prefix, temporal_target_loss, transport_partition

GEOMETRY = dict(window=45, shift=32, feature_frame_ms=10, analysis_frame_ms=25, output_frame_ms=40)


class Array:
    """Only the tolist protocol is needed by the detached CTC alignment."""

    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return self.rows


class OccurrenceTests(unittest.TestCase):
    def test_deadline_includes_analysis_window_and_follows_deployed_chunks(self):
        self.assertEqual(input_deadline_samples(0, GEOMETRY, 16000), 7440)
        self.assertEqual(input_deadline_samples(7, GEOMETRY, 16000), 7440)
        self.assertEqual(input_deadline_samples(8, GEOMETRY, 16000), 12560)

    def test_restricted_teacher_recomputes_only_available_pcm_on_the_original_grid(self):
        seen = []

        def infer(wave):
            seen.append(len(wave))
            return list(range(20))

        parts = restrict_teacher_prefix(infer, [0] * 20000, 17, GEOMETRY, 16000)
        self.assertEqual(seen, [7440, 12560, 17680])
        self.assertEqual(list(itertools.chain.from_iterable(parts)), list(range(17)))

    def test_disappearing_phone_is_not_reported_as_delayed(self):
        self.assertEqual(matched_emissions([0, 1, 1, 0, 0], [1, 2]), [(1, 2), None])

    def test_greedy_repeat_occurrences_keep_distinct_run_times(self):
        self.assertEqual(matched_emissions([1, 1, 0, 1], [1, 1]), [(0, 1), (3, 3)])

    def test_bounds_come_from_training_delays_and_restricted_deadlines(self):
        rows = [dict(split="train", transcript=[1, 2], teacher=[0, 1, 0, 2, 0],
                     student=[0, 0, 1, 0, 2, 0], restricted_teacher=[0, 0, 0, 1, 0, 2, 0])]
        hashes = dict(teacher_sha256="a" * 64, student_sha256="b" * 64, training_data_sha256="c" * 64)
        record = derive_calibration(rows, geometry=GEOMETRY, **hashes)
        self.assertEqual((record["ordinary_lower"], record["ordinary_upper"]), (1, 1))
        self.assertEqual(calibrated_bounds(record), DelayBounds(0, 2))
        self.assertEqual((record["occurrences"], record["restricted_occurrences"]), (2, 2))
        with self.assertRaisesRegex(ValueError, "cannot consume"):
            derive_calibration([{**rows[0], "split": "practice"}], geometry=GEOMETRY, **hashes)

    def test_equal_phones_keep_separate_occurrences_across_blank(self):
        lp = Array([[math.log(p) for p in row] for row in [[.1, .9], [.9, .1], [.1, .9]]])
        self.assertEqual(ctc_occurrences(lp, [1, 1]), [0, 1, 1])

    def test_repeat_without_enough_blank_time_is_not_a_complete_transcript(self):
        with self.assertRaisesRegex(ValueError, "complete CTC"):
            ctc_occurrences(Array([[math.log(.1), math.log(.9)]] * 2), [1, 1])

    def test_missing_greedy_phone_still_has_a_transcript_occurrence(self):
        lp = Array([[math.log(p) for p in row] for row in [[.1, .8, .1], [.8, .1, .1], [.8, .1, .1]]])
        self.assertEqual(ctc_occurrences(lp, [1, 2])[-1], 1)

    def test_occurrences_cannot_cross_even_when_the_symbols_are_equal(self):
        rows = correspondence_rows([0, 0, 1, 1], [0, 1, 1, 1], DelayBounds(-3, 3))
        self.assertEqual(rows, [[0], [0], [1, 2, 3], [1, 2, 3]])

    def test_out_of_range_rows_are_not_clamped_or_removed(self):
        rows = correspondence_rows([0, 0], [0, 0], DelayBounds(1, 1))
        self.assertEqual(rows, [[1], []])

    def test_invalid_occurrence_sequences_fail(self):
        for owners in ([0, 2], [1, 1], [0, 1, 0]):
            with self.assertRaises(ValueError):
                correspondence_rows(owners, [0, 1], DelayBounds(-2, 2))

    def test_guessed_or_nontraining_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            calibrated_bounds({"schema": 1, "split": "practice", "lower": -1, "upper": 1})
        with self.assertRaises(ValueError):
            calibrated_bounds({"schema": 1, "split": "train", "lower": -1, "upper": 1})


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is available on the supplied training image")
class TransportTests(unittest.TestCase):
    def test_training_dispatch_keeps_ctc_and_kd_weight_equal_to_the_matched_control(self):
        import torch

        # Execute the actual production loss function without importing its
        # module-level Modal image/volume declarations on a CPU unit-test host.
        source = Path(__file__).with_name("student_train.py")
        function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "losses")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        lp = torch.tensor([[[.1, .8, .1], [.8, .1, .1], [.1, .1, .8]],
                           [[.1, .8, .1], [.8, .1, .1], [.1, .1, .8]]]).log().requires_grad_()
        teacher = torch.tensor([[[.1, .7, .2], [.8, .1, .1], [.1, .2, .7]],
                                [[.1, .7, .2], [.8, .1, .1], [0, 0, 0]]])
        b = dict(feats=torch.zeros(2, 3, 80), feat_lens=torch.tensor([3, 2]),
                 targets=torch.tensor([1, 2, 1]), target_lens=torch.tensor([2, 1]),
                 n_teacher=torch.tensor([3, 2]), top_ids=torch.tensor([[[0, 1, 2]] * 3] * 2), top_p=teacher)
        model = lambda feats, lengths: (lp, lengths)
        record = dict(schema=1, split="train", occurrences=3, restricted_occurrences=3,
                      frame_ms=40, receptive_field=GEOMETRY, lower=0, upper=0,
                      teacher_sha256="a" * 64, student_sha256="b" * 64, training_data_sha256="c" * 64)
        cfg = dict(ctc_weight=1., kd_weight=.7, target="reference", kd_delay=0, temporal_calibration_record=record)
        root = str(source.parents[1] / "tools")
        with patch.object(sys, "path", [root, *sys.path]), patch("torch.autocast", side_effect=lambda *a, **k: __import__("contextlib").nullcontext()):
            # The ordinary training store is top-5; add zero-mass columns to
            # exercise its unchanged fixed-width gather too.
            b["top_ids"] = torch.cat((b["top_ids"], torch.zeros(2, 3, 2, dtype=torch.long)), -1)
            b["top_p"] = torch.cat((b["top_p"], torch.zeros(2, 3, 2)), -1)
            framewise = namespace["losses"](model, b, {**cfg, "kd_mode": "framewise"}, torch.device("cpu"))
            temporal = namespace["losses"](model, b, {**cfg, "kd_mode": "temporal"}, torch.device("cpu"))
        for key in ("ctc", "kd", "loss", "frames"):
            torch.testing.assert_close(temporal[key], framewise[key])
        want, = torch.autograd.grad(framewise["loss"], lp, retain_graph=True)
        got, = torch.autograd.grad(temporal["loss"], lp)
        torch.testing.assert_close(got, want)

    def test_forward_sum_matches_exhaustive_monotone_correspondences(self):
        import torch

        rows = [[0, 1], [0, 1, 2], [1, 2]]
        costs = [torch.tensor(c, dtype=torch.float64, requires_grad=True) for c in ([.2, .5], [.7, .1, .3], [.4, .8])]
        paths = [p for p in itertools.product(*[range(len(r)) for r in rows])
                 if all(rows[t][p[t]] <= rows[t + 1][p[t + 1]] for t in range(len(rows) - 1))]
        total = torch.stack([sum(costs[t][j] for t, j in enumerate(p)) for p in paths])
        lp, prior = transport_partition(costs, rows)
        torch.testing.assert_close(lp, torch.logsumexp(-total, 0))
        self.assertAlmostEqual(prior, math.log(len(paths)))

    def test_gradient_transports_each_teacher_rows_mass_exactly_once(self):
        import torch

        costs = [torch.tensor([.2, .7], dtype=torch.float64, requires_grad=True) for _ in range(3)]
        lp, prior = transport_partition(costs, [[0, 1]] * 3)
        (prior - lp).backward()
        for row in costs:
            self.assertTrue(torch.isfinite(row.grad).all())
            self.assertTrue((row.grad >= 0).all())
            torch.testing.assert_close(row.grad.sum(), torch.tensor(1., dtype=torch.float64))

    def test_constant_cost_has_no_advantage_from_more_correspondences(self):
        import torch

        for rows in ([[0], [1], [2]], [[0, 1, 2]] * 3):
            lp, prior = transport_partition([torch.full((len(r),), .75, dtype=torch.float64) for r in rows], rows)
            self.assertAlmostEqual((prior - lp).item(), 3 * .75)

    def test_crossing_only_correspondences_are_impossible(self):
        import torch

        with self.assertRaisesRegex(ValueError, "No mass-conserving"):
            transport_partition([torch.zeros(1), torch.zeros(1)], [[1], [0]])

    def test_source_at_the_edge_cannot_silently_disappear(self):
        import torch

        with self.assertRaisesRegex(ValueError, "No mass-conserving"):
            transport_partition([torch.zeros(1), torch.zeros(0)], [[1], []])

    def test_zero_width_transport_matches_framewise_kd_including_alternatives(self):
        import torch

        student = torch.tensor([[.1, .8, .1], [.8, .1, .1], [.1, .1, .8]], dtype=torch.float64).log().requires_grad_()
        ids = torch.tensor([[0, 1, 2]] * 3)
        p = torch.tensor([[.1, .7, .2], [.8, .1, .1], [.1, .2, .7]])
        loss, mass = temporal_target_loss(student, ids, p, [1, 2], DelayBounds(0, 0))
        normalized = p / p.sum(-1, keepdim=True)
        expected = -(student * normalized).sum()
        torch.testing.assert_close(loss, expected)
        self.assertEqual(mass, 3)
        got_grad, = torch.autograd.grad(loss, student, retain_graph=True)
        want_grad, = torch.autograd.grad(expected, student)
        torch.testing.assert_close(got_grad, want_grad)

    def test_unequal_durations_do_not_duplicate_teacher_edge_rows(self):
        import torch

        student = torch.tensor([[.2, .8]] * 4, dtype=torch.float64).log().requires_grad_()
        ids = torch.tensor([[0, 1]] * 2)
        p = torch.tensor([[.1, .9], [.2, .8]])
        loss, mass = temporal_target_loss(student, ids, p, [1], DelayBounds(0, 2))
        loss.backward()
        self.assertEqual(mass, 2)
        torch.testing.assert_close(-student.grad.sum(0), (p / p.sum(-1, keepdim=True)).sum(0).double())


if __name__ == "__main__":
    unittest.main()

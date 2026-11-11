"""Small CPU checks: python3 -m unittest discover -s tools -p test_student_ctc.py."""

import ast
import itertools
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal"))
from student_ctc import stable_ctc_loss


def exhaustive(logits, target, blank=0):
    lp = logits.log_softmax(-1)
    paths = []
    for path in itertools.product(range(logits.shape[-1]), repeat=len(logits)):
        collapsed = [c for t, c in enumerate(path) if c != blank and (t == 0 or c != path[t - 1])]
        if collapsed == target:
            paths.append(sum(lp[t, c] for t, c in enumerate(path)))
    return -torch.logsumexp(torch.stack(paths), 0)


class StableCTCTests(unittest.TestCase):
    def test_loss_and_logit_gradient_match_exhaustive_ctc_paths(self):
        torch.manual_seed(12)
        for blank, target in ((0, []), (0, [1]), (0, [1, 1]), (0, [1, 2]), (2, [0, 0])):
            for time in (3, 4):
                with self.subTest(blank=blank, target=target, time=time):
                    logits = torch.randn(time, 1, 3, dtype=torch.float64, requires_grad=True)
                    got = stable_ctc_loss(logits, torch.tensor(target, dtype=torch.long), torch.tensor([time]),
                                          torch.tensor([len(target)]), blank=blank)
                    expected = exhaustive(logits[:, 0], target, blank)
                    torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-12)
                    a, = torch.autograd.grad(got, logits, retain_graph=True)
                    b, = torch.autograd.grad(expected, logits)
                    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_recording_weights_and_padding_do_not_change_valid_gradients(self):
        torch.manual_seed(23)
        logits = torch.randn(7, 2, 3, dtype=torch.float64, requires_grad=True)
        targets, lengths, counts = torch.tensor([1, 1, 2]), torch.tensor([3, 5]), torch.tensor([2, 1])
        weights = torch.tensor([0.3, 1.7], dtype=torch.float64)
        loss = (stable_ctc_loss(logits, targets, lengths, counts, reduction="none") * weights).sum() / lengths.sum()
        expected = (exhaustive(logits[:3, 0], [1, 1]) * weights[0] +
                    exhaustive(logits[:5, 1], [2]) * weights[1]) / lengths.sum()
        a, = torch.autograd.grad(loss, logits, retain_graph=True)
        b, = torch.autograd.grad(expected, logits)
        torch.testing.assert_close(loss, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        self.assertEqual(a[3:, 0].count_nonzero().item(), 0)
        self.assertEqual(a[5:, 1].count_nonzero().item(), 0)

    def test_impossible_alignment_is_zeroed_without_masking_feasible_recording(self):
        logits = torch.zeros(2, 2, 3, dtype=torch.float64, requires_grad=True)
        got = stable_ctc_loss(logits, torch.tensor([1, 1, 2]), torch.tensor([2, 2]), torch.tensor([2, 1]), reduction="none")
        self.assertEqual(got[0].item(), 0)
        torch.testing.assert_close(got[1], exhaustive(logits[:, 1], [2]))
        got.sum().backward()
        self.assertEqual(logits.grad[:, 0].count_nonzero().item(), 0)
        self.assertGreater(logits.grad[:, 1].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_zero_frame_empty_transcript_has_zero_loss_and_gradient(self):
        logits = torch.randn(3, 2, 3, requires_grad=True)
        got = stable_ctc_loss(logits, torch.tensor([1]), torch.tensor([0, 3]), torch.tensor([0, 1]), reduction="none")
        self.assertEqual(got[0].item(), 0)
        got.sum().backward()
        self.assertEqual(logits.grad[:, 0].count_nonzero().item(), 0)

    def test_near_deterministic_alignment_retains_small_logit_gradient(self):
        # Two frames can emit label 1 unless both emit blank. Its excluded path
        # changes the gradient by ~3.8e-11, below float32 probability resolution.
        reference = torch.tensor([[[0., 12.]], [[0., 12.]]], dtype=torch.float64, requires_grad=True)
        expected = exhaustive(reference[:, 0], [1])
        b, = torch.autograd.grad(expected, reference)
        for dtype in (torch.float64, torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = reference.detach().to(dtype).requires_grad_()
                got = stable_ctc_loss(logits, torch.tensor([1]), torch.tensor([2]), torch.tensor([1]))
                a, = torch.autograd.grad(got, logits)
                self.assertTrue((a[:, :, 0] > 0).all())
                self.assertTrue((a[:, :, 1] < 0).all())
                torch.testing.assert_close(a.double(), b, atol=2e-15,
                                           rtol=0.005 if dtype == torch.bfloat16 else 1e-4)

    def test_long_sequence_matches_double_precision_native_loss_and_gradient(self):
        torch.manual_seed(42)
        logits = torch.randn(240, 2, 5, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([1, 2, 2, 3] * 8 + [3, 1, 4] * 12)
        lengths, counts = torch.tensor([213, 240]), torch.tensor([32, 36])
        got = stable_ctc_loss(logits, targets, lengths, counts)
        expected = F.ctc_loss(logits.log_softmax(-1), targets, lengths, counts, reduction="sum", zero_infinity=True)
        a, = torch.autograd.grad(got, logits, retain_graph=True)
        b, = torch.autograd.grad(expected, logits)
        torch.testing.assert_close(got, expected, atol=1e-10, rtol=1e-12)
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)

    def test_complete_float32_head_gradient_matches_exhaustive_reference(self):
        # The public boundary is logits: verify the projection's parameter
        # gradients too, not just a leaf log-probability subtraction.
        torch.manual_seed(18)
        features = torch.randn(4, 1, 3)
        head = torch.nn.Linear(3, 3)
        logits = head(features)
        got = stable_ctc_loss(logits, torch.tensor([1, 2]), torch.tensor([4]), torch.tensor([2]))
        a = torch.autograd.grad(got, tuple(head.parameters()))
        oracle = torch.nn.Linear(3, 3).double()
        oracle.load_state_dict(head.state_dict())
        expected = exhaustive(oracle(features.double())[:, 0], [1, 2])
        b = torch.autograd.grad(expected, tuple(oracle.parameters()))
        for actual, reference in zip(a, b):
            torch.testing.assert_close(actual.double(), reference, atol=1e-7, rtol=1e-6)

    def test_invalid_lengths_and_blank_transcripts_are_rejected(self):
        logits = torch.zeros(3, 1, 3)
        for target, lengths, counts in (([0], [3], [1]), ([1], [4], [1]), ([1], [3], [2])):
            with self.assertRaises(ValueError):
                stable_ctc_loss(logits, torch.tensor(target), torch.tensor(lengths), torch.tensor(counts))

    def test_finite_differences_and_reduced_precision_autocast_boundary(self):
        torch.manual_seed(3)
        logits = torch.randn(4, 1, 3, dtype=torch.float64, requires_grad=True)
        args = (torch.tensor([1, 2]), torch.tensor([4]), torch.tensor([2]))
        self.assertTrue(torch.autograd.gradcheck(lambda x: stable_ctc_loss(x, *args), (logits,)))
        low = logits.detach().float().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            got = stable_ctc_loss(low, *args)
        got.backward()
        self.assertEqual(got.dtype, torch.float32)
        self.assertEqual(low.grad.dtype, torch.float32)
        torch.testing.assert_close(got.double(), stable_ctc_loss(low.detach().double(), *args), atol=1e-6, rtol=1e-6)

    def test_training_dispatch_preserves_frame_reduction_and_kd(self):
        # Load the actual loss function without importing the Modal deployment or
        # constructing an encoder; its dependencies are replaced by a KD oracle.
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "modal/student_train.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "losses")
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "student_train.py", "exec"), namespace)
        student = types.ModuleType("student")
        student.soft_target_loss = lambda lp, ids, p, valid: -(lp.gather(2, ids) * p * valid[:, :, None]).sum() / valid.sum()
        batch = dict(feats=torch.zeros(2, 4, 2), feat_lens=torch.tensor([4, 3]), targets=torch.tensor([1, 2]),
                     target_lens=torch.tensor([1, 1]), n_teacher=torch.tensor([4, 3]),
                     top_ids=torch.zeros(2, 4, 5, dtype=torch.long), top_p=torch.full((2, 4, 5), 0.2))
        logits = torch.randn(2, 4, 3, requires_grad=True)

        def model(feats, lens, return_logits=False):
            return (logits if return_logits else logits.log_softmax(-1)), lens

        from unittest.mock import patch
        with patch.dict(sys.modules, student=student):
            cfg = dict(ctc_weight=0.7, kd_weight=0.3, kd_delay=0)
            native = namespace["losses"](model, batch, cfg, "cpu")
            stable = namespace["losses"](model, batch, {**cfg, "ctc_arithmetic": "stable"}, "cpu")
        self.assertEqual(stable["frames"].item(), 7)
        torch.testing.assert_close(stable["kd"], native["kd"], atol=0, rtol=0)
        torch.testing.assert_close(stable["ctc"], native["ctc"], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(stable["loss"], 0.7 * stable["ctc"] + 0.3 * stable["kd"])
        a, = torch.autograd.grad(native["loss"], logits, retain_graph=True)
        b, = torch.autograd.grad(stable["loss"], logits)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

"""Exact finite-path and derivative checks for transcript-conditioned entropy."""

import itertools
import unittest

import torch

from student_alignment_entropy import alignment_entropy, entropy_deficit


def collapse(path, blank=0):
    return [x for i, x in enumerate(path) if x != blank and (i == 0 or x != path[i - 1])]


def enumerate_stats(lp, transcript, blank=0):
    paths = [p for p in itertools.product(range(lp.shape[-1]), repeat=lp.shape[0])
             if collapse(p, blank) == transcript]
    weights = torch.stack([sum(lp[t, k] for t, k in enumerate(p)) for p in paths])
    log_z = weights.logsumexp(0)
    entropy = -(weights.softmax(0) * (weights - log_z)).sum()
    return entropy, log_z, torch.tensor(len(paths), dtype=lp.dtype).log()


class EntropyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exhaustive_paths_and_covariance_gradient(self):
        for transcript in ([], [1], [1, 1], [1, 2], [2, 1, 2]):
            torch.manual_seed(7)
            lp = torch.randn(1, 4, 3, dtype=torch.float64).log_softmax(-1).requires_grad_()
            got = alignment_entropy(lp, torch.tensor(transcript, dtype=torch.long), [4], [len(transcript)])
            want = enumerate_stats(lp[0], transcript)
            for actual, expected in zip(got, want):
                torch.testing.assert_close(actual[0], expected)
            for i in (0, 1):
                actual_grad = torch.autograd.grad(got[i].sum(), lp, retain_graph=True)[0]
                expected_grad = torch.autograd.grad(want[i], lp, retain_graph=True)[0]
                torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-10)

    def test_uniform_is_neutral_and_independent_of_alphabet_identity(self):
        lp = torch.full((2, 7, 6), -torch.tensor(6.).log(), dtype=torch.float64, requires_grad=True)
        h, z, baseline = alignment_entropy(lp, torch.tensor([[1, 1, 2], [4, 4, 3]]), [7, 7], [3, 3])
        torch.testing.assert_close(h, baseline, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(h[0], h[1])
        self.assertFalse(baseline.requires_grad)
        h.sum().backward()
        torch.testing.assert_close(lp.grad, torch.zeros_like(lp), atol=1e-12, rtol=0)

    def test_packed_variable_lengths_and_padding_have_no_influence(self):
        torch.manual_seed(21)
        lp = torch.randn(3, 6, 4, dtype=torch.float64).log_softmax(-1).requires_grad_()
        targets, lengths, sizes = torch.tensor([1, 2, 3, 3]), [4, 6, 3], [2, 2, 0]
        result = alignment_entropy(lp, targets, lengths, sizes)
        for row, transcript in enumerate(([1, 2], [3, 3], [])):
            for actual, want in zip(result, enumerate_stats(lp[row, :lengths[row]], transcript)):
                torch.testing.assert_close(actual[row], want)
        (result[0].sum() + result[1].sum()).backward()
        self.assertEqual(lp.grad[0, 4:].abs().sum().item(), 0)
        self.assertEqual(lp.grad[2, 3:].abs().sum().item(), 0)

    def test_gradcheck_and_nonzero_entropy_gradient(self):
        torch.manual_seed(2)
        lp = torch.randn(2, 5, 3, dtype=torch.float64).log_softmax(-1).requires_grad_()
        fn = lambda x: entropy_deficit(x, torch.tensor([1, 2, 2, 2]), [4, 5], [2, 2])
        self.assertTrue(torch.autograd.gradcheck(fn, (lp,)))
        fn(lp).backward()
        self.assertGreater(lp.grad.abs().sum().item(), 0.01)

    def test_nonzero_blank_and_impossible_transcript(self):
        lp = torch.randn(1, 4, 3, dtype=torch.float64).log_softmax(-1)
        result = alignment_entropy(lp, torch.tensor([0, 1]), [4], [2], blank=2)
        for actual, expected in zip(result, enumerate_stats(lp[0], [0, 1], blank=2)):
            torch.testing.assert_close(actual[0], expected)
        for targets, frames, size in (([1, 1], 2, 2), ([0], 4, 1), ([4], 4, 1)):
            with self.assertRaises(ValueError):
                alignment_entropy(lp, torch.tensor(targets), [frames], [size])

    def test_single_valid_path_has_zero_entropy(self):
        lp = torch.randn(1, 3, 3, dtype=torch.float64).log_softmax(-1).requires_grad_()
        entropy, _, baseline = alignment_entropy(lp, torch.tensor([1, 1]), [3], [2])
        torch.testing.assert_close(entropy, torch.zeros_like(entropy), atol=1e-12, rtol=0)
        self.assertEqual(baseline.item(), 0)
        entropy.sum().backward()
        torch.testing.assert_close(lp.grad, torch.zeros_like(lp), atol=1e-12, rtol=0)


class MatchedTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_help_admission_rejects_extra_mistakes_and_heldout_speakers(self):
        from student_entropy_finetune import help_none_rows

        clean = dict(id="ordinary", speaker="fit", split="train", kind="none", scored=True,
                     events=[], non_errors=[], unsupported=[])
        rows = [clean]
        for key, value in (("kind", "repeat"), ("scored", False), ("excluded", "audit"),
                           ("events", [{"errorClass": "wrong_word"}]),
                           ("non_errors", [{"kind": "repeat"}]), ("unsupported", [{"rule": "other"}])):
            rows.append({**clean, "id": key, key: value})
        rows.extend([{**clean, "id": "leaked", "speaker": "held"},
                     {**clean, "id": "held", "speaker": "held", "split": "eval"}])
        self.assertEqual(help_none_rows(rows), [clean])
        self.assertEqual(help_none_rows([{**clean, "scored": None}]), [])

    def test_control_and_entropy_differ_only_by_conditioned_path_reward(self):
        from student_entropy_finetune import objective

        torch.manual_seed(19)
        lp = torch.randn(1, 4, 3, dtype=torch.float64).log_softmax(-1).requires_grad_()
        frozen = torch.randn_like(lp).log_softmax(-1)
        batch = dict(targets=torch.tensor([1, 2]), target_lens=torch.tensor([2]),
                     top_ids=torch.zeros(1, 1, 5, dtype=torch.long), top_p=torch.zeros(1, 1, 5),
                     n_teacher=torch.tensor([1]))
        cfg = dict(kd_weight=.3, preservation_weight=1., entropy_weight=.05)
        control, _ = objective(lp, torch.tensor([4]), batch, frozen, cfg, False)
        trained, metrics = objective(lp, torch.tensor([4]), batch, frozen, cfg, True)
        h, z, neutral = enumerate_stats(lp[0], [1, 2])
        anchor = (frozen.exp() * (frozen - lp)).sum() / 4
        torch.testing.assert_close(control, -z / 4 + anchor)
        torch.testing.assert_close(trained - control, cfg["entropy_weight"] * (neutral - h) / 4)
        actual_grad = torch.autograd.grad(trained - control, lp, retain_graph=True)[0]
        oracle_grad = torch.autograd.grad(-cfg["entropy_weight"] * h / 4, lp)[0]
        torch.testing.assert_close(actual_grad, oracle_grad)
        self.assertEqual(metrics["kd"], 0)


if __name__ == "__main__":
    unittest.main()

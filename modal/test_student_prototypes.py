import itertools
import unittest

import torch

from student_prototypes import ConventionalHead, PrototypeHead, ctc_occupancies


class PrototypeTests(unittest.TestCase):
    def test_marginalization_and_normalization(self):
        torch.manual_seed(1)
        head = PrototypeHead(5, 3, 2)
        x = torch.randn(4, 5)
        actual = head(x)
        sims = torch.nn.functional.normalize(x) @ torch.nn.functional.normalize(head.weight, dim=-1).flatten(0, 1).T
        expected = (sims.reshape(4, 3, 2) * head.log_scale.exp() + head.mixture.log_softmax(-1)).logsumexp(-1) + head.bias
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.softmax(-1).sum(-1), torch.ones(4))
        # Duplicating a realization does not earn twice the phone mass.
        one = PrototypeHead(5, 3, 1)
        one.weight.data.copy_(head.weight[:, :1])
        head.weight.data[:, 1].copy_(head.weight.data[:, 0])
        torch.testing.assert_close(head(x), one(x))

    def test_separated_realizations_and_gradients(self):
        head = PrototypeHead(2, 2, 2)
        x = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        labels = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        head.initialize(x, labels)
        self.assertEqual(head(x).argmax(-1).tolist(), [0, 0, 1, 1])
        head(x).log_softmax(-1).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_control_preserves_original_and_matches_parameters(self):
        original = torch.nn.Linear(256, 43)
        control = ConventionalHead(original, 2)
        torch.testing.assert_close(control(torch.ones(7, 256)), original(torch.ones(7, 256)))
        a = sum(p.numel() for p in PrototypeHead(256, 43, 2).parameters())
        b = sum(p.numel() for p in control.parameters())
        self.assertLess(abs(a - b) / a, .001)

    def test_occupancies_against_exhaustive_paths(self):
        torch.manual_seed(4)
        lp = torch.randn(5, 3).log_softmax(-1)
        for labels in ([], [1], [1, 1], [1, 2]):
            mass = torch.zeros_like(lp)
            total = 0.
            for path in itertools.product(range(3), repeat=5):
                collapsed = [v for i, v in enumerate(path) if v != 0 and (i == 0 or v != path[i - 1])]
                if collapsed != labels:
                    continue
                p = lp[range(5), list(path)].sum().exp()
                total += p
                mass[range(5), list(path)] += p
            torch.testing.assert_close(ctc_occupancies(lp, labels), mass / total)

    def test_impossible_transcript_and_missing_initialization_support(self):
        with self.assertRaises(ValueError):
            ctc_occupancies(torch.zeros(1, 3).log_softmax(-1), [1, 1])
        with self.assertRaises(ValueError):
            PrototypeHead(2, 3).initialize(torch.eye(2), torch.tensor([[1., 0., 0.], [0., 1., 0.]]))


if __name__ == "__main__":
    unittest.main()

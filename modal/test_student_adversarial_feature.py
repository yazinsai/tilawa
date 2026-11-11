"""Small mechanism checks; real acoustic training remains on Modal."""
import unittest
import numpy as np
import torch
from student_adversarial_feature import head, loss, perturb, sequence_score


class AdversarialFeatureTest(unittest.TestCase):
    def test_bounded_attack_favors_incorrect_pronunciation(self):
        torch.manual_seed(8)
        model = head(3)
        x = torch.randn(8, 32)
        example = dict(correct=[1], rival=[2])
        basis = torch.eye(32)[:, :8]
        delta = perturb(model, x, example, basis, .3)
        self.assertLessEqual(float(delta.norm()) / len(x) ** .5, .300001)
        self.assertTrue(torch.equal(delta[:, 8:], torch.zeros_like(delta[:, 8:])))
        self.assertGreater(float(loss(model, x + delta, example)), float(loss(model, x, example)))
        self.assertFalse(delta.requires_grad)

    def test_candidate_order_is_unprivileged(self):
        torch.manual_seed(8)
        model, x = head(3), torch.randn(8, 32)
        example = dict(correct=[1], rival=[2])
        torch.testing.assert_close(loss(model, x, example), loss(model, x, example, True))
        self.assertFalse(x.requires_grad)

    def test_complete_normalized_timing(self):
        lp = torch.full((8, 3), -1.)
        for ids in ([1], [1, 1, 2]):
            self.assertAlmostEqual(float(sequence_score(lp, ids)), -8., places=5)
        self.assertTrue(torch.isneginf(sequence_score(lp[:1], [1, 1])))


if __name__ == '__main__':
    unittest.main()

"""Small mathematical tests; no recordings, practice labels or acoustic training."""
import itertools
import unittest

import numpy as np
import torch

from student_logit_uncertainty import (folds_for, head, outputs, possible, quadrature,
                                      sequence_likelihood, validate_rows)


def path_mass(probabilities, ids):
    total = 0
    for path in itertools.product(range(probabilities.shape[1]), repeat=len(probabilities)):
        collapsed = [p for t, p in enumerate(path) if p and (t == 0 or p != path[t-1])]
        if collapsed == ids:
            total += np.prod([probabilities[t, p] for t, p in enumerate(path)])
    return total


class LogisticNormalTests(unittest.TestCase):
    def test_exact_paths_and_block_quadrature(self):
        mean = torch.tensor([[.2, -.3, .1]]*4, dtype=torch.float64)
        loading = torch.tensor([[[0, .7, -.4]]]*4, dtype=torch.float64)
        z, weights = quadrature(1, 3)
        exact = 0
        for i, j in itertools.product(range(3), repeat=2):
            noise = torch.tensor([z[i], z[j], z[j], z[j]])
            rows = (mean+loading[:, 0]*noise).softmax(-1).numpy()
            exact += weights[i]*weights[j]*path_mass(rows, [1, 2])
        actual = sequence_likelihood(mean, loading, [1, 2], 4, from_frame=3, order=3).exp().item()
        self.assertAlmostEqual(actual, exact, places=12)

    def test_zero_covariance_matches_native_ctc(self):
        mean = torch.tensor([[.2, -.1, .4], [.1, .3, -.2], [-.2, .1, .3]], dtype=torch.float64)
        zero = torch.zeros(3, 2, 3, dtype=torch.float64)
        for ids in ([1], [1, 2], [1, 1]):
            native = -torch.nn.functional.ctc_loss(mean.log_softmax(-1).unsqueeze(1), torch.tensor(ids),
                                                  torch.tensor([3]), torch.tensor([len(ids)]), reduction='sum')
            actual = sequence_likelihood(mean, zero, ids, 4, from_frame=1)
            self.assertAlmostEqual(native.item(), actual.item(), places=12)
        self.assertFalse(possible(2, [1, 1]))
        self.assertTrue(possible(3, [1, 1]))

    def test_shared_uncertainty_can_reverse_without_changing_mean(self):
        mean = torch.tensor([[1., 0, .1]]*4, dtype=torch.float64)
        loading = torch.tensor([[[0, 4., 0]]]*4, dtype=torch.float64)
        a, b = [sequence_likelihood(mean, None, [p], 4).item() for p in (1, 2)]
        c, d = [sequence_likelihood(mean, loading, [p], 4).item() for p in (1, 2)]
        self.assertLess(a, b)
        self.assertGreater(c, d)
        self.assertAlmostEqual(np.exp(c), .3317346709873556, places=12)

    def test_marginalized_training_has_finite_mean_and_covariance_gradients(self):
        model = head(3, 2)
        x = torch.randn(12, 32)
        mean, loading = outputs(model, x, 3, 2)
        (-sequence_likelihood(mean, loading, [1, 2], 4)+.01*loading.square().mean()).backward()
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model[2].weight.grad[:3].abs().sum().item(), 0)
        self.assertGreater(model[2].weight.grad[3:].abs().sum().item(), 0)
        # Both arms have identical tensors and every control channel participates.
        model.zero_grad()
        deterministic, covariance = outputs(model, x, 3, 2, deterministic=True)
        self.assertIsNone(covariance)
        (-sequence_likelihood(deterministic, None, [1, 2], 4)).backward()
        self.assertGreater(model[2].weight.grad[3:].abs().sum().item(), 0)

    def test_product_quadrature_normal_moments(self):
        for rank, order in itertools.product((1, 2), (3, 5)):
            z, weights = quadrature(rank, order)
            np.testing.assert_allclose(weights.sum(), 1, atol=1e-14)
            np.testing.assert_allclose((weights[:, None]*z).sum(0), 0, atol=1e-14)
            np.testing.assert_allclose((weights[:, None]*z*z).sum(0), 1, atol=1e-14)

    def test_speaker_groups_and_fail_closed_provenance(self):
        rows = [dict(speaker=s) for s in ('a', 'a', 'b', 'c', 'd')]
        folds = folds_for(rows, ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(set(folds[s] for s in ('a', 'b', 'c', 'd')), {0, 1, 2})
        row = dict(id='recording', speaker='speaker', source='help-train', split='train', excluded=False,
                   kind='none', pronunciation=[1], features=[[0.]*32]*3, logProbs=[[-1.]*3]*3)
        validate_rows([row], dict(tokens='ab'), set())
        for change in (dict(split='test'), dict(excluded=True), dict(source='practice'), dict(speaker='')):
            with self.assertRaises(ValueError):
                validate_rows([dict(row, **change)], dict(tokens='ab'), set())
        with self.assertRaises(ValueError):
            validate_rows([row], dict(tokens='ab'), {'speaker'})
        with self.assertRaises(ValueError):
            validate_rows([row, row], dict(tokens='ab'), set())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

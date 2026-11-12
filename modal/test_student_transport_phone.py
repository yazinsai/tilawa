"""Mechanism checks; no lab examples or benchmark-specific expectations."""
import unittest

import numpy as np

from student_transport_phone import (
    context_weights, equal_phone_weights, occupancies, sequence_score,
    sinkhorn, speaker_folds, transport_loss,
)


class TransportTrainingTest(unittest.TestCase):
    def test_soft_occupancies_keep_ambiguous_state_mass(self):
        lp = np.log(np.full((8, 4), .25))
        targets, states, labels = occupancies(lp, [1, 2, 1])
        np.testing.assert_allclose(targets.sum(1), 1)
        self.assertTrue(np.any((states > 0) & (states < 1)))
        targets, groups = context_weights(lp, [1, 2, 1], 'بَت')
        np.testing.assert_allclose(sum(groups.values()), 1, atol=1e-7)
        self.assertIn((1, '-', 'a'), groups)
        self.assertIn((1, 'a', '-'), groups)
        self.assertIsNone(occupancies(lp[:2], [1, 1]))

    def test_sinkhorn_has_balanced_marginals(self):
        import torch
        cost = torch.tensor([[0., 1., 2.], [2., 1., 0.]])
        coupling = sinkhorn(cost, .5)
        torch.testing.assert_close(coupling.sum(1), torch.full((2,), .5))
        torch.testing.assert_close(coupling.sum(0), torch.full((3,), 1/3))

    def test_restricted_coupling_cannot_solve_shift_by_swapping_phones(self):
        import torch
        source = torch.tensor([[-2.], [2.]], requires_grad=True)
        target = torch.tensor([[2.], [-2.]])
        keys = [(1, 'a', 'i'), (2, 'a', 'i')]
        control = transport_loss(source, target, keys, False, .1)
        conditional = transport_loss(source, target, keys, True, .1)
        self.assertLess(control.item(), 1e-6)
        self.assertAlmostEqual(conditional.item(), 16.)
        conditional.backward()
        torch.testing.assert_close(source.grad, torch.tensor([[-4.], [4.]]))

    def test_phone_and_recording_balance_is_independent_of_length(self):
        rows = [{'targets': np.array([[1., 0.], [0., 1.]], dtype='float32')},
                {'targets': np.tile([[1., 0.]], (20, 1)).astype('float32')}]
        weights = equal_phone_weights(rows)
        np.testing.assert_allclose(weights.sum(0), [.5, .5], atol=1e-7)
        self.assertAlmostEqual(float(weights[0, 0]), .25)
        self.assertAlmostEqual(float(weights[2:, 0].sum()), .25)

    def test_complete_timing_marginal_does_not_reward_path_count(self):
        lp = np.log(np.full((8, 3), 1/3))
        for ids in ([1], [1, 2], [1, 1], [1, 2, 1]):
            self.assertAlmostEqual(sequence_score(lp, ids), 8*np.log(1/3))
        self.assertEqual(sequence_score(lp[:2], [1, 1]), -np.inf)

    def test_folds_include_preservation_only_speakers_and_balance_domains(self):
        rows = [{'speaker': f'{d}-{i}', 'domain': d} for d in ('phone', 'studio') for i in range(6)]
        controls = [{'speaker': 'preservation-only'}]
        folds = speaker_folds(rows, controls)
        self.assertIn('preservation-only', folds)
        for d in ('phone', 'studio'):
            self.assertEqual([sum(folds[r['speaker']] == f for r in rows if r['domain'] == d)
                              for f in range(3)], [2, 2, 2])


if __name__ == '__main__':
    unittest.main()

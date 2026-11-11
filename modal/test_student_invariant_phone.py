"""Training invariants on small synthetic data, run on Modal with the experiment."""
import unittest

import numpy as np
import torch

from student_invariant_phone import balanced_groups, classifier_penalty, condition_groups, head, occupancies, speaker_folds


class InvariantTrainingTests(unittest.TestCase):
    def test_soft_balancing_equalizes_all_environment_phone_masses(self):
        rows = [dict(features=np.zeros((2, 32)), targets=np.array(y, dtype='float32')) for y in (
            [[.8, .1, .1], [.6, .3, .1]], [[.1, .8, .1], [.1, .6, .3]])]
        groups, distribution = balanced_groups(rows, [0, 1], 3)
        for g in groups:
            np.testing.assert_allclose((g['y']*g['w'][:, None]).sum(0), distribution, atol=1e-7)

    def test_grouping_is_independent_of_transcript_and_speaker(self):
        rows = [dict(descriptor=[i]*8, speaker=str(i), pronunciation=[1]) for i in range(6)]
        labels, metadata = condition_groups(rows, 2)
        changed = [dict(r, speaker='other', pronunciation=[3, 2, 3]) for r in rows]
        labels2, metadata2 = condition_groups(changed, 2)
        np.testing.assert_array_equal(labels, labels2)
        self.assertEqual(metadata, metadata2)

    def test_stationarity_penalty_differentiates_into_representation(self):
        torch.manual_seed(1)
        model = head(3)
        x = torch.randn(8, 32)
        risk = -model(x).log_softmax(-1)[:, 1].mean()
        penalty = classifier_penalty(risk, model)
        gradient, = torch.autograd.grad(penalty, model[0].weight)
        self.assertGreater(penalty.item(), 0)
        self.assertGreater(gradient.norm().item(), 0)
        # Opposite group preferences do not cancel before squaring.
        logits = torch.tensor([[0., 0.]], requires_grad=True)
        risks = [-logits.log_softmax(-1)[0, label] for label in (0, 1)]
        separate = [torch.autograd.grad(r, logits, retain_graph=True)[0] for r in risks]
        self.assertGreater(sum(g.square().sum().item() for g in separate), 0)
        self.assertAlmostEqual(sum(separate).square().sum().item(), 0)

    def test_occupancies_preserve_complete_repeated_phone_topology(self):
        lp = np.log(np.array([[.1, .8, .1], [.8, .1, .1], [.1, .8, .1]]))
        targets, _ = occupancies(lp, [1, 1])
        np.testing.assert_allclose(targets, [[0, 1, 0], [1, 0, 0], [0, 1, 0]])
        self.assertIsNone(occupancies(lp[:2], [1, 1]))

    def test_speaker_folds_never_split_recordings_of_one_speaker(self):
        rows = [dict(speaker=s) for s in ('a', 'b', 'c', 'a')]
        folds = speaker_folds(rows, ['a', 'b', 'c', 'd'])
        self.assertEqual(set(folds), {'a', 'b', 'c', 'd'})
        self.assertEqual(len(set(folds.values())), 3)


if __name__ == '__main__':
    unittest.main()

import itertools
import unittest
from unittest.mock import patch

import torch

from student_count_distribution import (collapsed_count_log_distribution,
    distribution_count_loss, expected_collapsed_counts)


def enumerate_counts(probabilities, phone, cap):
    result = torch.zeros(cap + 2, dtype=probabilities.dtype)
    for path in itertools.product(range(probabilities.shape[1]), repeat=len(probabilities)):
        mass = probabilities.new_tensor(1.)
        previous, count = 0, 0
        for t, label in enumerate(path):
            mass = mass * probabilities[t, label]
            count += label == phone and previous != phone
            previous = label
        result[min(count, cap + 1)] += mass
    return result


class CollapsedCountTests(unittest.TestCase):
    def test_complete_distribution_and_saturated_overflow_against_all_paths(self):
        p = torch.tensor([[.4, .3, .3], [.1, .7, .2], [.6, .2, .2], [.1, .8, .1]], dtype=torch.float64)
        for cap in [0, 1, 2, 4]:
            got = collapsed_count_log_distribution(p.log(), [1, 2], cap).exp()
            for i, phone in enumerate([1, 2]):
                torch.testing.assert_close(got[i], enumerate_counts(p, phone, cap))
            torch.testing.assert_close(got.sum(-1), torch.ones(2, dtype=p.dtype))

    def test_ordinary_collapse_continuation_and_blank_separated_repetition(self):
        for path, wanted in [([1, 1, 1], 1), ([1, 0, 1], 2), ([1, 2, 1], 2), ([0, 0, 0], 0)]:
            p = torch.nn.functional.one_hot(torch.tensor(path), 3).to(torch.float64)
            got = collapsed_count_log_distribution(p.log(), [1], 2).exp()[0]
            expected = torch.zeros(4, dtype=p.dtype)
            expected[wanted] = 1
            torch.testing.assert_close(got, expected)

    def test_first_moment_matches_exact_distribution(self):
        torch.manual_seed(8)
        lp = torch.randn(9, 4, dtype=torch.float64).log_softmax(-1)
        got = collapsed_count_log_distribution(lp, [1, 2, 3], 5).exp()
        torch.testing.assert_close(got[:, -1], torch.zeros(3, dtype=lp.dtype))
        torch.testing.assert_close((got[:, :-1] * torch.arange(6)).sum(-1),
                                   expected_collapsed_counts(lp, [1, 2, 3]))

    def test_zero_and_one_can_have_same_mean_as_spread_counts(self):
        # One certain occurrence versus two independent opportunities, each
        # at probability one half: both mean one, but only the first has P0=0.
        fixed = torch.tensor([[0., 1.], [1., 0.], [1., 0.]], dtype=torch.float64)
        spread = torch.tensor([[.5, .5], [1., 0.], [.5, .5]], dtype=torch.float64)
        torch.testing.assert_close(expected_collapsed_counts(fixed.log(), [1]),
                                   expected_collapsed_counts(spread.log(), [1]))
        torch.testing.assert_close(collapsed_count_log_distribution(spread.log(), [1], 2).exp(),
                                   torch.tensor([[.25, .5, .25, 0.]], dtype=torch.float64))
        self.assertLess(float(distribution_count_loss(fixed.log(), torch.tensor([1]), [1])),
                        float(distribution_count_loss(spread.log(), torch.tensor([1]), [1])))

    def test_loss_and_gradient_match_enumerated_paths(self):
        torch.manual_seed(3)
        logits = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        target = torch.tensor([1, 1, 2])
        lp = logits.log_softmax(-1)
        loss = distribution_count_loss(lp, target, [1, 2])
        brute = -torch.stack([enumerate_counts(lp.exp(), 1, 2)[2],
                              enumerate_counts(lp.exp(), 2, 2)[1]]).log().mean()
        torch.testing.assert_close(loss, brute)
        grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        reference = torch.autograd.grad(brute, logits)[0]
        self.assertTrue(torch.isfinite(grad).all())
        torch.testing.assert_close(grad, reference)
        self.assertTrue(torch.autograd.gradcheck(lambda x: distribution_count_loss(
            x.log_softmax(-1), target, [1, 2]), (logits,)))

    def test_real_pause_zero_targets_reduce_extra_emissions(self):
        logits = torch.tensor([[.3, .1, -.2], [.3, .1, -.2]], dtype=torch.float64, requires_grad=True)
        loss = distribution_count_loss(logits.log_softmax(-1), torch.tensor([], dtype=torch.int64), [1, 2])
        grad, = torch.autograd.grad(loss, logits)
        self.assertTrue((grad[:, 0] < 0).all())
        self.assertTrue((grad[:, 1:] > 0).all())

    def test_empty_frames_and_nonleading_blank(self):
        got = collapsed_count_log_distribution(torch.empty(0, 3), [1, 2], 1).exp()
        torch.testing.assert_close(got, torch.tensor([[1., 0., 0.], [1., 0., 0.]]))
        with self.assertRaises(ValueError):
            collapsed_count_log_distribution(torch.zeros(2, 3), [0], 1)

    def test_three_arms_share_initialization_views_order_and_selected_updates(self):
        import copy
        import student_count_distribution as module
        from student_count import phone_head, log_probs
        import numpy as np
        head = phone_head(np.array([[.1, .2], [-.2, .3], [.3, -.1]], np.float32), np.zeros(3, np.float32))
        def row(speaker, offset):
            features = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 10 + offset
            return dict(speaker=speaker, views=[dict(features=features,
                shipped=log_probs(head, features).detach(), target=torch.tensor([1, 2]))])
        rows = [row('fit', 0.), row('fit', .1), row('selection', .2)]
        groups = dict(fit='fit', selection='selection')
        calls, validations, initial = [], [], []
        original = module.recording_loss
        def recording(head, views, mode, strength, consonants):
            calls.append((mode, strength, id(views)))
            return original(head, views, mode, strength, consonants)
        def validate(candidate, selection):
            self.assertEqual([r['speaker'] for r in selection], ['selection'])
            epoch = len(validations) // 5
            if epoch == 0:
                initial.append(copy.deepcopy(candidate.state_dict()))
            validations.append(epoch)
            return dict(objective=[10., 5., 1., 4.][epoch])
        with patch.object(module, 'recording_loss', side_effect=recording):
            heads, report = module.matched_refit(head, rows, groups, validate,
                (.1, 1.), 3, .001, 42, [1, 2])
        self.assertEqual(set(heads), {'ordinary', 'expected', 'distribution'})
        self.assertEqual(report['selectedEpoch'], 2)
        self.assertEqual(report['selectedUpdates'], 4)
        self.assertEqual(report['trainingUpdates'], 6)
        self.assertEqual(len(calls), 30)
        for state in initial:
            for key, value in state.items():
                torch.testing.assert_close(value, head.state_dict()[key], atol=0, rtol=0)
        for epoch in range(3):
            orders = [[v for _, _, v in calls[epoch * 10 + arm * 2:epoch * 10 + arm * 2 + 2]] for arm in range(5)]
            self.assertTrue(all(order == orders[0] for order in orders))


if __name__ == '__main__':
    unittest.main()

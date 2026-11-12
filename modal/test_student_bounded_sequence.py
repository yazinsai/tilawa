"""Mathematical controls for the bounded sequence objective (CPU, synthetic paths)."""
import itertools
import math
import unittest

import torch
from student_bounded_sequence import brier_from_log_probs, prefix_candidates, sequence_log_prob


class ProperSequenceTests(unittest.TestCase):
    def test_exact_partition_including_blank_and_repeated_labels(self):
        probabilities = torch.tensor([[.5,.3,.2],[.2,.6,.2],[.3,.2,.5],[.4,.4,.2]], dtype=torch.float64)
        masses = {}
        for path in itertools.product(range(3), repeat=4):
            collapsed = tuple(p for i,p in enumerate(path) if p and (i == 0 or p != path[i-1]))
            masses[collapsed] = masses.get(collapsed, 0)+math.prod(float(probabilities[t,p]) for t,p in enumerate(path))
        sequences = list(masses)
        actual = sequence_log_prob(probabilities.log(), sequences).exp()
        torch.testing.assert_close(actual, torch.tensor(list(masses.values()), dtype=torch.float64))
        self.assertAlmostEqual(float(actual.sum()), 1., places=12)

    def test_remaining_bucket_and_bounds_without_named_renormalization(self):
        p = torch.tensor([.2,.3], dtype=torch.float64)
        self.assertAlmostEqual(float(brier_from_log_probs(p.log())), .64+.09+.25, places=12)
        self.assertNotAlmostEqual(float(brier_from_log_probs(p.log())), .6**2+.6**2)
        for masses in ([1.,0.],[0.,1.],[0.,0.],[.2,.8]):
            value = float(brier_from_log_probs(torch.tensor(masses, dtype=torch.float64).log()))
            self.assertGreaterEqual(value, 0.)
            self.assertLessEqual(value, 2.)

    def test_poorly_explained_outcomes_have_bounded_gradient(self):
        gradients = []
        for logit in (-2.,-20.):
            z = torch.tensor(logit, dtype=torch.float64, requires_grad=True)
            logp = torch.nn.functional.logsigmoid(z)
            brier_from_log_probs(logp[None]).backward()
            gradients.append(abs(float(z.grad)))
            z.grad = None
            (-torch.nn.functional.logsigmoid(z)).backward()
            self.assertGreater(abs(float(z.grad)), .8)
        self.assertLess(gradients[1], gradients[0]*1e-6)

    def test_proposals_are_distinct_collapsed_strings_and_nested(self):
        rows = [[-.7,-1.,-2.],[-.2,-2.,-3.],[-.7,-1.,-2.]]
        proposals = prefix_candidates(rows, limit=32, branches=3)
        self.assertEqual(len(proposals), len(set(proposals)))
        self.assertIn((1,1), proposals)
        self.assertIn((), proposals)
        self.assertTrue(all(all(p != 0 for p in s) for s in proposals))
        self.assertTrue(all(len(s)+sum(a == b for a,b in zip(s,s[1:])) <= len(rows) for s in proposals))
        transcript = (1,)
        rivals = [s for s in proposals if s != transcript]
        self.assertEqual([transcript,*rivals[:8]], [transcript,*rivals[:32]][:9])


if __name__ == '__main__':
    unittest.main()

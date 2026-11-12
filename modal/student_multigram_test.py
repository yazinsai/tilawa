"""Exact-path and gradient checks for the training automaton."""
import itertools
import math
import unittest

from student_multigram import compiled_graph, graph_score, mine_units


class MultigramTest(unittest.TestCase):
    def test_exhaustive_paths_and_gradient(self):
        import torch
        units = ['a', 'b', 'ab', 'aa']
        for n in range(1, 5):
            logits = torch.randn(n, len(units)+1, dtype=torch.float64, requires_grad=True)
            rows = logits.log_softmax(-1)
            for text in ['a', 'ab', 'aa', 'aab', 'abab', 'aaaa']:
                total = 0.
                for path in itertools.product(range(len(units)+1), repeat=n):
                    spoken = ''.join(units[u-1] for t,u in enumerate(path) if u and (not t or u != path[t-1]))
                    if spoken == text:
                        total += math.exp(sum(float(rows[t,u].detach()) for t,u in enumerate(path)))
                actual = graph_score(rows, compiled_graph(text, units))
                if not total:
                    self.assertEqual(float(actual), -math.inf)
                else:
                    self.assertAlmostEqual(float(actual), math.log(total), places=10)
                    grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
                    self.assertTrue(torch.isfinite(grad).all())

    def test_singleton_equivalence(self):
        import torch
        text = 'aaba'
        rows = torch.randn(9,3, dtype=torch.float64).log_softmax(-1)
        expected = -torch.nn.functional.ctc_loss(rows[:,None,:], torch.tensor([1,1,2,1]), [9], [4], reduction='sum')
        self.assertAlmostEqual(float(graph_score(rows,compiled_graph(text,['a','b']))),float(expected),places=10)

    def test_unit_mining_requires_independent_training_support(self):
        rows = [dict(pronunciation='ab'*10,speaker='one')]
        self.assertEqual(mine_units(rows,'ab'),['a','b'])
        units = mine_units(rows+[dict(pronunciation='ab'*10,speaker='two')],'ab')
        self.assertIn('ab', units)
        self.assertNotIn('z', units)


if __name__ == '__main__':
    unittest.main()

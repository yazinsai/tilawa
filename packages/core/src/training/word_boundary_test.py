"""Small exhaustive check of latent CTC first-word-entry supervision."""
import itertools
import unittest

import numpy as np

from word_boundary import ctc_boundaries


class BoundaryMarginalTest(unittest.TestCase):
    def test_ctc_boundary_marginal_matches_exhaustive_paths(self):
        p = np.array([[.2, .7, .1], [.5, .3, .2], [.4, .1, .5], [.6, .1, .3]])
        total, entries = 0., np.zeros(len(p))
        for path in itertools.product(range(3), repeat=len(p)):
            collapsed, prev, first_second = [], None, None
            for t, token in enumerate(path):
                if token != prev and token != 0:
                    collapsed.append(token)
                    if len(collapsed) == 2:
                        first_second = t
                prev = token
            if collapsed == [1, 2]:
                mass = np.prod([p[t, token] for t, token in enumerate(path)])
                total += mass
                entries[first_second] += mass
        z, marginal = ctc_boundaries(np.log(p), [1, 2], [0, 1])
        self.assertAlmostEqual(np.exp(z), total)
        np.testing.assert_allclose(marginal, entries/total, atol=1e-12)
        self.assertAlmostEqual(marginal.sum(), 1.)

    def test_repeated_label_requires_a_blank_before_next_word(self):
        p = np.array([[.1, .9], [.8, .2], [.1, .9]])
        z, marginal = ctc_boundaries(np.log(p), [1, 1], [0, 1])
        self.assertAlmostEqual(np.exp(z), .9*.8*.9)
        np.testing.assert_allclose(marginal, [0., 0., 1.])


if __name__ == '__main__':
    unittest.main()

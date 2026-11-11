"""Tiny CPU numerical checks; no recording or practice data."""
import unittest
import numpy as np
from student_heldband import forward_backward, topology, predictors, conditional


class HeldBandTrainingTest(unittest.TestCase):
    def test_complete_chain_and_posteriors_match_exhaustive_paths(self):
        lp = np.array([[-.2, -.8, -1], [-.9, -.3, -.4], [-.1, -.7, -.9]])
        for labels in ([1], [1, 2], [1, 1]):
            ids, stay, one, two = topology(labels, .8)
            paths = []

            def visit(t, s, weight, path):
                weight *= np.exp(lp[t, ids[s]])
                path = path + [s]
                if t == len(lp) - 1:
                    if s >= len(ids) - 2:
                        paths.append((weight, path))
                    return
                for step, probs in enumerate((stay, one, two)):
                    if probs[s] and s + step < len(ids):
                        visit(t + 1, s + step, weight * probs[s], path)

            visit(0, 0, .5, [])
            visit(0, 1, .5, [])
            total = sum(w for w, _ in paths)
            expected = np.zeros_like(lp)
            for w, path in paths:
                for t, s in enumerate(path):
                    expected[t, ids[s]] += w / total
            score, gamma = forward_backward(lp, labels, .8, posterior=True)
            self.assertAlmostEqual(score, np.log(total), places=12)
            np.testing.assert_allclose(gamma, expected, atol=1e-12)
        score, gamma = forward_backward(lp[:2], [1, 1], .8, posterior=True)
        self.assertEqual(score, -np.inf)
        self.assertIsNone(gamma)

    def test_phone_predictors_shrink_to_background_and_use_only_visible_inputs(self):
        rng = np.random.default_rng(0)
        visible = rng.normal(size=200)
        identity = np.tile([-1., 1.], 100)
        z = np.column_stack([visible, 2 * visible + identity])
        weights = np.zeros((len(z), 3))
        weights[identity < 0, 1] = 1
        weights[identity > 0, 2] = 1
        p = predictors(z, weights, [0], [1], .1)
        self.assertEqual(p['phones'][0], p['background'])
        own = conditional(z[identity < 0], p['phones'][1], p).sum()
        rival = conditional(z[identity < 0], p['phones'][2], p).sum()
        self.assertGreater(own, rival)
        # Exported coefficients have exactly one intercept and the declared visible inputs.
        self.assertEqual(np.asarray(p['phones'][1]['weights']).shape, (2, 1))


if __name__ == '__main__':
    unittest.main()

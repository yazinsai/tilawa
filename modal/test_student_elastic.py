"""Small exhaustive numerical checks; no recordings or accelerator needed."""
import math
import unittest
from student_elastic import deformation_prior, joint_chain, warp_spectrum, prepare, fit, score

try:
    import numpy as np
except ImportError:
    np = None


class JointPaths(unittest.TestCase):
    def test_complete_paths_cross_boundary_without_reset(self):
        stay, grid = [.6, .7, .6], [-1, 0, 1]
        initial, transitions = deformation_prior(grid, .5)
        emission = [[[.01*(t+s+w) for w in range(3)] for s in range(6)] for t in range(7)]
        totals = [0., 0.]

        def visit(t, s, w, mass, value):
            value += emission[t][s][w]
            if t == 6:
                if s == 5:
                    totals[0] += mass
                    totals[1] += mass*math.exp(value)
                return
            for following in [s, s+1]:
                if following >= 6:
                    continue
                timing = stay[s % 3] if following == s else 1-stay[s % 3]
                for v in range(3):
                    if math.isfinite(transitions[w][v]):
                        visit(t+1, following, v, mass*timing*math.exp(transitions[w][v]), value)

        for w in range(3):
            visit(0, 0, w, math.exp(initial[w]), 0)
        self.assertAlmostEqual(joint_chain(emission, stay, grid, .5), math.log(totals[1]/totals[0]), 12)

    def test_normalized_constant_emissions(self):
        for grid in [[0], [-2, -1, 0, 1, 2]]:
            values = [[[.2]*len(grid) for _ in range(6)] for _ in range(9)]
            self.assertAlmostEqual(joint_chain(values, [.6, .7, .6], grid, 2), 1.8, 12)

    def test_pinned_monotonic_correspondence(self):
        for d in [-2, -1, 0, 1, 2]:
            x = warp_spectrum(list(range(24)), d)
            self.assertEqual(x[0], 0)
            self.assertEqual(x[-1], 23)
            self.assertTrue(all(a < b for a, b in zip(x, x[1:])))

    @unittest.skipIf(np is None, 'Numerical fitting requires NumPy')
    def test_training_provenance_and_finite_latent_fit(self):
        x = np.random.default_rng(11).normal(size=(12, 24))
        x -= x.mean(1, keepdims=True)
        rows = [dict(kind='none', split='train', excluded=False, speaker=s,
            pronunciation='ab', features=x.tolist()) for s in ['one', 'two', 'three']]
        for invalid in [dict(rows[0], kind='wrong_word'), dict(rows[0], split='held-out'), dict(rows[0], excluded=True)]:
            with self.assertRaises(ValueError):
                prepare([invalid]+rows[1:], 'ab', set())
        with self.assertRaises(ValueError):
            prepare(rows, 'ab', {'one'})
        fitted, history = fit(prepare(rows, 'ab', set()), 'ab', 8, epochs=2)
        self.assertTrue(all(math.isfinite(v) for v in history))
        self.assertTrue(np.isfinite(fitted['means']).all())
        self.assertTrue((np.asarray(fitted['variances']) > 0).all())
        fitted.update(displacements=[-2, -1, 0, 1, 2], regularization=2)
        self.assertTrue(math.isfinite(score(fitted, x, [1, 2])))

    @unittest.skipIf(np is None, 'Vectorized parity requires NumPy')
    def test_vectorized_forward_matches_joint_enumeration(self):
        model = dict(scale=1., means=np.zeros((3, 3, 24)), variances=np.ones((3, 3, 24)),
            stay=[.6, .7, .6], displacements=[-2, -1, 0, 1, 2], regularization=2)
        x = np.zeros((9, 24))
        model['means'][1, 0, 7] = .5
        for elastic in [False, True]:
            grid = model['displacements'] if elastic else [0]
            emissions = []
            for row in x:
                values = []
                for p in [1, 2]:
                    for q in range(3):
                        values.append([sum(-.5*(math.log(2*math.pi)+(v-m)**2)
                            for v, m in zip(row, warp_spectrum(model['means'][p, q], d))) for d in grid])
                emissions.append(values)
            self.assertAlmostEqual(score(model, x, [1, 2], elastic), joint_chain(emissions, model['stay'], grid, 2), 10)


if __name__ == '__main__':
    unittest.main()

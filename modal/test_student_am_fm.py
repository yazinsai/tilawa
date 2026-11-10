"""Tiny numerical checks only; no recording is trained on the host."""
import itertools
import unittest

import numpy as np
from student_am_fm import chain, densities, fit, prepare


class AmFmTest(unittest.TestCase):
    def test_exact_boundaries_and_marginals(self):
        emission = np.array([[.1, -.2], [.3, -.4], [.5, .2], [-.2, .5]])
        value, gamma = chain(emission, .7, True)
        paths = [p for p in itertools.product(range(2), repeat=4) if p[0] == 0 and p[-1] == 1 and all(b-a in [0, 1] for a, b in zip(p, p[1:]))]
        logs = np.array([sum(emission[t, p[t]] for t in range(4)) + sum(np.log(.7 if a == b else .3) for a, b in zip(p, p[1:])) for p in paths])
        prior = np.log(len(paths)) + 2*np.log(.7) + np.log(.3)
        partition = np.logaddexp.reduce(logs)
        self.assertAlmostEqual(value, partition-prior)
        exact = sum(np.exp(v-partition)*np.eye(2)[list(p)] for v, p in zip(logs, paths))
        np.testing.assert_allclose(gamma, exact)

    def test_missing_frequency_and_matched_control(self):
        m = dict(bands=[0], means=[[0, 0], [0, 0]], variances=[[1, 1], [1, 1]], covariance=[[0], [.8]])
        x = np.array([[[.4, 20]], [[.4, -20]]])
        e, *_ = densities(m, x, np.zeros((2, 1)), [1])
        independent, *_ = densities(m, x, np.zeros((2, 1)), [1], False)
        np.testing.assert_allclose(e, independent)
        self.assertAlmostEqual(e[0, 0], e[1, 0])

    def test_provenance_and_finite_em(self):
        samples = []
        for speaker in ['one', 'two', 'three']:
            features = np.zeros((12, 24))
            features[:, 0] = np.linspace(-.5, .5, 12)
            features[:, 1] = features[:, 0]*.8
            features[:, 2::3] = .9
            samples.append(dict(kind='none', split='train', excluded=False, speaker=speaker, pronunciation='ab', features=features.tolist()))
        rows = prepare(samples, 'ab', set())
        m, history = fit(rows, 'ab', [0], 16, epochs=2)
        self.assertTrue(np.isfinite(history).all())
        var = np.array(m['variances']).reshape(3, 1, 2)
        self.assertTrue((np.array(m['covariance'])**2 < var[:, :, 0]*var[:, :, 1]).all())
        with self.assertRaises(ValueError):
            prepare(samples, 'ab', {'one'})
        samples[0]['kind'] = 'substituted-word'
        with self.assertRaises(ValueError):
            prepare(samples, 'ab', set())


if __name__ == '__main__':
    unittest.main()

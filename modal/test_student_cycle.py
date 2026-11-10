"""Small synthetic numerical tests; no host-side recording training."""
import itertools
import unittest

import numpy as np
from student_cycle import chain, densities, fit, prepare


class CycleTest(unittest.TestCase):
    def test_exact_boundaries_and_occupancies(self):
        emission = np.array([[.1, -.2], [.3, -.4], [.5, .2], [-.2, .5]])
        value, gamma = chain(emission, .7, True)
        paths = [p for p in itertools.product(range(2), repeat=4) if p[0] == 0 and p[-1] == 1 and
                 all(b-a in [0, 1] for a, b in zip(p, p[1:]))]
        logs = np.array([sum(emission[t, p[t]] for t in range(4)) +
                         sum(np.log(.7 if a == b else .3) for a, b in zip(p, p[1:])) for p in paths])
        prior = np.log(len(paths)) + 2*np.log(.7) + np.log(.3)
        partition = np.logaddexp.reduce(logs)
        self.assertAlmostEqual(value, partition-prior)
        exact = sum(np.exp(v-partition)*np.eye(2)[list(p)] for v, p in zip(logs, paths))
        np.testing.assert_allclose(gamma, exact)

    def test_unobserved_phase_and_matched_pooled_control(self):
        m = dict(means=np.zeros((2, 16)).tolist(), variances=np.ones((2, 16)).tolist())
        x = np.zeros((2, 16))
        x[:, 2:] = np.array([1000, -1000])[:, None]
        conditioned = densities(m, x, np.zeros(2), [1])
        control = densities(m, x, np.ones(2), [1], False)
        np.testing.assert_allclose(conditioned, control)
        self.assertAlmostEqual(conditioned[0, 0], conditioned[1, 0])

    def test_provenance_and_finite_latent_boundary_fit(self):
        samples = []
        for speaker in ['one', 'two', 'three']:
            features = np.zeros((12, 17))
            features[:, :16] = np.linspace(-.5, .5, 12)[:, None]
            features[:, 16] = 1
            samples.append(dict(kind='none', split='train', excluded=False, speaker=speaker,
                                pronunciation='ab', features=features.tolist()))
        rows = prepare(samples, 'ab', [])
        m, history = fit(rows, 'ab', 16, epochs=2)
        self.assertTrue(np.isfinite(history).all())
        self.assertTrue((np.array(m['variances']) > 0).all())
        self.assertTrue((np.array(m['cycleCounts']) <= np.array(m['counts']) + 1e-6).all())
        with self.assertRaises(ValueError):
            prepare(samples, 'ab', ['one'])
        samples[0]['kind'] = 'substituted-word'
        with self.assertRaises(ValueError):
            prepare(samples, 'ab', [])


if __name__ == '__main__':
    unittest.main()

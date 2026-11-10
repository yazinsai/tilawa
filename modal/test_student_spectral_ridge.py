"""Small synthetic numerical and provenance checks; no recording training."""
import itertools
import unittest
import numpy as np
from student_spectral_ridge import chain, fit, prepare, emissions


class RidgeTests(unittest.TestCase):
    def test_chain_matches_enumeration_and_state_marginals(self):
        e = np.array([[.1, .2], [.3, -.1], [-.2, .7], [.5, .9]])
        paths = [[0]+list(p) for p in itertools.product([0, 1], repeat=3)
                 if p[-1] == 1 and all(a <= b for a, b in zip((0,)+p[:-1], p))]
        weights, priors = [], []
        for path in paths:
            prior = sum(np.log(.8 if a == b else .2) for a, b in zip(path, path[1:]))
            priors.append(prior)
            weights.append(prior+sum(e[t, s] for t, s in enumerate(path)))
        value, gamma = chain(e, .8, True)
        self.assertAlmostEqual(value, np.logaddexp.reduce(weights)-np.logaddexp.reduce(priors))
        expected = np.zeros_like(e)
        for path, weight in zip(paths, weights):
            for t, s in enumerate(path):
                expected[t, s] += np.exp(weight-np.logaddexp.reduce(weights))
        np.testing.assert_allclose(gamma, expected)

    def test_prepare_rejects_wrong_split_and_scripted_transcripts(self):
        rows = [dict(source='help-train', kind='none', split='train', excluded=False, speaker=str(i),
                     pronunciation='ab', featuresByRadius={'1': np.zeros((4, 56)).tolist()}) for i in range(3)]
        self.assertEqual(len(prepare(rows, 'ab', set(), 1)), 3)
        for update in [dict(kind='wrong_word'), dict(split='eval'), dict(excluded=True), dict(source='practice')]:
            with self.assertRaises(ValueError):
                prepare([dict(r, **update) for r in rows], 'ab', set(), 1)
        with self.assertRaises(ValueError):
            prepare(rows, 'ab', {'0'}, 1)

    def test_em_is_finite_and_control_marginalizes_only_connections(self):
        rows = [dict(x=np.array([[i]*32+[2]*24, [i]*32+[2]*24, [i+1]*32+[0]*24, [i+1]*32+[0]*24], dtype=float),
                     ids=[1, 2], speaker=str(i)) for i in range(3)]
        m, history = fit(rows, 'ab', 8, epochs=2)
        self.assertTrue(np.isfinite(history).all())
        self.assertTrue((np.array(m['variances']) > 0).all())
        x = (rows[0]['x']-m['center'])/m['scale']
        control = emissions(m, x, [1, 2], False)
        changed = x.copy()
        changed[:, 32:] += 10
        np.testing.assert_equal(control, emissions(m, changed, [1, 2], False))
        self.assertFalse(np.array_equal(emissions(m, x, [1, 2]), emissions(m, changed, [1, 2])))


if __name__ == '__main__':
    unittest.main()

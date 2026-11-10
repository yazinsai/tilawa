"""Small numerical checks for the acoustic trainer; no recording data."""
import itertools
import unittest
import numpy as np
from student_intraphone import chain, fit, prepare


class IntraphoneTrainingTest(unittest.TestCase):
    def test_forward_backward_equals_enumerated_boundaries(self):
        e = np.array([[0., -3., -4.], [.2, .1, -2.], [-1., .4, .1], [-2., -.1, .3]])
        stay = [.6, .8, .5]
        paths, weights, priors = [], [], []
        for middle in itertools.product(range(3), repeat=2):
            path = [0, *middle, 2]
            if any(b-a not in [0, 1] for a, b in zip(path, path[1:])):
                continue
            prior = sum(np.log(stay[a] if a == b else 1-stay[a]) for a, b in zip(path, path[1:]))
            paths.append(path)
            priors.append(prior)
            weights.append(prior + sum(e[t, s] for t, s in enumerate(path)))
        mass = np.logaddexp.reduce(weights)
        value, gamma, selfc, stepc = chain(e, stay, True)
        self.assertAlmostEqual(value, mass-np.logaddexp.reduce(priors), places=12)
        expected = np.zeros_like(e)
        for path, weight in zip(paths, weights):
            for t, s in enumerate(path):
                expected[t, s] += np.exp(weight-mass)
        np.testing.assert_allclose(gamma, expected, atol=1e-12)
        np.testing.assert_allclose(gamma.sum(1), 1, atol=1e-12)
        self.assertAlmostEqual(float(selfc.sum()+stepc.sum()), 4)

    def test_training_provenance_is_enforced(self):
        rows = [dict(kind='none', split='train', excluded=False, speaker=str(i),
                     features=np.zeros((9, 20)).tolist(), pronunciation='a') for i in range(3)]
        self.assertEqual(len(prepare(rows, 'a', set())), 3)
        for key, value in [('kind', 'wrong_word'), ('split', 'exam'), ('excluded', True)]:
            bad = [dict(r) for r in rows]
            bad[0][key] = value
            with self.assertRaises(ValueError):
                prepare(bad, 'a', set())
        with self.assertRaises(ValueError):
            prepare(rows, 'a', {'0'})

    def test_em_keeps_latent_phase_posteriors_and_positive_parameters(self):
        rows = [dict(x=np.repeat(np.array([-1., 0., 1.])[:, None], 20, axis=1).repeat(3, axis=0),
                     ids=[1], speaker=str(i)) for i in range(3)]
        model, history = fit(rows, 'a', 8, epochs=2)
        self.assertEqual(len(history), 2)
        self.assertTrue(np.isfinite(history).all())
        self.assertTrue((np.array(model['variances']) > 0).all())
        self.assertTrue(all(0 < p < 1 for p in model['stay']))
        self.assertGreater(model['counts'][1], 0)
        self.assertEqual(np.array(model['means']).shape, (2, 3, 20))


if __name__ == '__main__':
    unittest.main()

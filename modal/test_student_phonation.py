"""Small synthetic numerical checks; no audio corpus is fitted here."""
import itertools
import unittest
import numpy as np
from student_phonation import chain, fit, prepare


class PhonationTests(unittest.TestCase):
    def test_forward_backward_exhaustive_product_paths(self):
        emission = np.arange(36).reshape(4, 3, 3)*.017-.8
        matrix = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
        initial = np.array([.2, .5, .3])
        for switching in [False, True]:
            mass, prior, occupancy = 0., 0., np.zeros_like(emission)
            transitions = np.zeros((3, 3))
            for steps in itertools.product([0, 1], repeat=3):
                phone = np.r_[0, np.cumsum(steps)]
                if phone[-1] != 2:
                    continue
                for z in itertools.product(range(3), repeat=4):
                    weight = initial[z[0]]*.5**3
                    for t in range(1, 4):
                        weight *= matrix[z[t-1], z[t]] if switching else float(z[t-1] == z[t])
                    prior += weight
                    weight *= np.exp(sum(emission[t, phone[t], z[t]] for t in range(4)))
                    mass += weight
                    for t in range(4):
                        occupancy[t, phone[t], z[t]] += weight
                    for t in range(3):
                        transitions[z[t], z[t+1]] += weight
            value, gamma, counts = chain(emission, matrix, initial, True, switching)
            self.assertAlmostEqual(value, np.log(mass/prior), places=12)
            np.testing.assert_allclose(gamma, occupancy/mass, atol=1e-12)
            np.testing.assert_allclose(counts, transitions/mass, atol=1e-12)
            np.testing.assert_allclose(gamma.sum((1, 2)), np.ones(4), atol=1e-12)

    def test_training_provenance(self):
        row = dict(kind='none', split='train', excluded=False, speaker='a', pronunciation='بت', features=np.zeros((8, 23)).tolist())
        rows = [dict(row, speaker=s) for s in ['a', 'b', 'c']]
        self.assertEqual(len(prepare(rows, 'بت', set())), 3)
        for key, value in [('kind', 'wrong_word'), ('split', 'eval'), ('excluded', True)]:
            with self.assertRaises(ValueError):
                prepare([dict(r, **{key: value}) for r in rows], 'بت', set())
        with self.assertRaises(ValueError):
            prepare(rows, 'بت', {'a'})

    def test_toy_em_has_finite_shared_parameters(self):
        rng = np.random.default_rng(7)
        rows = [dict(x=rng.normal(size=(12, 23)), ids=[1, 2], speaker=str(i)) for i in range(3)]
        model, history = fit(rows, 'بت', 8, 8, epochs=2)
        self.assertTrue(np.isfinite(history).all())
        self.assertEqual(np.asarray(model['offsets']).shape, (3, 20))
        self.assertTrue((np.asarray(model['variances']) > 0).all())
        np.testing.assert_allclose(np.asarray(model['transition']).sum(1), np.ones(3))


if __name__ == '__main__':
    unittest.main()

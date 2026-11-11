import itertools
import unittest
import numpy as np
from student_regularity import chain, prepare


class RegularityTrainingTest(unittest.TestCase):
    def test_complete_boundary_mass_and_occupancies(self):
        e = np.array([[.2, -.4], [.1, .3], [-.5, .6], [.2, .8]])
        stay = .7
        paths = [p for p in itertools.product(range(2), repeat=4)
            if p[0] == 0 and p[-1] == 1 and all(b-a in [0, 1] for a, b in zip(p, p[1:]))]
        weights = []
        priors = []
        for path in paths:
            prior = sum(np.log(stay if a == b else 1-stay) for a, b in zip(path, path[1:]))
            priors.append(prior)
            weights.append(prior+sum(e[t, s] for t, s in enumerate(path)))
        mass = np.logaddexp.reduce(weights)
        value, gamma = chain(e, stay, True)
        self.assertAlmostEqual(value, mass-np.logaddexp.reduce(priors), places=12)
        expected = np.zeros_like(e)
        for path, weight in zip(paths, weights):
            for t, s in enumerate(path):
                expected[t, s] += np.exp(weight-mass)
        np.testing.assert_allclose(gamma, expected)

    def test_rejects_scripted_or_held_out_supervision(self):
        features = {g: [dict(regularity=[0]*12, energy=[0]*12)]*3 for g in ['fine', 'wide']}
        records = [dict(kind='none', split='train', excluded=False, speaker=str(i), pronunciation='a', features=features)
            for i in range(3)]
        self.assertEqual(len(prepare(records, 'a', set())), 3)
        for field, value in [('kind', 'substituted_word'), ('split', 'eval'), ('excluded', True)]:
            with self.assertRaises(ValueError):
                prepare([{**records[0], field: value}, *records[1:]], 'a', set())
        with self.assertRaises(ValueError):
            prepare(records, 'a', {'0'})


if __name__ == '__main__':
    unittest.main()

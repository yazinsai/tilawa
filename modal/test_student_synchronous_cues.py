import itertools
import unittest
import numpy as np
from scipy.special import logsumexp
from student_synchronous_cues import attributes, chain, class_map, emissions, fit, graph, prepare


class SynchronousCuesTest(unittest.TestCase):
    def test_boundaries_and_posteriors_match_exhaustive_enumeration(self):
        ids, stay = [1, 2], .7
        labels, transitions = graph(ids, stay)
        e = np.arange(4*len(labels)).reshape(4, len(labels))*.03
        paths, weights, priors = [], [], []
        for path in itertools.product(range(len(labels)), repeat=4):
            if path[0] not in [0, 1] or path[-1] not in [len(labels)-2, len(labels)-1]:
                continue
            prior = -np.log(2)+sum(transitions[a, b] for a, b in zip(path, path[1:]))
            if not np.isfinite(prior):
                continue
            paths.append(path)
            weights.append(prior+sum(e[t, s] for t, s in enumerate(path)))
            priors.append(prior)
        value, gamma = chain(e, ids, stay, True)
        expected = np.zeros_like(e)
        for path, weight in zip(paths, weights):
            for t, s in enumerate(path):
                expected[t, s] += np.exp(weight-logsumexp(weights))
        self.assertAlmostEqual(value, logsumexp(weights)-logsumexp(priors), places=11)
        np.testing.assert_allclose(gamma, expected, atol=1e-12)
        np.testing.assert_allclose(gamma.sum(1), 1, atol=1e-12)

    def test_repeated_articulations_keep_background_separation(self):
        _, tr = graph([1, 1], .7)
        self.assertFalse(np.isfinite(tr[1, 3]))
        _, tr = graph([1, 2], .7)
        self.assertTrue(np.isfinite(tr[1, 3]))
        mapping, _ = class_map('تد')
        self.assertEqual(mapping[1][:2], mapping[2][:2])
        self.assertNotEqual(mapping[1][2], mapping[2][2])

    def test_training_provenance_and_small_fit(self):
        rng = np.random.default_rng(42)
        rows = [dict(speaker=str(i), kind='none', split='train', excluded=False,
                     pronunciation='بت', features=rng.normal(size=(12, 20)).tolist()) for i in range(3)]
        prepared = prepare(rows, 'بت', set())
        m, _ = fit(prepared, 'بت', 8, .7, epochs=2)
        self.assertTrue(all(np.isfinite(np.array(head)).all() for head in m['means']))
        self.assertTrue(all((np.array(head) > 0).all() for head in m['variances']))
        labels, _ = graph(prepared[0]['ids'], m['stay'])
        x = (prepared[0]['x']-m['center'])/m['scale']
        e = emissions(m, x, labels)
        self.assertTrue(np.isfinite(chain(e, prepared[0]['ids'], m['stay'])))
        for change in [dict(split='eval'), dict(kind='substituted_word'), dict(excluded=True)]:
            invalid = [dict(r, **change) for r in rows]
            with self.assertRaises(ValueError):
                prepare(invalid, 'بت', set())
        with self.assertRaises(ValueError):
            prepare(rows, 'بت', {'1'})
        self.assertEqual(attributes('ن'), ['coronal', 'nasal', 'voiced'])


if __name__ == '__main__':
    unittest.main()

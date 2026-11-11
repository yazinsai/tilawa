"""Small mathematical and training-isolation checks; run on the CPU service."""
import itertools
import unittest
import numpy as np
from student_reverse_predictive import chain, emission, fit, normalization, prepare, topology


class PredictiveTests(unittest.TestCase):
    def test_complete_chain_matches_enumeration(self):
        for ids in ([1, 2], [1, 1]):
            labels, step, skip = topology(ids)
            e = np.arange(5*len(labels)).reshape(5, -1)*.03
            scores, priors = [], []
            for path in itertools.product(range(len(labels)), repeat=5):
                if path[0] not in (0, 1) or path[-1] < len(labels)-2:
                    continue
                p = -np.log(2)
                for a, b in zip(path, path[1:]):
                    if b == a:
                        p += np.log(.8)
                    elif b == a+1:
                        p += step[b]
                    elif b == a+2 and skip[b]:
                        p += np.log(.1)
                    else:
                        break
                else:
                    priors.append(p)
                    scores.append(p+sum(e[t, s] for t, s in enumerate(path)))
            self.assertAlmostEqual(chain(e, ids), np.logaddexp.reduce(scores)-np.logaddexp.reduce(priors))
            _, gamma, _ = chain(e, ids, True)
            np.testing.assert_allclose(gamma.sum(1), 1, atol=1e-12)

    def test_expanded_gaussian_matches_residual(self):
        rng = np.random.default_rng(5)
        x, z, b, a = (rng.normal(size=(4, 80)) for _ in range(4))
        v = rng.uniform(.5, 2, (4, 80))
        m = dict(intercept=b, coefficient=a, variance=v)
        exact = -.5*(np.log(2*np.pi*v)[None]+(x[:, None]-b[None]-a[None]*z[:, None])**2/v[None]).sum(2)
        np.testing.assert_allclose(emission(m, x, z, np.arange(4)), exact, atol=1e-10)

    def test_training_guards(self):
        row = dict(kind='none', split='train', excluded=False, speaker='a', pronunciation='a', features=[[0]*80]*4)
        for bad in (dict(kind='wrong_word'), dict(split='eval'), dict(excluded=True)):
            with self.assertRaises(ValueError):
                prepare([dict(row, **bad)], 'a', set())
        with self.assertRaises(ValueError):
            prepare([row], 'a', {'a'})

    def test_equal_capacity_deterministic_fit_and_fold_normalization(self):
        rows = [dict(x=np.tile(np.arange(12)[:, None], (1,80))*.01, ids=[1,2], speaker=s) for s in 'abc']
        train, valid, center, _ = normalization(rows[:2], [dict(rows[2], x=rows[2]['x']+100)])
        self.assertLess(center.max(), 1)
        self.assertGreater(valid[0]['x'].min(), 100)
        f, _ = fit(train, 3, 8, False, epochs=2)
        r, _ = fit(train, 3, 8, True, epochs=2)
        for key in ('intercept', 'coefficient', 'variance'):
            self.assertEqual(np.array(f[key]).shape, np.array(r[key]).shape)
            self.assertTrue(np.isfinite(r[key]).all())


if __name__ == '__main__':
    unittest.main()

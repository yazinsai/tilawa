import itertools
import math
import unittest
import numpy as np
from student_occurrence_mixture import boundary, fit


class MixtureTest(unittest.TestCase):
    def test_enumerated_components_and_boundaries(self):
        v = np.random.default_rng(1).normal(size=(4, 3, 2))
        w = np.array([[.4, .6], [.3, .7], [.8, .2]])
        for ids in ([1], [2], [1, 2], [1, 1]):
            labels = [0]+[v for p in ids for v in (p, 0)]
            for tied in (True, False):
                totals = [0., 0.]
                def visit(path, weight):
                    s = path[-1]
                    if len(path) == len(v):
                        if s < len(labels)-2:
                            return
                        totals[1] += weight
                        groups = []
                        for state, times in itertools.groupby(enumerate(path), key=lambda pair: pair[1]):
                            times = [t for t, _ in times]
                            p = labels[state]
                            if tied and p:
                                groups.append(sum(w[p, k]*math.exp(sum(v[t, p, k] for t in times)) for k in range(2)))
                            else:
                                groups.extend(sum(w[p, k]*math.exp(v[t, p, k]) for k in range(2)) for t in times)
                        totals[0] += weight*math.prod(groups)
                        return
                    edges = [s]+([s+1] if s+1 < len(labels) else [])+([s+2] if s+2 < len(labels) and labels[s+2] and labels[s+2] != labels[s] else [])
                    for n in edges:
                        visit(path+[n], weight/len(edges))
                visit([0], 1.)
                visit([1], 1.)
                expected = math.log(totals[0]/totals[1])
                self.assertAlmostEqual(boundary(v, w, ids, tied), expected, places=12)
                if tied:
                    score, posts, entries = boundary(v, w, ids, True, True)
                    np.testing.assert_allclose(posts.sum((1, 2)), 1.)
                    self.assertAlmostEqual(entries[1:].sum(), len(ids), places=10)
                    # Occupancy is the derivative of the joint partition w.r.t. emissions.
                    epsilon = 1e-5
                    altered = v.copy();altered[1, 1, 0] += epsilon
                    derivative = (boundary(altered, w, ids)-score)/epsilon
                    self.assertAlmostEqual(derivative, posts[1, 1, 0], places=5)

    def test_identical_single_component(self):
        v = np.random.default_rng(2).normal(size=(6, 3, 1));w = np.ones((3, 1))
        for ids in ([1], [1, 1], [1, 2]):
            self.assertAlmostEqual(boundary(v, w, ids, True), boundary(v, w, ids, False), places=12)

    def test_small_training_fit_has_finite_shared_emissions(self):
        rng = np.random.default_rng(4)
        rows = []
        for speaker in ('a', 'b', 'c'):
            probabilities = np.full((24, 4), .01)
            probabilities[:, 0] = .95
            probabilities[3:9, 1] = .97
            probabilities[14:20, 2] = .97
            probabilities /= probabilities.sum(1, keepdims=True)
            rows.append(dict(speaker=speaker, ids=[1, 2], features=rng.normal(size=(24, 16)),
                log_probs=np.log(probabilities)))
        model = fit(rows, 16., 4)
        for key in ('center', 'projection', 'means', 'variances', 'weights'):
            self.assertTrue(np.isfinite(model[key]).all())
        self.assertTrue((model['variances'] > 0).all())
        np.testing.assert_allclose(model['weights'].sum(1), 1.)
        self.assertEqual(model['count'][3], 0.)


if __name__ == '__main__':
    unittest.main()

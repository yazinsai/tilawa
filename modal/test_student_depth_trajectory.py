import itertools
import unittest
import numpy as np
from student_depth_trajectory import boundary, topology, update


class MarginalTests(unittest.TestCase):
    def test_exhaustive_ctc_paths_and_normalization(self):
        values = np.log(np.array([[.3,.6,.1],[.4,.2,.4],[.2,.3,.5],[.5,.1,.4]]))
        for ids in ([1],[1,2],[1,1]):
            labels, edges = topology(ids)
            paths = [[0],[1]]
            for _ in range(1,len(values)):
                paths = [p+[n] for p in paths for n in edges[p[-1]]]
            paths = [p for p in paths if p[-1] >= len(labels)-2]
            weights = [np.prod([1/len(edges[s]) for s in p[:-1]]) for p in paths]
            likelihood = sum(w*np.exp(sum(values[t,labels[s]] for t,s in enumerate(p))) for p,w in zip(paths,weights))
            expected = np.log(likelihood/sum(weights)) if paths else -np.inf
            self.assertAlmostEqual(boundary(values,ids),expected)

    def test_fractional_occupancy_sums_to_one(self):
        _, w = boundary(np.log(np.full((5,3),1/3)),[1,2],True)
        np.testing.assert_allclose(w.sum(1),1)

    def test_conditioning_does_not_reward_extra_paths(self):
        values = np.full((7,3),-.75)
        for ids in ([1],[1,2],[1,1]):
            self.assertAlmostEqual(boundary(values,ids),-.75*7)

    def test_covariance_regularization_remains_finite(self):
        phones = update(np.array([2.,1.]),np.zeros((2,16)),np.zeros((2,16,16)),4)
        for e in phones:
            self.assertTrue(np.isfinite(e['precision']).all())
            self.assertTrue(np.isfinite(e['logDet']))


if __name__ == '__main__':
    unittest.main()

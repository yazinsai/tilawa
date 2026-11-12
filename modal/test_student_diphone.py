import unittest
from itertools import product
import numpy as np
from student_diphone_phone import occupancies, pair_targets, pair_ids, phone_rows


class DiphoneTargets(unittest.TestCase):
    def test_latent_middle_articulation_is_shared(self):
        # A certain A-B-C path. Both neighboring pairs receive B's occupancy.
        states=np.zeros((3,7),dtype='float32')
        states[np.arange(3),[1,3,5]]=1
        y=pair_targets(states,[1,2,3],3)
        ab,bc=pair_ids([1,2,3],3)
        np.testing.assert_array_equal(y[:,[ab,bc]],[[1,0],[.5,.5],[0,1]])
        np.testing.assert_allclose(y.sum(1),1)

    def test_ctc_boundaries_are_marginal_not_greedy(self):
        found=occupancies(np.log(np.full((5,3),1/3)),[1,2])
        self.assertIsNotNone(found)
        targets,states=found
        self.assertTrue(np.any((states>0)&(states<1)))
        np.testing.assert_allclose(targets.sum(1),1,atol=1e-6)
        np.testing.assert_allclose(pair_targets(states,[1,2],2).sum(1),1,atol=1e-6)
        self.assertIsNone(occupancies(np.log(np.full((1,3),1/3)),[1,2]))

    def test_grouped_control_preserves_mass(self):
        got=np.exp(phone_rows(np.log([[.1,.2,.3,.15,.25]]),2))
        np.testing.assert_allclose(got,[[.1,.5,.4]])

    def test_soft_occupancies_match_exhaustive_acoustic_paths(self):
        probability=np.array([[.2,.7,.1],[.5,.3,.2],[.3,.1,.6],[.6,.15,.25]])
        for ids in ([1,2],[1,1]):
            expected=np.zeros_like(probability)
            mass=0.
            for path in product(range(3),repeat=4):
                collapsed=[p for t,p in enumerate(path) if p and (t==0 or p!=path[t-1])]
                if collapsed!=ids:
                    continue
                weight=float(np.prod(probability[np.arange(4),path]))
                mass+=weight
                expected[np.arange(4),path]+=weight
            actual,_=occupancies(np.log(probability),ids)
            np.testing.assert_allclose(actual,expected/mass,atol=1e-6)


if __name__=='__main__':
    unittest.main()

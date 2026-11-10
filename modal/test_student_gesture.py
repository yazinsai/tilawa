import itertools
import unittest
import numpy as np
from student_gesture import signature, control, lattice, prepare


class GestureTests(unittest.TestCase):
    def test_collinear_tempo_invariance(self):
        a=np.array([[0,0],[1,0],[1,1]],float)
        b=np.array([[0,0],[.1,0],[.3,0],[1,0],[1,1]],float)
        for order in [2,3]:
            np.testing.assert_allclose(signature(a,order),signature(b,order),atol=1e-12)
            self.assertFalse(np.allclose(control(a,order),control(b,order)))

    def test_exhaustive_boundary_sum(self):
        e=np.arange(2*4*4,dtype=float).reshape(2,4,4)*-.01
        # Three optional-silence / phone / optional-silence states cover four frames.
        values=[]
        for lengths in itertools.product(range(4),repeat=3):
            if lengths[1]==0 or sum(lengths)!=4:
                continue
            a=0; value=0
            for p,length in zip([0,1,0],lengths):
                if length: value+=e[p,a,length]
                a+=length
            values.append(value)
        value,w=lattice(e,[1],True)
        self.assertAlmostEqual(value,np.log(np.exp(values).mean()),places=12)
        self.assertAlmostEqual(w[1].sum(),1,places=12)

    def test_training_rejects_scripts_and_heldout_speakers(self):
        r=dict(kind='substituted-word',split='train',excluded=False,speaker='s',
            pronunciation='a',features=[[0]*80])
        with self.assertRaises(ValueError): prepare([r],'a',set())
        r['kind']='none'
        with self.assertRaises(ValueError): prepare([r],'a',{'s'})


if __name__=='__main__': unittest.main()

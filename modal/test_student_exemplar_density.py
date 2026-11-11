"""Small exact checks of transcript occupancies and training exclusion."""
import itertools
import unittest
import numpy as np
from student_exemplar_density import ctc, prepare, fit, likelihood


class ExemplarTest(unittest.TestCase):
    def test_marginal_uses_all_complete_paths(self):
        p=np.array([[.2,.5,.3],[.4,.2,.4],[.3,.6,.1]])
        z,gamma=ctc(np.log(p),[1,1],True)
        weights=np.zeros_like(p)
        total=0
        for path in itertools.product(range(3),repeat=3):
            collapsed=[v for i,v in enumerate(path) if v and (not i or v!=path[i-1])]
            if collapsed!=[1,1]:continue
            w=np.prod([p[t,v] for t,v in enumerate(path)])
            total+=w
            for t,v in enumerate(path):weights[t,v]+=w
        self.assertAlmostEqual(np.exp(z),total)
        np.testing.assert_allclose(gamma,weights/total,atol=1e-12)
        np.testing.assert_allclose(gamma.sum(1),1,atol=1e-12)

    def test_training_requires_none_and_training_speakers(self):
        bad=dict(source='help-train',kind='substituted_word',split='train',excluded=False,speaker='a')
        with self.assertRaises(ValueError):prepare([bad])
        for key,value in [('kind','repeat'),('split','eval'),('excluded',True),('source','practice')]:
            with self.assertRaises(ValueError):prepare([{**bad,'kind':'none',key:value}])

    def test_class_frequency_correction(self):
        rows=[dict(bank=np.array([[-1.],[1.]]),soft=np.array([[.9,.1],[.9,.1]])) for _ in range(3)]
        model=fit(rows,.1,2,.01)
        lp=np.log(np.array([[.2,.4,.4]]))
        output=likelihood(model,np.array([[0.]]),lp)
        self.assertAlmostEqual(output[0,1],output[0,2])
        self.assertEqual(output[0,0],lp[0,0])


if __name__=='__main__':unittest.main()

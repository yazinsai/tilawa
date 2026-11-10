"""Small numerical and provenance checks; no recordings or practice input."""
import math
import unittest
import numpy as np
from student_micropitch import chain, fit, prepare, residuals, states


class MicropitchTest(unittest.TestCase):
    def test_contour(self):
        frames=[dict(logPitch=math.log(150)+i*.0008,reliability=.99) for i in range(150)]
        self.assertAlmostEqual(residuals(frames,.35,.8)[70,0],0.,places=10)
        frames[70]['logPitch']+=.04
        self.assertAlmostEqual(residuals(frames,.35,.8)[70,0],.04,places=10)

    def test_boundary_sum(self):
        ss=states('بَت');m=dict(scale=.02,variance=1.,stay=[.65,.9,.65],means={s[0]:0. for s in ss},control={s[1]:0. for s in ss})
        x=np.array([[i*.002,1] for i in range(7)])
        m['means'][ss[1][0]]=.7
        totals=[0.,0.]
        def paths(path):
            if len(path)==len(x):
                if path[-1]!=len(ss)-1:return
                prior=likelihood=1.
                for t,s in enumerate(path):
                    if t: prior*=m['stay'][ss[s][2]] if s==path[t-1] else 1-m['stay'][ss[path[t-1]][2]]
                    delta=x[t,0]/m['scale']-m['means'][ss[s][0]]
                    likelihood*=math.exp(-.5*(math.log(2*math.pi)+delta*delta))/m['scale']
                totals[0]+=prior*likelihood;totals[1]+=prior;return
            paths(path+[path[-1]])
            if path[-1]+1<len(ss):paths(path+[path[-1]+1])
        paths([0])
        value,gamma=chain(m,x,'بَت',True,True)
        self.assertAlmostEqual(value,math.log(totals[0]/totals[1]),places=10)
        np.testing.assert_allclose(gamma.sum(1),1.,atol=1e-10)

    def test_provenance_and_fit(self):
        rows=[dict(speaker=str(i),kind='none',split='train',excluded=False,pronunciation='بَت',
                   pitch=[dict(logPitch=math.log(150)+.01*math.sin(j/4),reliability=.99) for j in range(100)]) for i in range(3)]
        self.assertEqual(len(prepare(rows,'بَت',set())),3)
        with self.assertRaises(ValueError):prepare(rows,'بَت',{'1'})
        with self.assertRaises(ValueError):prepare([dict(rows[0],kind='wrong_word')]+rows[1:],'بَت',set())
        model,_=fit(rows,'بَت','toy',.35,.8,16.,epochs=1)
        self.assertTrue(all(math.isfinite(v) for v in model['means'].values()))
        self.assertGreater(model['counts']['َ:0:ب'],0)


if __name__=='__main__':unittest.main()

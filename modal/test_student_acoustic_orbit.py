"""Small mathematical checks; actual fitting and parity run on Modal."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('orbit',Path(__file__).with_name('student_acoustic_orbit.py'))
orbit=importlib.util.module_from_spec(spec);spec.loader.exec_module(orbit)

class OrbitTests(unittest.TestCase):
    def test_boundary_matches_all_complete_paths(self):
        values=np.array([[.1,.3,.2],[.5,.4,.1],[.2,.8,.4],[.4,.2,.7]])
        for ids in ([1],[2],[1,2],[1,1]):
            labels,edges=orbit.topology(ids)
            result=[]
            def visit(t,s,weight,score):
                score+=values[t,labels[s]]
                if t==len(values)-1:
                    if s>=len(labels)-2:result.append((weight,score))
                    return
                for k in edges[s]:visit(t+1,k,weight/len(edges[s]),score)
            visit(0,0,1.,0.);visit(0,1,1.,0.)
            expected=np.log(sum(w*np.exp(s) for w,s in result)/sum(w for w,s in result))
            self.assertAlmostEqual(orbit.boundary(values,ids),expected,places=12)
            _,occupancy=orbit.boundary(values,ids,True)
            np.testing.assert_allclose(occupancy.sum(1),1.)
    def test_full_pronunciation_and_repeat_separation(self):
        self.assertEqual(orbit.boundary(np.zeros((1,3)),[1,2]),-np.inf)
        self.assertEqual(orbit.boundary(np.zeros((2,3)),[1,1]),-np.inf)
        self.assertAlmostEqual(orbit.boundary(np.zeros((3,3)),[1,1]),0.)
    def test_local_query_changes_the_discount_direction(self):
        model=dict(featureCenter=[0,0],projection=[[1,0],[0,1]],means=[[9,9],[0,0],[1,2]],counts=[1,1,1],
            globalPrecision=[[1,0],[0,1]],regions=[dict(center=[0,2],precision=[[1,0],[0,.1]]),
            dict(center=[20,2],precision=[[.1,0],[0,1]])],neighborhoodScale=1.,temperature=1.)
        local=orbit.emissions(model,np.array([[0,2],[20,2]]),'local')
        self.assertGreater(local[0,1],local[0,2]);self.assertLess(local[1,1],local[1,2])
        for control in ('euclidean','global'):
            values=orbit.emissions(model,np.array([[0,2]]),control)
            self.assertLess(values[0,1],values[0,2])
        np.testing.assert_allclose(np.exp(local).sum(1),1.)
    def test_stream_clock_and_cache_without_primary_mutation(self):
        manifest=dict(tokens='ab',shift=32,window=45,fbank=dict(mels=2,padValue=-23.),states=[dict(name='cache',shape=[1],dtype='float32')])
        metadata=dict(featureOutput='h',width=2)
        class Backend:
            calls=0
            def run(self,names,feeds):
                np.testing.assert_array_equal(feeds['cache'],[self.calls])
                self.calls+=1
                return [np.zeros((1,8,3)),np.full((1,8,2),self.calls),np.array([self.calls],dtype=np.float32)]
        backend=Backend()
        h,lp=orbit.encode(backend,manifest,metadata,np.zeros((98,2),dtype=np.float32))
        self.assertEqual(h.shape,(23,2));self.assertEqual(lp.shape,(23,3));self.assertEqual(backend.calls,3)
        self.assertEqual(h[8,0],2)
    def test_spd_metric_fit_retains_distinct_phone_contrasts(self):
        try: import torch
        except ImportError: self.skipTest('Tiny optimizer test requires the remote torch dependency')
        rng=np.random.default_rng(1);rows=[]
        for i in range(3):
            phone=np.tile([0,1,2],50)
            clean=np.stack([3.*(phone-1),rng.normal(0,.05,len(phone))],1)
            views=np.array([clean]+[clean+np.stack([np.zeros(len(phone)),rng.normal(0,1,len(phone))],1) for _ in range(3)])
            rows.append(dict(views=views,occupancy=np.eye(3)[phone],speaker=str(i),ids=[1,2]*50))
        model=orbit.fit(rows,2,1.,steps=12)
        self.assertEqual(len(model['regions']),16)
        for region in model['regions']:
            factor=np.array(region['precision'])
            self.assertTrue(np.all(np.linalg.eigvalsh(factor.T@factor)>0))
        for arm in ('local','euclidean','global'):
            values=orbit.emissions(model,rows[0]['views'][0],arm)
            self.assertGreater(np.mean(values.argmax(1)==np.tile([0,1,2],50)),.9)

if __name__=='__main__':unittest.main()

"""Small numerical/provenance tests; real extraction and fitting run on Modal."""
import copy
import unittest
import numpy as np
from scipy.special import logsumexp
import student_moving_zero as m


def model():
    def arm(k):
        return dict(mean=np.tile([.5,.1,.8,.15], (3,k,1)).tolist(),variance=np.full((3,k,4), .1).tolist())
    return dict(schema=1,inventory='test',tokens='ab',frameMs=40,stay=.8,maxFrames=2048,phoneCounts=[5,5],
        center=[0.]*8,scale=[1.]*8,stationaryVariance=[.0036,.01],
        poles=dict(mean=np.zeros((3,8)).tolist(),variance=np.ones((3,8)).tolist()),control=arm(2),
        moving=dict(**arm(1),alpha=np.full((3,4), .9).tolist(),noise=np.full((3,4), .0025).tolist()))


def frames(n=6):
    return [dict(poles=[0.]*8,zeros=[[.2,.1],[.5+t*.02,.1],[.8-t*.01,.15]],valid=True) for t in range(n)]


def enumerate_paths(model, window, labels, moving):
    """Independent path enumeration over boundary, correspondence and nuisance."""
    states = [0]
    for p in labels:
        states += [p,0]
    initial, trans, pole = m.likelihoods(model, window, moving)
    mass, timing = [], []
    def choices(s):
        return [s]+([s+1] if s+1 < len(states) else [])+([s+2] if s+2 < len(states) and states[s+2] and states[s+2] != states[s] else [])
    def walk(t, s, h, j, weight, value):
        if t == len(window['x'])-1:
            if s >= len(states)-2:
                mass.append(weight+value)
            return
        dests = choices(s)
        for d in dests:
            lp = 0 if len(dests) == 1 else np.log(.8 if d == s else .2/(len(dests)-1))
            for k in range(6):
                walk(t+1,d,h,k,weight+lp-np.log(6),value+trans[t,states[d],j,k]+pole[t+1,states[d]]+window['nuisance'][t+1,h,k])
    def clock(t,s,weight):
        if t == len(window['x'])-1:
            if s >= len(states)-2:
                timing.append(weight)
            return
        dests=choices(s)
        for d in dests:
            clock(t+1,d,weight+(0 if len(dests)==1 else np.log(.8 if d==s else .2/(len(dests)-1))))
    for s in (0,1):
        clock(0,s,-np.log(2))
        for h in range(3):
            for j in range(6):
                walk(0,s,h,j,-np.log(36),initial[0,states[s],j]+pole[0,states[s]]+window['nuisance'][0,h,j])
    return float(logsumexp(mass)-logsumexp(timing))


class MovingZeroTests(unittest.TestCase):
    def test_joint_forward_sum_and_soft_posteriors(self):
        md=model();w=m.prepare_window(frames(3))
        for moving in (False,True):
            for labels in ([1],[1,2],[1,1]):
                value, gamma, pairs=m.chain(md,w,labels,moving,True)
                self.assertAlmostEqual(value,enumerate_paths(md,w,labels,moving),10)
                np.testing.assert_allclose(gamma.sum((1,2)),1,atol=1e-10)
                if moving:
                    np.testing.assert_allclose(pairs.sum((1,2,3)),1,atol=1e-10)
        self.assertFalse(np.isfinite(m.chain(md,m.prepare_window(frames(2)),[1,1],True)))

    def test_permutation_and_temporal_information(self):
        md=model();raw=frames(10)
        permuted=copy.deepcopy(raw)
        for i,r in enumerate(permuted):
            r['zeros']=r['zeros'][::-1] if i%2 else [r['zeros'][1],r['zeros'][2],r['zeros'][0]]
        for moving in (False,True):
            self.assertAlmostEqual(m.chain(md,m.prepare_window(raw),[1,2],moving),m.chain(md,m.prepare_window(permuted),[1,2],moving),10)
        shuffled=[raw[j] for j in [0,9,1,8,2,7,3,6,4,5]]
        self.assertAlmostEqual(m.chain(md,m.prepare_window(raw),[1],False),m.chain(md,m.prepare_window(shuffled),[1],False),10)
        self.assertGreater(m.chain(md,m.prepare_window(raw),[1],True),m.chain(md,m.prepare_window(shuffled),[1],True))

    def test_rational_roots_and_gain_invariance(self):
        r=.8;angle=.3*np.pi
        np.testing.assert_allclose(m.root_pairs([1,-2*r*np.cos(angle),r*r]),[[.3,-np.log(r)/np.pi]],atol=1e-10)
        audio=.2*np.sin(np.arange(1600)*.13)+.02*np.sin(np.arange(1600)*.71)
        first=m.observations(audio);scaled=m.observations(audio*2)
        self.assertTrue(all(r['valid'] for r in first))
        for a,b in zip(first,scaled):
            np.testing.assert_allclose(a['zeros'],b['zeros'],atol=1e-7)

    def test_training_provenance_and_matched_capacity(self):
        records=[dict(source='help-train',speaker=str(i),kind='none',split='train',excluded=False,pronunciation='ab',
            observations=[dict(r,poles=[t*.1]*8) for t,r in enumerate(frames(10))]) for i in range(3)]
        rows=m.prepare(records,'ab',set())
        md,history=m.fit(rows,'ab',16)
        self.assertTrue(all(np.isfinite(h).all() for h in history.values()))
        control=sum(np.asarray(md['control'][k]).size for k in ('mean','variance'))
        moving=sum(np.asarray(md['moving'][k]).size for k in ('mean','variance','alpha','noise'))
        self.assertEqual(control,moving)
        for key,value in [('kind','repeat'),('split','held-out'),('excluded',True)]:
            changed=copy.deepcopy(records);changed[0][key]=value
            with self.assertRaises(ValueError):m.prepare(changed,'ab',set())
        with self.assertRaises(ValueError):m.prepare(records,'ab',{'1'})


if __name__ == '__main__':
    unittest.main()

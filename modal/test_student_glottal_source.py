import itertools
import unittest
import numpy as np
from student_glottal_source import eligible, speaker_folds, resynthesize, occurrence_occupancies, excitation_analysis, consistency


class SourceViews(unittest.TestCase):
    def setUp(self):
        rate=16000
        self.phase=np.arange(rate)*160/rate
        self.audio=(.1*np.sin(2*np.pi*self.phase)+.04*np.sin(4*np.pi*self.phase)).astype(np.float32)
        self.voiced=np.ones(rate,bool); self.voiced[5000:7000]=False
        self.audio[5000:7000]=np.random.default_rng(8).normal(0,.03,2000)

    def test_identity_filter_roundtrip(self):
        out=resynthesize(self.audio,self.phase,self.voiced)
        np.testing.assert_allclose(out,self.audio,atol=2e-6,rtol=1e-5)

    def test_source_change_preserves_unvoiced_and_duration(self):
        a=resynthesize(self.audio,self.phase,self.voiced,.45,.04)
        b=resynthesize(self.audio,self.phase,self.voiced,.75,.16)
        self.assertEqual(a.shape,self.audio.shape)
        np.testing.assert_array_equal(a[~self.voiced],self.audio[~self.voiced])
        np.testing.assert_array_equal(b[~self.voiced],self.audio[~self.voiced])
        self.assertGreater(np.linalg.norm(a[self.voiced]-b[self.voiced]),.1)
        np.testing.assert_array_equal(a,resynthesize(self.audio,self.phase,self.voiced,.45,.04))

    def test_pitch_is_retained_on_known_periodic_input(self):
        from scipy.signal import periodogram
        for opening in [.45,.75]:
            a=resynthesize(self.audio,self.phase,self.voiced,opening,0.)
            f,p=periodogram(a[8000:15000],16000)
            region=(f>=100)&(f<=250)
            self.assertAlmostEqual(f[region][np.argmax(p[region])],160,delta=3)

    def test_voicing_analysis_protects_noise_interval(self):
        pitch,phase,safe=excitation_analysis(self.audio)
        self.assertFalse(safe[5500:6500].any())
        self.assertTrue(safe[9000:14000].any())
        self.assertEqual(len(phase),len(self.audio))
        self.assertTrue(np.isfinite(pitch).all())


class TranscriptOccurrences(unittest.TestCase):
    def test_soft_occupancies_agree_with_enumerated_paths(self):
        lp=np.log(np.array([[.4,.5,.1],[.4,.4,.2],[.2,.7,.1],[.6,.3,.1]]))
        ids=[1,1]; labels=[0,1,0,1,0]; paths=[]
        for states in itertools.product(range(5),repeat=4):
            if states[0] not in (0,1) or states[-1] not in (3,4): continue
            if any(b not in [a,a+1]+([a+2] if a+2<5 and labels[a+2] and labels[a+2]!=labels[a] else [])
                   for a,b in zip(states,states[1:])): continue
            weight=np.exp(sum(lp[t,labels[s]] for t,s in enumerate(states)))
            paths.append((states,weight))
        expected=np.zeros((4,2)); z=sum(w for _,w in paths)
        for states,w in paths:
            for t,s in enumerate(states):
                if s%2: expected[t,s//2]+=w/z
        np.testing.assert_allclose(occurrence_occupancies(lp,ids),expected,atol=1e-7)
        self.assertGreater(expected[:,0].sum(),0)
        self.assertGreater(expected[:,1].sum(),0)

    def test_incomplete_repeated_phone_abstains(self):
        with self.assertRaises(ValueError): occurrence_occupancies(np.log([[.5,.5],[.5,.5]]),[1,1])

    def test_grouped_selection_excludes_speakers_and_bad_labels(self):
        base=dict(kind='none',split='train',speaker='s0',events=[],non_errors=[])
        self.assertTrue(eligible(base,set()))
        for key,value in [('kind','word'),('split','eval'),('events',[{}]),('non_errors',[{}]),('excluded',True)]:
            self.assertFalse(eligible({**base,key:value},set()))
        self.assertFalse(eligible(base,{'s0'}))
        folds=speaker_folds(f's{i}' for i in range(12))
        self.assertEqual(folds,speaker_folds(reversed(list(folds))))
        self.assertEqual(set(folds.values()),{0,1,2})


class PairedObjective(unittest.TestCase):
    def test_identity_consistency_and_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch unavailable; run on the training image')
        occupancy=torch.tensor([[[1.,0.],[0.,1.]],[[1.,0.],[0.,1.]]])
        same=torch.tensor([[[0.,2.,-2.],[0.,-2.,2.]],[[0.,2.,-2.],[0.,-2.,2.]]])
        self.assertAlmostEqual(float(consistency(same,occupancy)),0,places=6)
        changed=same.clone(); changed[1]=changed[1].flip(-1); changed.requires_grad_()
        loss=consistency(changed,occupancy)
        self.assertGreater(float(loss),0)
        loss.backward()
        self.assertTrue(torch.isfinite(changed.grad).all())
        self.assertGreater(float(changed.grad.abs().sum()),0)


if __name__=='__main__': unittest.main()

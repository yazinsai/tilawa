import unittest
from student_acoustic_span import compose_recording, partition, resample


class FragmentControls(unittest.TestCase):
    def row(self):
        import numpy as np
        x = np.arange(32*16,dtype=np.float32).reshape(16,32)
        return dict(id='take', features=x.tolist(), words=[
            {'from':0,'to':8,'ids':[1,2],'phoneBounds':[[1,3],[5,7]]},
            {'from':8,'to':16,'ids':[1,2],'phoneBounds':[[9,11],[13,15]]}])

    def test_same_recording_exact_phones_frames_and_distinct_occurrences(self):
        r=self.row(); words=compose_recording(r)
        self.assertEqual(len(words),2)
        for w in words:
            self.assertEqual(w['ids'],[1,2])
            self.assertEqual(len(set(w['donors'])),2)
            for pi,(wi,pj) in enumerate(w['donors']):
                self.assertEqual(r['words'][wi]['ids'][pj],w['ids'][pi])
                self.assertNotEqual((wi,pj),(w['word'],pi))
            for coherent,negative in zip(w['coherent'],w['fragments']):
                self.assertEqual(coherent.shape,(8,32))
                self.assertEqual(coherent.shape,negative.shape)
                self.assertFalse((coherent==negative).all())

    def test_cannot_create_missing_phone_or_cross_recording_donor(self):
        r=self.row();r['words']=r['words'][:1]
        self.assertEqual(compose_recording(r),[])
        r=self.row();r['words'][1]['ids']=[1,3]
        self.assertEqual(compose_recording(r),[])

    def test_identical_join_geometry_and_reproducible_pairing(self):
        import numpy as np
        r=self.row(); a=compose_recording(r); b=compose_recording(r)
        for u,v in zip(a,b):
            self.assertEqual(u['donors'],v['donors'])
            for x,y in zip(u['coherent'],v['coherent']):np.testing.assert_array_equal(x,y)
        x=np.ones((3,32),np.float32)
        np.testing.assert_array_equal(resample(resample(x,5),3),x)
        self.assertEqual(partition('training-speaker'),partition('training-speaker'))

    def test_temporal_readout_and_marginalized_boundaries_have_finite_gradients(self):
        import torch
        from student_acoustic_span import make_head,predict
        head=make_head(4)
        self.assertEqual(head.hidden.in_features,80)
        score=predict(head,torch.randn(8,32),torch.tensor([1,2,1]))
        score.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))

if __name__=='__main__':unittest.main()

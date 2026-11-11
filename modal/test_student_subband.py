"""Mechanism checks; run with torch in the authorized training image."""
import unittest
import torch
from student_subband import SubbandEar, frequency_mix, reliability_target, source_partition


class SubbandTest(unittest.TestCase):
    def test_source_selection_excludes_whole_release_and_gold_speaker_clusters(self):
        base=dict(shard='s',split='train',bucket='clean-match',n_samples=32000)
        rows=[dict(base,row=i,cluster=c) for i,c in enumerate(('a','a','b','b','c',None))]
        rows[3]['split']='test'
        selected,held=source_partition(rows,{('s',0)})
        self.assertEqual([r['row'] for r in selected],[4])
        self.assertEqual(held,{'a','b'})

    def test_control_has_equal_parameters_and_genuinely_broadband_inputs(self):
        fused,control=SubbandEar(16,False),SubbandEar(16,True)
        self.assertEqual(sum(p.numel() for p in fused.parameters()),sum(p.numel() for p in control.parameters()))
        mix=frequency_mix()
        self.assertTrue(torch.allclose(mix@mix.T,torch.eye(80),atol=1e-5))
        for lo,hi in ((0,32),(16,48),(32,64),(48,80)):
            self.assertTrue(torch.all(mix[lo:hi].square().sum(0)>0))

    def test_corrupted_outside_band_cannot_change_that_bands_representation(self):
        model=SubbandEar(16).eval()
        x=torch.randn(1,101,80); y=x.clone();y[:,:,48:80]+=10
        with torch.no_grad():
            a=model.encoders[0](x[:,:,:32]);b=model.encoders[0](y[:,:,:32])
        self.assertTrue(torch.equal(a,b))
        # The matched early mix exposes that same encoder to the remote corruption.
        a=x@model.mix.T;b=y@model.mix.T
        self.assertFalse(torch.allclose(a[:,:,:32],b[:,:,:32]))

    def test_reliability_targets_follow_physical_corruption(self):
        mask=torch.ones(1,101,80);mask[:,:,:32]=0
        q=reliability_target(mask)
        self.assertTrue(torch.all(q[:,:,0]<q[:,:,3]))
        self.assertTrue(torch.allclose(q.sum(-1),torch.ones_like(q[:,:,0])))
        self.assertTrue(torch.isfinite(reliability_target(torch.zeros_like(mask))).all())
        self.assertFalse(torch.allclose(q,reliability_target(mask,True)))

    def test_native_output_clock_and_normalized_ctc_distribution(self):
        model=SubbandEar(16).eval()
        for n in (77,101,409):
            with torch.no_grad(): lp,rel=model(torch.randn(1,n,80))
            self.assertEqual(lp.shape,(1,(n-9)//4+1,43))
            self.assertTrue(torch.allclose(lp.exp().sum(-1),torch.ones_like(lp[:,:,0]),atol=1e-6))
            self.assertTrue(torch.allclose(rel.exp().sum(-1),torch.ones_like(rel[:,:,0]),atol=1e-6))


if __name__=='__main__':unittest.main()

"""Structural checks for pause masks, run in the train image."""
import copy
import unittest
import torch
from student_pause_mask import allowed_sources,PauseStreaming,stream,time_windows
torch.set_num_threads(2)


class PauseMaskTests(unittest.TestCase):
    def test_indexed_windows_match_convolution_windows(self):
        x=torch.arange(14.).reshape(2,7)
        torch.testing.assert_close(time_windows(x,5,3),x.unfold(-1,3,1),rtol=0,atol=0)

    def test_canonical_seam_skips_empty_assimilated_words(self):
        from student_pause_train import seam_offsets
        self.assertEqual(seam_offsets({'p':'abc','o':[0,0,1,1,2]},0),(0,1))
        self.assertEqual(seam_offsets({'p':'abc','o':[0,0,1,1,2]},2),(1,2))
        self.assertIsNone(seam_offsets({'p':'abc','o':[0,0,1,1,2]},4))

    def test_training_cut_keeps_real_pause_frames(self):
        import numpy as np
        from student_pause_train import pause_floors
        ids=[1,1]+[0]*8+[2,2,3,3]
        lp=torch.full((14,4),-10.)
        for i,p in enumerate(ids): lp[i,p]=0
        floors,pauses=pause_floors(lp,[1,2,3],[(1,2)],np.random.default_rng(0),1.)
        self.assertEqual(pauses,[(2,10)])
        self.assertEqual(floors[:4].tolist(),[-1]*4)
        self.assertEqual(floors[4:24].unique().tolist(),[4])
        self.assertEqual(floors[24:].unique().tolist(),[-1])
        none,_=pause_floors(lp,[1,2,3],[(1,2)],np.random.default_rng(0),0.)
        self.assertEqual(none.unique().tolist(),[-1])

    def test_only_pre_pause_sources_are_blocked(self):
        floors=torch.tensor([[-1,-1,7,7]])
        allowed=allowed_sources(floors,torch.tensor([5]),7,3)[0]
        self.assertTrue(allowed[:2].all())
        self.assertFalse(allowed[2:,:5].any())
        self.assertTrue(allowed[2:,5:].all())

    def test_unmasked_stream_preserves_outputs_and_all_state(self):
        from student_train import build_model
        from student_export import wrap
        torch.manual_seed(1)
        m=build_model('lite')
        plain=wrap(copy.deepcopy(m),16,128)
        masked=PauseStreaming(m)
        a,b=plain.init_states(),masked.init_states()
        for _ in range(2):
            x=torch.randn(1,45,80)
            with torch.no_grad():
                y,a=plain(x,a); z,b=masked(x,b,torch.full((1,16),-1,dtype=torch.int64))
            torch.testing.assert_close(y,z,rtol=1e-5,atol=1e-6)
            for u,v in zip(a,b): torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)

    def test_mask_keeps_clock_and_supports_batch_gradients(self):
        from student_train import build_model
        torch.manual_seed(3)
        m=PauseStreaming(build_model('lite'))
        x=torch.randn(2,77,80)
        floors=torch.full((2,32),-1,dtype=torch.int64); floors[0,16:]=16
        y=stream(m,x,floors)
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        _,states=m(x[:,:45],m.init_states(2),floors[:,:16])
        self.assertEqual(states[-1].tolist(),[16,16])

    def test_attention_has_no_pre_pause_key_dependency(self):
        from student_train import build_model
        m=PauseStreaming(build_model('lite'))
        a=m.masked[0][0]
        a.pause_floor=torch.full((1,16),128,dtype=torch.int64)
        a.pause_processed=torch.tensor([128])
        x=torch.randn(16,1,192)
        # Use the encoder's real relative-position module.
        stage=m.encoder.encoders[0]
        pos=stage.encoder_pos(x,128)
        cache=torch.randn(128,1,128)
        with torch.no_grad():
            u,_=a.streaming_forward(x,pos,cache,128,torch.zeros(1,144,dtype=torch.bool))
            v,_=a.streaming_forward(x,pos,cache+100,128,torch.zeros(1,144,dtype=torch.bool))
        torch.testing.assert_close(u,v,rtol=0,atol=0)
        self.assertEqual(u[...,:128].count_nonzero().item(),0)
        a.pause_floor[:,:8]=-1; a.pause_floor[:,8:]=136
        changed=x.clone(); changed[:8]+=100
        with torch.no_grad():
            u,_=a.streaming_forward(x,pos,cache,128,torch.zeros(1,144,dtype=torch.bool))
            v,_=a.streaming_forward(changed,pos,cache,128,torch.zeros(1,144,dtype=torch.bool))
        torch.testing.assert_close(u[:,:,8:],v[:,:,8:],rtol=0,atol=0)
        self.assertEqual(u[:,:,8:,:136].count_nonzero().item(),0)

    def test_convolution_has_no_pre_pause_cache_dependency(self):
        from student_train import build_model
        m=PauseStreaming(build_model('lite'))
        conv=m.masked[1][0]
        conv.pause_floor=torch.full((1,16),128,dtype=torch.int64)
        conv.pause_processed=torch.tensor([128])
        x=torch.randn(1,192,16); cache=torch.randn(1,192,15)
        with torch.no_grad():
            u,_=conv.streaming_forward(x,cache)
            v,_=conv.streaming_forward(x,cache+100)
        torch.testing.assert_close(u,v,rtol=0,atol=0)

    def test_frontend_pad_cannot_cross_pause(self):
        from student_train import build_model
        m=PauseStreaming(build_model('lite'))
        conv=m.encoder_embed.convnext
        conv.pause_floor=torch.full((1,16),128,dtype=torch.int64)
        conv.pause_processed=torch.tensor([128])
        c=conv.depthwise_conv.groups
        x=torch.randn(1,c,19,19); cache=torch.randn(1,c,3,19)
        with torch.no_grad():
            u,_=conv.streaming_forward(x,cache)
            v,_=conv.streaming_forward(x,cache+100)
        torch.testing.assert_close(u,v,rtol=0,atol=0)


if __name__=='__main__': unittest.main()

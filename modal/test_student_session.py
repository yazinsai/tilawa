"""Small behavioral tests; executed on the supervisor training image."""
import unittest
import torch
from torch import nn
from student_session import SessionStudent, FrameEncoder


class Conditioning(unittest.TestCase):
    def test_frame_pool_is_content_order_invariant(self):
        torch.manual_seed(0)
        encoder = FrameEncoder(12)
        frames = torch.randn(9, 1, 12)
        self.assertTrue(torch.allclose(encoder(frames).mean(0), encoder(frames.flip(0)).mean(0), atol=1e-7))

    def test_zero_vector_and_initialization_are_identity(self):
        # Exercise the actual hook for both subsampling and encoder layouts.
        model = object.__new__(SessionStudent)
        nn.Module.__init__(model)
        for batch_dim, shape in [(0, (2, 5, 12)), (1, (5, 2, 12))]:
            film = nn.Linear(64, 24, bias=False)
            nn.init.zeros_(film.weight)
            model.embedding = torch.randn(2, 64)
            hook = model._film(film, -1, batch_dim)
            x = torch.randn(*shape)
            self.assertTrue(torch.equal(x, hook(None, (), x)))
            nn.init.normal_(film.weight)
            model.embedding = torch.zeros(2, 64)
            self.assertTrue(torch.equal(x, hook(None, (), x)))
            model.embedding = torch.randn(2, 64)
            self.assertFalse(torch.equal(x, hook(None, (), x)))


class StreamingArchitecture(unittest.TestCase):
    def test_frozen_stream_features_and_identity_logits(self):
        import student_train as T
        from student_session_export import Streaming
        torch.set_num_threads(2)
        T.SIZES['session-test'] = dict(num_encoder_layers='1,1,1,1,1,1', feedforward_dim='32,32,32,32,32,32',
            encoder_dim='16,16,16,16,16,16', encoder_unmasked_dim='16,16,16,16,16,16')
        old = T.COMMON.copy()
        try:
            T.COMMON.update(num_heads='2,2,2,2,2,2', query_head_dim='8', value_head_dim='4', pos_head_dim='2', pos_dim=8,
                cnn_module_kernel='3,3,3,3,3,3')
            parent = T.build_model('session-test').eval()
            model = SessionStudent(parent).eval()
            s = Streaming(model)
            x = torch.randn(1,45,80)
            with torch.no_grad():
                a, f, states = s(x, torch.randn(1,64), s.init_states())
                b, _, _ = s(x, torch.zeros(1,64), s.init_states())
            self.assertEqual(tuple(f.shape), (1,16,64))
            self.assertTrue(torch.equal(a,b))
            self.assertEqual(len(states), len(s.init_states()))
            self.assertTrue(all(not p.requires_grad for p in model.frozen.parameters()))
        finally:
            T.COMMON.clear(); T.COMMON.update(old)


if __name__ == '__main__':
    unittest.main()

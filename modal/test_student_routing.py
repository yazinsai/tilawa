"""Small CPU tests; no corpus, checkpoint, Modal credentials or icefall needed."""
import unittest

import torch
from torch import nn

torch.set_num_threads(1)

from student_routing import (
    AcousticGrid, HeadCorrespondence, STUDENT_GRID, TEACHER_GRID,
    capture_streaming_attention, capture_teacher_attention,
    deterministic_streaming, feature_hint_loss, normalized_rows,
    project_teacher, routing_loss,
)


class RoutingTests(unittest.TestCase):
    def test_grid_uses_frontend_positions_not_sequence_lengths(self):
        positions = TEACHER_GRID.centres(torch.arange(8))
        self.assertEqual(STUDENT_GRID.bins(positions).tolist(), [-2, -1, 0, 1, 2, 3, 4, 5])
        pooled = AcousticGrid(840, 320, 2)
        self.assertEqual(pooled.bins(positions).tolist(), [-1, -1, 0, 0, 1, 1, 2, 2])
        self.assertEqual(pooled.centres(torch.arange(3)).tolist(), [1000, 1640, 2280])

    def test_invalid_grids_fail(self):
        for args in ((0, 0), (0, 1, 0), (0.0, 1)):
            with self.assertRaises(ValueError):
                AcousticGrid(*args)

    def test_masked_mass_and_gradients_are_exactly_excluded(self):
        p = torch.tensor([[[float('nan'), 2., 6.]]], requires_grad=True)
        support = torch.tensor([[[False, True, True]]])
        got = normalized_rows(p, support)
        torch.testing.assert_close(got, torch.tensor([[[0., .25, .75]]]))
        got[..., 1].sum().backward()
        self.assertEqual(p.grad[..., 0].item(), 0)
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_empty_or_invalid_supported_rows_fail(self):
        for p, mask in ((torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool)),
                        (torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.bool)),
                        (torch.tensor([[float('nan'), 1.]]), torch.ones(1, 2, dtype=torch.bool))):
            with self.assertRaises(ValueError):
                normalized_rows(p, mask)

    def test_teacher_rebins_mass_and_averages_query_rows(self):
        # Two teacher observations per student bin. The last key is a future
        # bin and the first is an unavailable startup-cache entry.
        attention = torch.tensor([[[1., 2., 3., 4., 90., 90.]]]).expand(1, 6, 6).clone()
        grid = AcousticGrid(0, 2, 2)
        target, paired = project_teacher(attention, AcousticGrid(0, 2), grid,
                                          torch.tensor([0, 1, 7]), torch.tensor([-1, 0, 1, 2]),
                                          torch.tensor([[False, True, True, False]] * 3))
        self.assertEqual(paired.tolist(), [True, True, False])
        torch.testing.assert_close(target, torch.tensor([[[0., .3, .7, 0.], [0., .3, .7, 0.]]]))

    def test_query_specific_support_excludes_nan_before_arithmetic(self):
        a = torch.ones(1, 3, 3)
        a[0, 0, 2] = float('nan')
        a[0, 2] = float('nan')  # unpaired query
        target, paired = project_teacher(a, AcousticGrid(0, 2), AcousticGrid(0, 2),
                                          torch.tensor([0, 1]), torch.arange(3),
                                          torch.tensor([[True, True, False], [False, True, True]]))
        self.assertTrue(paired.all())
        torch.testing.assert_close(target, torch.tensor([[[.5, .5, 0.], [0., .5, .5]]]))

    def test_each_teacher_head_is_conditioned_before_mixing(self):
        a = torch.tensor([[[.02, .08, .90]] * 3, [[.8, .2, 0.]] * 3])
        target, _ = project_teacher(a, AcousticGrid(0, 2), AcousticGrid(0, 2),
                                    torch.tensor([0]), torch.arange(3),
                                    torch.tensor([[True, True, False]]))
        mixture = HeadCorrespondence(1, 2)(target)
        # Conditioning the mixed raw maps would give .745/.255 instead.
        torch.testing.assert_close(mixture, torch.tensor([[[.5, .5, 0.]]]))

    def test_head_correspondence_learns_with_frozen_teacher(self):
        teacher = torch.tensor([[[.99, .01]], [[.01, .99]]], requires_grad=True)
        student = torch.tensor([[[.9, .1]]])
        mapping = HeadCorrespondence(1, 2)
        opt = torch.optim.Adam(mapping.parameters(), lr=.1)
        allowed = torch.ones(1, 2, dtype=torch.bool)
        initial = routing_loss(student, teacher, mapping, allowed).item()
        for _ in range(20):
            opt.zero_grad()
            routing_loss(student, teacher, mapping, allowed).backward()
            opt.step()
        self.assertLess(routing_loss(student, teacher, mapping, allowed).item(), initial / 5)
        self.assertIsNone(teacher.grad)
        mapping.freeze()
        self.assertFalse(mapping.logits.requires_grad)

    def test_unequal_heads_transfer_routing_gradients_only_on_support(self):
        student = torch.tensor([[[.2, .4, .4]], [[.4, .2, .4]]], requires_grad=True)
        teacher = torch.tensor([[[.8, .2, 0.]], [[.2, .8, 0.]], [[.5, .5, 0.]]], requires_grad=True)
        mapping = HeadCorrespondence(2, 3).freeze()
        loss = routing_loss(student, teacher, mapping, torch.tensor([[True, True, False]]))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(mapping.logits.grad)
        self.assertTrue((student.grad[..., :2].abs() > 0).all())
        self.assertTrue((student.grad[..., 2] == 0).all())

    def test_equal_routes_have_zero_loss_despite_masked_values(self):
        a = torch.tensor([[[.25, .75, 123.]]])
        target = torch.tensor([[[.25, .75, 0.]]])
        self.assertAlmostEqual(routing_loss(a, target, HeadCorrespondence(1, 1),
                                           torch.tensor([[True, True, False]])).item(), 0., places=6)

    def test_no_overlap_fails_instead_of_repeating_teacher_edges(self):
        with self.assertRaises(ValueError):
            project_teacher(torch.ones(1, 2, 2), TEACHER_GRID, STUDENT_GRID,
                            torch.tensor([100]), torch.tensor([100]), torch.ones(1, 1, dtype=torch.bool))

    def test_feature_control_matches_values_and_detaches_teacher(self):
        x = torch.tensor([[1., 2.], [float('nan'), float('nan')]], requires_grad=True)
        t = torch.tensor([[2., 1., 3.], [float('nan')] * 3], requires_grad=True)
        projection = nn.Linear(2, 3)
        loss = feature_hint_loss(x, t, projection, torch.tensor([True, False]))
        loss.backward()
        self.assertIsNone(t.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad[1] == 0).all())


class FakeStudentAttention(nn.Module):
    num_heads = 2

    def streaming_forward(self, x, pos_emb, cached_key, left_context_len, key_padding_mask):
        # Return object identity as well as numeric parity through capture.
        weights = torch.ones(2, x.shape[1], x.shape[0], left_context_len + x.shape[0])
        return weights, cached_key


class FakeTeacherAttention(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2, torch.ones(hidden_states.shape[0], 3, hidden_states.shape[1], hidden_states.shape[1])


class CaptureTests(unittest.TestCase):
    def test_stream_capture_uses_actual_cache_and_mask_without_changing_outputs(self):
        module = FakeStudentAttention()
        x, cache = torch.randn(4, 1, 8), torch.randn(3, 1, 8)
        mask = torch.tensor([[True, True, False, False, False, False, False]])
        plain = module.streaming_forward(x, None, cache, 3, mask)
        with capture_streaming_attention({module: 2}, processed_frames=4) as captured:
            result = module.streaming_forward(x=x, pos_emb=None, cached_key=cache,
                                              left_context_len=3, key_padding_mask=mask)
        torch.testing.assert_close(result[0], plain[0])
        self.assertIs(result[1], cache)
        c = captured[module]
        self.assertIs(c.probabilities, result[0])
        self.assertIs(c.features, x)
        self.assertEqual(c.queries.tolist(), [2, 3, 4, 5])
        self.assertEqual(c.keys.tolist(), [-1, 0, 1, 2, 3, 4, 5])
        self.assertEqual(c.allowed[0, 0].tolist(), [False, False, True, True, True, True, True])
        self.assertEqual(c.grid.pool, 2)
        self.assertNotIn('streaming_forward', module.__dict__)

    def test_capture_restores_methods_after_exceptions_and_nested_scopes(self):
        module = FakeStudentAttention()
        with capture_streaming_attention({module: 1}, 0):
            outer = module.streaming_forward
            with self.assertRaises(RuntimeError):
                with capture_streaming_attention({module: 1}, 0):
                    raise RuntimeError('forward failed')
            self.assertIs(module.streaming_forward, outer)
        self.assertNotIn('streaming_forward', module.__dict__)
        with self.assertRaises(ValueError):
            with capture_streaming_attention({module: 2}, 3):
                pass

    def test_teacher_capture_is_detached_and_removes_hooks_after_failure(self):
        module = FakeTeacherAttention().eval()
        x = torch.randn(1, 4, 8, requires_grad=True)
        with self.assertRaises(RuntimeError):
            with capture_teacher_attention([module]) as captured:
                out = module(hidden_states=x)
                self.assertFalse(captured[module][0].requires_grad)
                self.assertFalse(captured[module][1].requires_grad)
                torch.testing.assert_close(out[0], x * 2)
                raise RuntimeError('failed')
        self.assertFalse(module._forward_hooks)
        module.train()
        with self.assertRaises(ValueError):
            with capture_teacher_attention([module]):
                pass

    def test_execution_mode_restores_every_module_and_geometry(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(1, 1)
                self.encoder.chunk_size = (16, 32)
                self.encoder.left_context_frames = (64, 128)
                self.dropout = nn.Dropout(.3)

            def streaming_eval(self, chunk, left):
                self.encoder.chunk_size = (chunk,)
                self.encoder.left_context_frames = (left,)

        model = Model().train()
        model.encoder.eval()
        with self.assertRaises(RuntimeError):
            with deterministic_streaming(model, 16, 128):
                self.assertFalse(model.dropout.training)
                self.assertEqual(model.encoder.chunk_size, (16,))
                raise RuntimeError('failed')
        self.assertTrue(model.training)
        self.assertTrue(model.dropout.training)
        self.assertFalse(model.encoder.training)
        self.assertEqual(model.encoder.chunk_size, (16, 32))
        self.assertEqual(model.encoder.left_context_frames, (64, 128))


if __name__ == '__main__':
    unittest.main()

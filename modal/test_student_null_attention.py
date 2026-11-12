"""Small-tensor behavior and ONNX tests; no audio, checkpoints or lab data."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from student_null_attention import QueryAttention, install_query_attention


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.query_head_dim, self.pos_head_dim, self.dropout = 2, 3, 2, 0
        self.in_proj = nn.Linear(8, 16)
        self.linear_pos = nn.Linear(4, 4, bias=False)

    def oracle(self, x, pos, cached=None, hidden=None):
        projected = self.in_proj(x)
        t, b = x.shape[:2]
        q, k, p = projected.split([6, 6, 4], dim=-1)
        if cached is not None:
            k = torch.cat([cached, k])
        key_len = k.shape[0]
        q = q.reshape(t, b, 2, 3).permute(2, 1, 0, 3)
        k = k.reshape(key_len, b, 2, 3).permute(2, 1, 3, 0)
        p = p.reshape(t, b, 2, 2).permute(2, 1, 0, 3)
        pos = self.linear_pos(pos).reshape(-1, key_len + t - 1, 2, 2).permute(2, 0, 3, 1)
        relative = p @ pos
        # Upstream eager expression, independent of the adapter's gather.
        absolute = relative.as_strided((2, b, t, key_len),
            (relative.stride(0), relative.stride(1), relative.stride(2) - relative.stride(3), relative.stride(3)),
            storage_offset=relative.stride(3) * (t - 1))
        scores = q @ k + absolute
        if hidden is not None:
            scores = scores.masked_fill(hidden, -1000)
        return scores.softmax(-1)


class NullAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)
        self.base = Base().eval()
        self.x = torch.randn(4, 2, 8)
        self.pos = torch.randn(1, 7, 4)

    def test_exact_extended_zero_value_attention_and_residual(self):
        adapter = QueryAttention(self.base, "null").eval()
        scores = torch.randn(2, 2, 4, 5)
        query = torch.randn(2, 2, 4, 3)
        values = torch.randn(2, 2, 5, 6)
        with torch.no_grad():
            adapter.score_weight.normal_()
            adapter.score_bias.zero_()
        hidden = torch.zeros(2, 4, 5, dtype=torch.bool)
        hidden[:, :, 0] = True
        got = adapter.distribute(scores, query, hidden)
        null = (query * adapter.score_weight[:, None, None]).sum(-1) + adapter.score_bias[:, None, None]
        extended = torch.cat([scores.masked_fill(hidden[None], float('-inf')), null[..., None]], -1).softmax(-1)
        zero_values = torch.cat([values, torch.zeros_like(values[:, :, :1])], -2)
        torch.testing.assert_close(got @ values, extended @ zero_values)
        self.assertTrue(torch.equal(got[..., 0], torch.zeros_like(got[..., 0])))
        # Null dominance changes retrieval, not the local residual input.
        adapter.score_bias.data.fill_(100)
        retrieval = adapter.distribute(scores, query, hidden) @ values
        residual = torch.randn_like(retrieval)
        torch.testing.assert_close(residual + retrieval, residual)

    def test_query_condition_and_no_temperature_abstention(self):
        null = QueryAttention(self.base, 'null').eval()
        temperature = QueryAttention(copy.deepcopy(self.base), 'temperature').eval()
        null.score_bias.data.zero_()
        null.score_weight.data.fill_(1)
        scores = torch.zeros(2, 1, 2, 3)
        query = torch.ones(2, 1, 2, 3)
        query[:, :, 0] *= -1
        hidden = torch.zeros(1, 2, 3, dtype=torch.bool)
        mass = null.distribute(scores, query, hidden).sum(-1)
        self.assertTrue((mass[:, :, 0] > mass[:, :, 1]).all())
        temperature.score_bias.data.fill_(4)
        torch.testing.assert_close(temperature.distribute(scores, query, hidden).sum(-1), torch.ones_like(mass))

    def test_temperature_one_matches_pinned_position_expression(self):
        adapter = QueryAttention(self.base, 'temperature').eval()
        padding = torch.tensor([[True, False, False, False], [False, False, False, True]])
        mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
        # Make each query have at least one allowed real key.
        padding[:, 0] = False
        hidden = padding[:, None] | mask
        torch.testing.assert_close(adapter(self.x, self.pos, padding, mask),
                                   self.base.oracle(self.x, self.pos, hidden=hidden), atol=2e-7, rtol=2e-6)

    def test_streaming_real_key_cache_unchanged_and_zero_context(self):
        adapter = QueryAttention(self.base, 'temperature').eval()
        cached = torch.randn(3, 2, 6)
        pos = torch.randn(1, 10, 4)
        mask = torch.tensor([[True, True, False, False, False, False, False], [False] * 7])
        got, state = adapter.streaming_forward(self.x, pos, cached, 3, mask)
        expected = self.base.oracle(self.x, pos, cached, mask[:, None])
        torch.testing.assert_close(got, expected)
        keys = torch.cat([cached, self.base.in_proj(self.x)[..., 6:12]])
        torch.testing.assert_close(state, keys[-3:])
        _, empty = adapter.streaming_forward(self.x, self.pos, cached[:0], 0, None)
        self.assertEqual(tuple(empty.shape), (0, 2, 6))

    def test_masked_history_cannot_influence_output(self):
        adapter = QueryAttention(self.base, 'null').eval()
        cached = torch.randn(3, 2, 6)
        pos = torch.randn(1, 10, 4)
        mask = torch.tensor([[True] * 3 + [False] * 4] * 2)
        a, _ = adapter.streaming_forward(self.x, pos, cached, 3, mask)
        b, _ = adapter.streaming_forward(self.x, pos, cached * 100, 3, mask)
        torch.testing.assert_close(a, b)
        all_hidden = torch.ones(2, 4, 4, dtype=torch.bool)
        logits, q, _ = adapter.logits(self.x, self.pos)
        for mode in ('null', 'temperature'):
            adapter.mode = mode
            weights = adapter.distribute(logits, q, all_hidden)
            self.assertTrue(torch.equal(weights, torch.zeros_like(weights)))

    def test_finite_gradients_only_new_projection_and_freeze_install(self):
        class Model(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.encoders = nn.ModuleList()
                for _ in range(2):
                    stage = nn.Module()
                    stage.encoder = nn.Module()
                    stage.encoder.layers = nn.ModuleList()
                    for _ in range(2):
                        layer = nn.Module()
                        layer.self_attn_weights = copy.deepcopy(base)
                        stage.encoder.layers.append(layer)
                    self.encoder.encoders.append(stage)
                self.head = nn.Linear(8, 5)
        model = Model(self.base)
        adapter = install_query_attention(model, 'null')
        self.assertEqual(sum(isinstance(m, QueryAttention) for m in model.modules()), 1)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertEqual(len(names), 2)
        self.assertTrue(all(n.endswith(('score_weight', 'score_bias')) for n in names))
        original = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        loss = adapter(self.x, self.pos).square().sum()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters() if p.requires_grad))
        torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.1).step()
        for n, p in model.named_parameters():
            if n in original:
                self.assertIsNone(p.grad)
                self.assertTrue(torch.equal(p, original[n]))
        with self.assertRaisesRegex(ValueError, 'already installed'):
            install_query_attention(model, 'null')
        copy_model = Model(self.base)
        install_query_attention(copy_model, 'null')
        copy_model.load_state_dict(model.state_dict())
        torch.testing.assert_close(copy_model.encoder.encoders[-1].encoder.layers[-1].self_attn_weights(self.x, self.pos),
                                   adapter(self.x, self.pos))

    def test_training_mode_refused(self):
        adapter = QueryAttention(self.base, 'null')
        with self.assertRaisesRegex(ValueError, 'eval mode'):
            adapter(self.x, self.pos)

    def test_onnx_streaming_parity_with_fixed_zero_value(self):
        import numpy as np
        import onnx
        import onnxruntime as ort
        for mode in ('null', 'temperature'):
            adapter = QueryAttention(copy.deepcopy(self.base), mode).eval()
            adapter.score_weight.data.normal_()
            adapter.score_bias.data.zero_()
            class Stream(nn.Module):
                def __init__(self, attention):
                    super().__init__()
                    self.attention = attention
                def forward(self, x, pos, cached, mask):
                    return self.attention.streaming_forward(x, pos, cached, 3, mask)
            stream = Stream(adapter).eval()
            pos, cached = torch.randn(1, 10, 4), torch.randn(3, 2, 6)
            mask = torch.zeros(2, 7, dtype=torch.bool)
            with tempfile.TemporaryDirectory() as d:
                path = str(Path(d) / 'model.onnx')
                torch.onnx.export(stream, (self.x, pos, cached, mask), path, opset_version=17, dynamo=False,
                                  input_names=['x', 'pos', 'cached', 'mask'], output_names=['weights', 'new_key'])
                onnx.checker.check_model(onnx.load(path))
                session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                for count in (0, 3, 7):
                    mask[:, :count] = True
                    expected = stream(self.x, pos, cached, mask)
                    got = session.run(None, {'x':self.x.numpy(), 'pos':pos.numpy(), 'cached':cached.numpy(), 'mask':mask.numpy()})
                    for a, b in zip(expected, got):
                        np.testing.assert_allclose(a.detach().numpy(), b, rtol=3e-6, atol=3e-7)


if __name__ == '__main__':
    unittest.main()

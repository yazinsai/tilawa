"""Sampler invariants independent of Quran text, training or practice recordings."""
import io
import unittest

import numpy as np
import torch

from student_band_fusion import BandFusion, LOG_EPS, MELS, REGIONS


class BandFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_identity_and_equal_active_parameters(self):
        x = torch.randn(2, 45, MELS)
        x[:, :6] = LOG_EPS
        asynchronous, synchronous = BandFusion(), BandFusion(False)
        synchronous.load_state_dict(asynchronous.state_dict())
        for model in (asynchronous, synchronous):
            torch.testing.assert_close(model(x), x, rtol=0, atol=0)
            model(x).square().sum().backward()
            self.assertGreater(float(model.readout.weight.grad.abs().sum()), 0)
        self.assertEqual(sum(p.numel() for p in asynchronous.parameters()),
                         sum(p.numel() for p in synchronous.parameters()))
        self.assertTrue(all(p.requires_grad for p in synchronous.parameters()))

    def test_regions_delay_independently_and_never_read_future(self):
        model = BandFusion()
        with torch.no_grad():
            model.readout.bias.copy_(torch.atanh(torch.linspace(0.125, 0.875, REGIONS)))
        x = torch.arange(45).float()[None, :, None].expand(1, 45, MELS).clone()
        y = model(x)
        delay = 4 * torch.linspace(0.125, 0.875, REGIONS).repeat_interleave(10)
        torch.testing.assert_close(y[0, 8], x[0, 8] - delay)
        future_changed = x.clone()
        future_changed[:, 20:] = 1000
        torch.testing.assert_close(model(future_changed)[:, :20], y[:, :20], rtol=0, atol=0)

    def test_streaming_overlap_advances_only_committed_raw_prefix(self):
        model = BandFusion()
        with torch.no_grad():
            model.readout.weight.uniform_(-0.2, 0.2)
            model.readout.bias.fill_(0.4)
        x = torch.randn(1, 141, MELS)
        expected = model(x)
        history = model.initial_state()
        for offset in range(0, 97, 32):
            y, history = model.streaming_forward(x[:, offset:offset + 45], history, 32)
            torch.testing.assert_close(y, expected[:, offset:offset + 45], rtol=0, atol=1e-5)
            torch.testing.assert_close(history + LOG_EPS, x[:, offset + 28:offset + 32], rtol=0, atol=2e-6)
        # A new listener starts in silence rather than inheriting old history.
        y, _ = model.streaming_forward(x[:, :45], model.initial_state(), 32)
        torch.testing.assert_close(y, expected[:, :45], rtol=0, atol=1e-5)

    def test_padding_and_specaugment_do_not_become_sampled_speech(self):
        model = BandFusion()
        with torch.no_grad():
            model.readout.bias.fill_(0.5)
        x = torch.randn(2, 45, MELS)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, 10:14, 20:30] = True
        y = model(x, torch.tensor([24, 45]), mask)
        self.assertTrue((y[0, 24:] == LOG_EPS).all())
        self.assertTrue((y[:, 10:14, 20:30] == 0).all())

    def test_mixed_precision_preserves_long_recording_coordinates(self):
        model = BandFusion()
        with torch.no_grad():
            model.readout.bias.fill_(0.4)
        x = torch.arange(3000).float()[None, :, None].expand(1, 3000, MELS)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = model(x)
        # Under bf16 coordinate arithmetic all these interior delays would
        # round away, or jump by several frames at large positions.
        expected_delay = 4 * torch.tensor(0.4, dtype=torch.bfloat16).tanh().float()
        torch.testing.assert_close(y[:, 100:] - x[:, 100:],
                                   torch.full_like(y[:, 100:], -expected_delay), rtol=0, atol=3e-4)

    def test_sampler_onnx_state_and_noninteger_interpolation_parity(self):
        import onnxruntime as ort

        model = BandFusion().eval()
        with torch.no_grad():
            model.readout.bias.copy_(torch.linspace(-0.5, 1, REGIONS))
            model.readout.weight.uniform_(-0.1, 0.1)

        class Window(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sampler = model

            def forward(self, x, state):
                return self.sampler.streaming_forward(x, state, 32)

        window = Window()
        x = torch.randn(1, 45, MELS)
        state = model.initial_state()
        data = io.BytesIO()
        torch.onnx.export(window, (x, state), data, opset_version=17, dynamo=False,
                          input_names=["x", "history"], output_names=["features", "new_history"])
        session = ort.InferenceSession(data.getvalue(), providers=["CPUExecutionProvider"])
        for _ in range(4):
            expected, new_state = window(x, state)
            actual = session.run(None, {"x": x.numpy(), "history": state.numpy()})
            np.testing.assert_allclose(actual[0], expected.detach().numpy(), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(actual[1], new_state.numpy(), rtol=0, atol=0)
            x, state = torch.randn_like(x), new_state.detach()


if __name__ == "__main__":
    unittest.main()

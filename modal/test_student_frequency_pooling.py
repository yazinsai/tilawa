import unittest

import torch
from torchaudio.compliance import kaldi

from student_frequency_pooling import EPS, FrequencyPooling, fft_powers


class PoolingTest(unittest.TestCase):
    def test_identity_matches_training_features_including_quiet_onsets(self):
        torch.manual_seed(1)
        wave = torch.cat((torch.zeros(2400), torch.randn(3200) * 1e-4, torch.randn(3200) * 0.1))
        actual = FrequencyPooling()(fft_powers(wave))
        expected = kaldi.fbank(wave[None], num_mel_bins=80, sample_frequency=16000,
                               dither=0, energy_floor=1e-10, high_freq=-400)
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)

    def test_training_preserves_positive_contiguous_support_and_mass(self):
        pool = FrequencyPooling()
        initial = pool.weights().detach().clone()
        power = torch.rand(8, 257, requires_grad=True)
        pool(power).square().mean().backward()
        self.assertGreater(pool.delta.grad.abs().sum().item(), 0)
        with torch.no_grad():
            pool.delta.sub_(pool.delta.grad * 5)
        self.assertFalse(torch.equal(initial, pool.weights()))
        for row, allowed in zip(pool.weights(), pool.support):
            bins = allowed.nonzero().flatten()
            self.assertEqual(len(bins), int(bins[-1] - bins[0] + 1))
            self.assertTrue((row[allowed] > 0).all())
            self.assertTrue((row[~allowed] == 0).all())
        torch.testing.assert_close(pool.weights().sum(-1), initial.sum(-1))

    def test_export_reproduces_trained_filters_and_floor(self):
        pool = FrequencyPooling()
        with torch.no_grad():
            pool.delta.normal_(0, 0.4)
        restored = torch.zeros(80, 257)
        for m, f in enumerate(pool.export_contract()["filters"]):
            restored[m, f["start"]:f["start"] + len(f["weights"])] = torch.tensor(f["weights"]) * f["scale"]
        power = torch.rand(7, 257)
        torch.testing.assert_close(pool(power), (power @ restored.t()).clamp_min(EPS).log())
        self.assertTrue((pool(torch.zeros(1, 257)) == torch.tensor(EPS).log()).all())
        self.assertFalse(FrequencyPooling(False).delta.requires_grad)


if __name__ == "__main__":
    unittest.main()

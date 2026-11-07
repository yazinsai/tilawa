import unittest
import numpy as np
from student_spectral_residual import clean_help, mel_matrix, null_projection, frontend, forced_labels, fit_parameters


class ResidualTests(unittest.TestCase):
    def test_only_verified_training_speakers(self):
        row = dict(split="train", kind="none", scored=True, speaker="training", events=[], non_errors=[])
        self.assertTrue(clean_help(row))
        for change in (dict(split="test"), dict(split="validation"), dict(kind="substituted_word"),
                       dict(events=[{}]), dict(non_errors=[{}]), dict(excluded=True), dict(speaker="")):
            self.assertFalse(clean_help(dict(row, **change)))

    def test_detail_cannot_change_any_mel_energy(self):
        w = mel_matrix(); q = null_projection(w)
        self.assertGreater(q.shape[1], 100)
        np.testing.assert_allclose(w @ q, 0, atol=1e-12)
        np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-12)
        self.assertTrue(np.all(q[w.sum(0) == 0] == 0))
        power = np.ones(256); perturbation = q @ np.arange(q.shape[1]) * 1e-6
        np.testing.assert_allclose(w @ power, w @ (power + perturbation), atol=1e-12)

    def test_frontend_normalization_and_repeated_phone_alignment(self):
        wave = np.random.default_rng(3).normal(0, .1, 400 + 160 * 4).astype(np.float32)
        mel, power, condition = frontend(wave, 16000)
        self.assertEqual(len(mel), 5)
        np.testing.assert_allclose(condition[:, 1:].sum(1), 1)
        np.testing.assert_allclose(power @ mel_matrix().T, condition[:, 1:], atol=1e-7)
        lp = np.full((3, 2), -100.); lp[np.arange(3), [1, 0, 1]] = 0
        np.testing.assert_array_equal(forced_labels(lp, [1, 1]), [1, 0, 1])

    def test_unseen_phone_abstains_and_covariance_remains_finite(self):
        rng = np.random.default_rng(2)
        q = null_projection(mel_matrix())[:, :4]
        records = [dict(mel=np.c_[np.ones(90), rng.normal(size=(90, 80))],
                        power=rng.normal(size=(90, 256)), y=np.tile([0, 1, 1], 30))]
        p, ld = fit_parameters(records, q, 1e-4, 10, 3)
        self.assertEqual(p["covered"], [True, True, False])
        self.assertTrue(np.isfinite(ld))
        self.assertTrue(np.isfinite(p["means"]).all())


if __name__ == "__main__":
    unittest.main()

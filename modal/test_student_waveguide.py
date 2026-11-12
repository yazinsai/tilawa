"""Small physical invariants and evidence-integrity tests; run on Modal."""
import unittest

import numpy as np
import torch

import student_waveguide as w


class WaveguideTest(unittest.TestCase):
    def test_delay_line_matches_travelling_wave_equations(self):
        z = torch.exp(-2j*torch.pi*torch.arange(129)/256)
        area = torch.tensor(2.); length = torch.tensor(3.); loss = torch.tensor(.03)
        yd, yo = w.tube(area, length, loss, z)
        a = (1-length*w.RATE/35000)/(1+length*w.RATE/35000)
        q = torch.exp(-loss*length)*(a+z)/(1+a*z)
        # Unit left travelling wave and independently propagated reflected wave.
        forward = torch.ones_like(z); backward = .4*torch.ones_like(z)
        left_p = forward+q*backward; right_p = q*forward+backward
        left_u = area*(forward-q*backward)
        right_u = area*(backward-q*forward)
        torch.testing.assert_close(yd*left_p+yo*right_p, left_u)
        torch.testing.assert_close(yo*left_p+yd*right_p, right_u)
        self.assertTrue(torch.all(q.abs() < 1))

    def test_physical_gradients_and_finite_reconstruction(self):
        u = torch.zeros(4, 10, requires_grad=True)
        voiced, turbulent = w.transfer(u)
        loss = voiced.abs().square().mean()+turbulent.abs().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(u.grad).all())
        self.assertGreater(float(u.grad.abs().sum()), 0.)
        impulse = torch.fft.irfft(voiced, n=256)
        self.assertTrue(torch.isfinite(impulse).all())

    def test_nasal_branch_and_oral_constriction_change_response(self):
        base = torch.zeros(1, 10)
        closed = base.clone(); closed[:, 6] = -12
        open_velum = base.clone(); open_velum[:, 6] = 8
        narrowed = base.clone(); narrowed[:, 3] = -8
        h0, t0 = w.transfer(base)
        h1, _ = w.transfer(closed); h2, _ = w.transfer(open_velum)
        _, t1 = w.transfer(narrowed)
        self.assertGreater(float((h1-h2).abs().mean()), .01)
        self.assertGreater(float((t0-t1).abs().mean()), .01)

    def test_complete_sequence_normalization_and_no_missing_phone(self):
        zero = torch.zeros(8, 3, 5)
        self.assertAlmostEqual(float(w.sequence_score(zero)), 0., places=5)
        impossible = w.sequence_score(torch.zeros(2, 3, 5))
        self.assertTrue(torch.isneginf(impossible))
        # One frame per phone is an allowed complete pronunciation.
        e = torch.tensor([[[2.], [-9.]], [[-9.], [3.]]])
        self.assertAlmostEqual(float(w.sequence_score(e)), 5., places=5)

    def test_same_realization_is_held_for_a_phone(self):
        e = torch.tensor([[[3., -3.]], [[-3., 3.]], [[3., -3.]]])
        # Two possible whole runs have totals +/-3, rather than the framewise
        # winner's fabricated total 9.
        expected = np.logaddexp(3., -3.)-np.log(2.)
        self.assertAlmostEqual(float(w.sequence_score(e)), expected, places=5)

    def test_nuisance_excludes_disputed_and_following_audio(self):
        rng = np.random.default_rng(8)
        audio = rng.normal(0, .1, 4*w.RATE).astype(np.float32)
        before = 2*w.RATE
        first = w.nuisance(audio, before)
        changed = audio.copy(); changed[before:] = rng.normal(0, 2, len(changed)-before)
        self.assertEqual(first, w.nuisance(changed, before))
        self.assertIsNone(w.nuisance(audio, 20))

    def test_geometry_always_positive_and_bounded(self):
        p = w.geometry(torch.randn(100, 10)*100).numpy()
        self.assertTrue(np.all(p >= w.LOW*(1-1e-6)))
        self.assertTrue(np.all(p <= w.HIGH*(1+1e-6)))


if __name__ == '__main__':
    unittest.main()

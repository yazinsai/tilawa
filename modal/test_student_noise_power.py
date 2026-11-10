"""Numerical/provenance checks; synthetic inputs only, no scored or learner recordings."""
import itertools
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import ncx2
from student_noise_power import chain, fit, noisy_logp, prepare


class NoisePowerTest(unittest.TestCase):
    def test_exact_noncentral_density_and_normalization(self):
        for y, s, n in [(0.1, 0.2, 1), (1, 0.01, 1), (2, 3, .1), (100, 90, 2)]:
            expected = ncx2.logpdf(2*y/n, 2, 2*s/n)+np.log(2/n)
            self.assertAlmostEqual(float(noisy_logp(y, s, n)), expected, places=10)
        value = quad(lambda y: float(np.exp(noisy_logp(y, .5, .7))), 0, np.inf)[0]
        self.assertAlmostEqual(value, 1, places=9)

    def test_complete_optional_noise_boundary_marginal(self):
        e = np.array([[.1, -.2], [.3, .4], [-.1, .5]])
        background = np.array([-.3, .2, -.4])
        paths = []
        for path in itertools.product(range(5), repeat=3):
            if path[0] not in (0, 1) or path[-1] not in (3, 4):
                continue
            if any(b<a or b>a+2 or (b==a+2 and b%2!=1) for a,b in zip(path, path[1:])):
                continue
            paths.append(sum(background[t] if s%2==0 else e[t, (s-1)//2] for t,s in enumerate(path)))
        expected = np.logaddexp.reduce(paths)-np.log(len(paths))
        score, gamma = chain(e, True, background)
        self.assertAlmostEqual(score, expected, places=12)
        self.assertTrue((gamma.sum(1) <= 1+1e-12).all())
        self.assertTrue((gamma.sum(0) >= 1-1e-12).all())

    def test_finite_small_em_fit(self):
        rng = np.random.default_rng(1729)
        rows = []
        for speaker in range(3):
            clean = np.exp(rng.normal(0, .2, (20, 16)))
            noise = rng.normal(0, .1, (20, 16))+1j*rng.normal(0, .1, (20, 16))
            rows.append(dict(power=abs(np.sqrt(clean)+noise)**2, noise=np.full(16, .02), ids=[1,2], speaker=str(speaker)))
        model, history = fit(rows, 'بت', 8, epochs=2)
        self.assertTrue(np.isfinite(history).all())
        self.assertTrue((model['deviations']>0).all())
        self.assertTrue((model['counts'][1:]>0).all())

    def test_rejects_test_sources_and_excluded_speakers_before_audio(self):
        for row in [dict(source='practice', split='train', authorized=True, speaker='s'),
                    dict(source='help-train', kind='substituted_word', split='train', authorized=True, speaker='s'),
                    dict(source='quran-training', transcriptVerified=True, split='train', authorized=True, speaker='heldout')]:
            with self.assertRaises(ValueError):
                prepare([row], 'بت', {'heldout'})


if __name__ == '__main__':
    unittest.main()

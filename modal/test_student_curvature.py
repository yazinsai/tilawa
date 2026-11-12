"""Small numerical tests; no training data or practice cases."""
import unittest
import numpy as np
from student_curvature import chain, view, prepare, fit


class CurvatureTests(unittest.TestCase):
    def test_gain_and_tilt(self):
        x = np.sin(np.arange(16)[None, :] * .7)
        np.testing.assert_allclose(view(x, True), view(x+2.4-.13*np.arange(16), True), atol=1e-12)

    def test_complete_path_posterior(self):
        rows = -.2*(np.arange(6)[:, None]-np.arange(3)[None, :])**2
        for labels in ([1], [1, 2], [1, 1], [2, 1, 2]):
            value, states, posterior = chain(rows, labels, True)
            np.testing.assert_allclose(posterior.sum(1), 1, atol=1e-12)
            # Enumerate paths independently and compare total density / timing.
            paths = [(0, .5, np.exp(rows[0, 0])), (1, .5, np.exp(rows[0, labels[0]]))]
            for t in range(1, len(rows)):
                following = []
                for s, prior, density in paths:
                    targets = [s]
                    if s+1 < len(states): targets.append(s+1)
                    if s+2 < len(states) and states[s+2] and states[s+2] != states[s]: targets.append(s+2)
                    for d in targets:
                        p = 1 if len(targets) == 1 else .8 if d == s else .2/(len(targets)-1)
                        following.append((d, prior*p, density*np.exp(rows[t, states[d]])))
                paths = following
            accepting = [(p, d) for s, p, d in paths if s >= len(states)-2]
            expected = np.log(sum(p*d for p,d in accepting)/sum(p for p,d in accepting))
            self.assertAlmostEqual(value, expected, 12)
            self.assertAlmostEqual(chain(np.zeros_like(rows), labels), 0, 12)

    def test_repeated_phone_cannot_be_skipped(self):
        with self.assertRaises(ValueError): chain(np.zeros((2, 2)), [1, 1], True)

    def test_training_source_guard(self):
        base = dict(source='help-train', kind='none', split='train', speaker='s', excluded=False,
                    pronunciation='ab', features={str(k): np.zeros((8, k)) for k in (16, 24, 32)})
        for change, excluded in ((dict(source='practice'), set()), (dict(kind='wrong_word'), set()),
                                 (dict(split='heldout'), set()), (dict(excluded=True), set()), ({}, {'s'})):
            with self.assertRaisesRegex(ValueError, 'Only none-labelled'):
                prepare([{**base, **change}], 'ab', excluded)

    def test_soft_em_has_finite_equal_capacity_arms(self):
        rng = np.random.default_rng(7)
        rows = [dict(speaker=str(s), ids=[1, 2], pronunciation='ab',
                     features={'16': rng.normal(size=(8, 16))}) for s in range(3)]
        model, history = fit(rows, 'ab', 16, 32)
        for arm in ('control', 'curvature'):
            self.assertTrue(np.isfinite(history[arm]).all())
            for emission in model[arm]['emissions']:
                self.assertEqual(len(emission['mean']), 14)
                self.assertTrue(np.isfinite(emission['mean']).all())
                self.assertTrue((np.array(emission['variance']) > 0).all())


if __name__ == '__main__': unittest.main()

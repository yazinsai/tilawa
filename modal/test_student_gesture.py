import itertools
import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logsumexp

from student_gesture import evaluate, mixtures, objective, prepare, sources


class GestureTests(unittest.TestCase):
    def model(self):
        rng = np.random.default_rng(4)
        spectra = rng.random((9, 20))+.1
        spectra /= spectra.sum(1, keepdims=True)
        return dict(spectra=spectra.tolist(), variances=np.full((9, 20), .5).tolist(),
                    stay=np.linspace(.4, .8, 9).tolist(), weights=[.2, .4])

    def test_exact_sum_over_every_boundary(self):
        model = self.model()
        x = np.log(np.asarray(model['spectra'])[[3, 4, 4, 5, 6, 7, 7, 8]])
        for overlap in [True, False]:
            states, left, right, phase = sources([1, 2], overlap)
            mean = np.log(mixtures(np.asarray(model['spectra']), model['weights'], states, left, right, phase))
            paths, priors = [], []
            for increments in itertools.product([0, 1], repeat=len(x)-1):
                path = np.r_[0, np.cumsum(increments)]
                if path[-1] != 5:
                    continue
                prior = sum(np.log(1-model['stay'][states[s]] if a else model['stay'][states[s]])
                            for s, a in zip(path[:-1], increments))
                density = -.5*np.sum(np.log(np.pi)+(x-mean[path])**2/.5)
                priors.append(prior); paths.append(prior+density)
            self.assertAlmostEqual(evaluate(model, x, [1, 2], overlap), logsumexp(paths)-logsumexp(priors), places=10)
            _, gamma, transitions = evaluate(model, x, [1, 2], overlap, True)
            np.testing.assert_allclose(gamma.sum(1), 1, atol=1e-10)
            self.assertAlmostEqual(transitions.sum(), len(x)-1, places=9)

    def test_joint_emission_gradient(self):
        model = self.model()
        spectra = np.asarray(model['spectra'])
        design = np.column_stack(sources([1, 2], True))
        x = np.log(spectra[[3, 4, 5, 6, 7, 8]])+.1
        params = np.r_[np.log(spectra).ravel(), np.log(model['weights'])]
        args = (design, x, np.linspace(.1, 1, 6), np.asarray(model['variances']), spectra, 8)
        numeric = approx_derivative(lambda p: objective(p, *args)[0], params).ravel()
        np.testing.assert_allclose(objective(params, *args)[1], numeric, atol=1e-5, rtol=1e-4)

    def test_no_phones_are_erased(self):
        with self.assertRaisesRegex(ValueError, 'Impossible'):
            evaluate(self.model(), np.full((5, 20), -np.log(20)), [1, 1], True)

    def test_training_rejects_scripts_and_held_out_speakers(self):
        row = dict(source='help-train', kind='none', split='train', excluded=False,
                   speaker='training', pronunciation='a', features=np.full((3, 20), -np.log(20)).tolist())
        for change in [dict(kind='wrong_word'), dict(split='test'), dict(excluded=True), dict(source='practice')]:
            with self.assertRaises(ValueError):
                prepare([{**row, **change}], 'ab', set())
        with self.assertRaises(ValueError):
            prepare([row], 'ab', {'training'})


if __name__ == '__main__':
    unittest.main()

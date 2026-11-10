import unittest

import numpy as np
from scipy.optimize import check_grad
from scipy.special import logsumexp

from student_reverberant import DIMENSIONS, emission_objective, evaluate, fit, kernel, prepare


def toy():
    spectra = np.full((4, DIMENSIONS), .2/(DIMENSIONS-1))
    spectra[0] = 1/DIMENSIONS
    spectra[1, 0] = .8
    spectra[2, 1] = .8
    spectra[3] = .7*spectra[1]+.3*spectra[2]
    return dict(spectra=spectra.tolist(), variances=np.full_like(spectra, .1).tolist(),
        initial=[.1, .3, .4, .2], stay=[.6, .7, .8, .9], mix=.7, decay=.9)


class ReverberantTests(unittest.TestCase):
    def test_exact_path_and_incoming_memory_marginals(self):
        m = toy(); ids = [1, 2]
        x = np.log(np.array(m['spectra'])[[1, 1, 1, 3, 2, 2, 2]])
        acoustic, prior, states = [], [], []
        for boundary in range(1, len(x)):
            path = [0]*boundary+[1]*(len(x)-boundary)
            for seed, prob in enumerate(m['initial']):
                lp = np.log(prob)
                value = 0
                for t, i in enumerate(path):
                    if t:
                        stay = m['stay'][ids[path[t-1]]]
                        lp += np.log(stay if i == path[t-1] else 1-stay)
                    phones = [ids[i]]+[seed if t < l else ids[path[t-l]] for l in range(1, 5)]
                    mean = np.log((np.array(m['spectra'])[phones]*kernel(m['mix'], m['decay'])[:, None]).sum(0))
                    var = np.array(m['variances'])[ids[i]]
                    value -= .5*np.sum(np.log(2*np.pi*var)+(x[t]-mean)**2/var)
                acoustic.append(lp+value); prior.append(lp); states.append(path)
        score, gamma, identities, transitions = evaluate(m, x, ids, True)
        self.assertAlmostEqual(score, logsumexp(acoustic)-logsumexp(prior), places=10)
        masses = np.exp(acoustic-logsumexp(acoustic))
        for t, g in enumerate(gamma):
            self.assertAlmostEqual(g.sum(), 1., places=10)
            for i, p in enumerate(ids):
                actual = g[identities[t][:, 0] == p].sum()
                expected = sum(w for w, path in zip(masses, states) if path[t] == i)
                self.assertAlmostEqual(actual, expected, places=10)
        self.assertAlmostEqual(transitions[:, 1].sum(), 1., places=10)
        self.assertAlmostEqual(transitions.sum(), len(x)-1, places=10)

    def test_control_and_memory_match_when_convolution_is_disabled(self):
        m = toy(); m['mix'] = 0
        x = np.log(np.array(m['spectra'])[[1, 1, 2, 2, 2, 2]])
        self.assertAlmostEqual(evaluate(m, x, [1, 2], memory=False), evaluate(m, x, [1, 2]), places=10)

    def test_normalized_emission_gradient_including_decay(self):
        m = toy(); rng = np.random.default_rng(11)
        spectra = np.array(m['spectra'])
        phones = rng.integers(0, 4, (12, 5)); x = np.log(spectra[rng.integers(0, 4, 12)])
        weights = rng.uniform(.1, 1, 12)
        params = np.r_[np.log(spectra).ravel(), -.5, .8]
        args = (phones, x, weights, np.array(m['variances']), spectra, 8)
        self.assertLess(check_grad(lambda v: emission_objective(v, *args)[0],
            lambda v: emission_objective(v, *args)[1], params, epsilon=1e-6), .003)

    def test_training_provenance_and_small_fit(self):
        m = toy(); x = np.log(np.array(m['spectra'])[[1, 1, 2, 2, 2, 2]])
        raw = [dict(source='help-train', kind='none', split='train', excluded=False,
            speaker=s, pronunciation='ab', features=x.tolist()) for s in ['one', 'two', 'three']]
        rows = prepare(raw, 'abc', set())
        with self.assertRaises(ValueError): prepare(raw, 'abc', {'one'})
        for field, value in [('kind', 'substituted_word'), ('split', 'eval'), ('excluded', True), ('source', 'practice')]:
            bad = [dict(r, **{field: value}) for r in raw]
            with self.assertRaises(ValueError): prepare(bad, 'abc', set())
        fitted, history = fit(rows, 'abc', 8, epochs=1)
        self.assertTrue(np.isfinite(history[0]['objective']))
        np.testing.assert_allclose(np.array(fitted['spectra']).sum(1), 1, atol=1e-10)
        self.assertTrue(np.all(np.array(fitted['variances']) > 0))
        self.assertTrue(0 < fitted['mix'] < 1 and 0 < fitted['decay'] < 1)


if __name__ == '__main__':
    unittest.main()

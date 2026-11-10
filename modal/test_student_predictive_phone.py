"""Small numerical/provenance unit tests; no recording or practice data."""
import itertools
import unittest
import numpy as np
from student_predictive_phone import chain, covariance, fit, prepare, predictive_validation


class PredictivePhoneTests(unittest.TestCase):
    def test_boundary_marginal_matches_exhaustive_enumeration(self):
        emissions = np.array([[.1, .8], [.2, .3], [.9, .4], [.1, .7]])
        stay = .8
        mass, prior = [], []
        for boundary in range(1, len(emissions)):
            path = [0]*boundary+[1]*(len(emissions)-boundary)
            p = np.log1p(-stay)+(len(emissions)-2)*np.log(stay)
            prior.append(p)
            mass.append(p+sum(emissions[t, s] for t, s in enumerate(path)))
        value, gamma = chain(emissions, stay, True)
        self.assertAlmostEqual(value, np.logaddexp.reduce(mass)-np.logaddexp.reduce(prior))
        np.testing.assert_allclose(gamma.sum(1), 1)

    def test_rejects_mistake_scripts_eval_and_excluded_speakers(self):
        rows = [dict(kind='none', split='train', source='help-train', excluded=False,
            speaker=str(s), pronunciation='بت', features=np.zeros((6, 8)).tolist()) for s in range(3)]
        self.assertEqual(len(prepare(rows, 'بت', set())), 3)
        for key, value in [('kind', 'wrong_word'), ('split', 'eval'), ('excluded', True), ('source', 'practice')]:
            changed = [dict(rows[0], **{key: value}), *rows[1:]]
            with self.assertRaises(ValueError):
                prepare(changed, 'بت', set())
        with self.assertRaises(ValueError):
            prepare(rows, 'بت', {'0'})

    def test_toy_fit_has_supported_positive_definite_posteriors(self):
        rng = np.random.default_rng(7)
        rows = [dict(x=np.r_[rng.normal(0,.2,(6,8)), rng.normal(2,.2,(6,8))], ids=[1,2], speaker=str(s)) for s in range(3)]
        model, history = fit(rows, 'بت', 2, epochs=2)
        self.assertTrue(np.isfinite(history).all())
        for p in model['phones']:
            np.linalg.cholesky(covariance(p))
        self.assertGreater(model['phones'][1]['count'], 0)
        self.assertTrue(np.isfinite(predictive_validation(model, rows[:1], draws=4)))


if __name__ == '__main__':
    unittest.main()

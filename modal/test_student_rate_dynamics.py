import unittest

import numpy as np
from scipy.special import logsumexp

from student_rate_dynamics import DIMENSIONS, emissions, evaluate, objective, prepare


class RateDynamicsTests(unittest.TestCase):
    def setUp(self):
        self.model = dict(targets=[np.zeros(20).tolist(), np.ones(20).tolist(), (-np.ones(20)).tolist()],
            shared=np.full(20, .6).tolist(), innovation=np.ones(20).tolist())
        self.x = np.tile(np.array([.2, .5, .8, -.1, -.4])[:, None], (1, 20))

    def test_exact_marginal_and_occupancies(self):
        for dynamic in [False, True]:
            e = emissions(self.model, self.x, [1, 2], dynamic)
            paths = [[0]*cut+[1]*(5-cut) for cut in range(1, 5)]
            values = [sum(e[t, s] for t, s in enumerate(path)) for path in paths]
            value, gamma = evaluate(self.model, self.x, [1, 2], dynamic, True)
            self.assertAlmostEqual(value, logsumexp(values)-np.log(len(paths)))
            expected = np.zeros((5, 2))
            for path, weight in zip(paths, np.exp(values-logsumexp(values))):
                for t, s in enumerate(path): expected[t, s] += weight
            np.testing.assert_allclose(gamma, expected, atol=1e-12)
            np.testing.assert_allclose(gamma.sum(1), 1)

    def test_target_approach_accumulates_real_duration(self):
        target, rho, state = 1., .6, 0.
        short = []
        for t in range(8):
            state = rho*state+(1-rho)*target; short.append(state)
        np.testing.assert_allclose(short, 1-rho**np.arange(1, 9))
        self.assertLess(short[0], short[-1])
        # Control emissions for an identical spectrum do not depend on the
        # preceding observation, whereas treatment transitions do.
        a = emissions(self.model, self.x, [1], False, np.zeros(20))
        b = emissions(self.model, self.x, [1], False, np.ones(20))
        np.testing.assert_array_equal(a, b)
        self.assertNotEqual(emissions(self.model, self.x, [1], True, np.zeros(20))[0,0],
            emissions(self.model, self.x, [1], True, np.ones(20))[0,0])

    def test_analytic_gradients_both_equal_capacity_arms(self):
        rng = np.random.default_rng(5)
        x, before = rng.normal(size=(12, 20)), rng.normal(size=(12, 20))
        phone, mass = np.arange(12)%3, rng.uniform(.1, 1, 12)
        for dynamic in [False, True]:
            params = np.r_[rng.normal(size=60), np.full(20, .3)]
            args = (x, before, phone, mass, np.ones(20), np.zeros((3,20)), 8, dynamic)
            numerical = np.array([(objective(params+np.eye(80)[i]*1e-5, *args)[0]-
                objective(params-np.eye(80)[i]*1e-5, *args)[0])/2e-5 for i in range(80)])
            np.testing.assert_allclose(objective(params, *args)[1], numerical, rtol=1e-6, atol=1e-6)
            self.assertEqual(len(objective(params, *args)[1]), 80)

    def test_acceptance_requires_all_repeated_phones(self):
        with self.assertRaises(ValueError): evaluate(self.model, self.x[:1], [1, 1], True)
        value = evaluate(self.model, self.x[:2], [1, 1], True)
        e = emissions(self.model, self.x[:2], [1,1], True)
        self.assertAlmostEqual(value, e[0,0]+e[1,1])

    def test_training_inputs_fail_closed(self):
        base = dict(source='help-train', kind='none', split='train', excluded=False,
            pronunciation='ab', features=np.full((3,20), np.log(.05)).tolist())
        raw = [dict(base, speaker=str(s)) for s in range(3)]
        self.assertEqual(len(prepare(raw, 'ab', set())), 3)
        for key, value in [('source','practice'), ('kind','wrong_word'), ('split','eval'), ('excluded',True)]:
            with self.assertRaises(ValueError): prepare([dict(r, **{key:value}) for r in raw], 'ab', set())
        with self.assertRaises(ValueError): prepare(raw, 'ab', {'1'})


if __name__ == '__main__': unittest.main()

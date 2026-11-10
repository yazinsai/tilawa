import itertools
import unittest

import numpy as np
from student_quotients import DIMENSIONS, fit, gaussian, lattice, prepare, units


class QuotientTests(unittest.TestCase):
    def test_exact_latent_segment_sum(self):
        def emission(values):
            return dict(mean=[x if d == 0 else 0 for x in values for d in range(DIMENSIONS)],
                        variance=[1]*DIMENSIONS*len(values))
        model = dict(scale=[1]*DIMENSIONS, maxPart=2, maxFrames=1024,
            absolute={'بَ': emission([5]), 'تَ': emission([0])},
            quotient={'بَ': emission([0]), 'تَ': emission([-2])})
        raw = np.zeros((6, DIMENSIONS)); raw[:, 0] = [0, .5, 1, 2, 2, 1]
        for relative in [False, True]:
            scores, priors = [], []
            for durations in itertools.product([1, 2], repeat=4):
                if sum(durations) != len(raw):
                    continue
                ends = np.r_[0, np.cumsum(durations)]
                score, timing = 0, 0
                for i, u in enumerate(['بَ', 'تَ']):
                    a, c, b = ends[2*i:2*i+3]
                    left, right = raw[a:c].mean(0), raw[c:b].mean(0)
                    x = left-right if relative else left
                    score += gaussian(x, model['quotient' if relative else 'absolute'][u])
                    cuts = min(b-a-1, 2)-max(1, b-a-2)+1
                    timing -= np.log(3)+np.log(cuts)
                scores.append(score+timing); priors.append(timing)
            expected = np.logaddexp.reduce(scores)-np.logaddexp.reduce(priors)
            self.assertAlmostEqual(lattice(model, raw, 'بَتَ', relative)[0], expected, places=10)
            coloured = raw+np.sin(np.arange(DIMENSIONS))[None]*8
            if relative:
                self.assertAlmostEqual(lattice(model, coloured, 'بَتَ', relative)[0], expected, places=10)

    def test_training_provenance(self):
        rows = [dict(speaker=str(s),kind='none',split='train',excluded=False,
            pronunciation='بَتَ',features=np.zeros((8, DIMENSIONS)).tolist()) for s in range(3)]
        self.assertEqual(len(prepare(rows, 'بتَ', set())), 3)
        for field, value in [('kind', 'substituted_word'), ('split', 'eval'), ('excluded', True)]:
            changed = [dict(r) for r in rows];changed[0][field] = value
            with self.assertRaises(ValueError): prepare(changed, 'بتَ', set())
        with self.assertRaises(ValueError): prepare(rows, 'بتَ', {'1'})

    def test_small_latent_fit(self):
        rows = [dict(speaker=str(s), pronunciation='بَتَ', x=np.tile(np.arange(8)[:, None], (1, DIMENSIONS))/4+s/10) for s in range(3)]
        model, history = fit(rows, 'بتَ', 'toy', 8, epochs=2)
        self.assertEqual(len(history), 2)
        self.assertEqual(units('ببََتَ'), ['ب', 'بَ', 'َ', 'تَ'])
        for arm in ['quotient', 'absolute']:
            for emission in model[arm].values():
                self.assertTrue(np.isfinite(emission['mean']).all())
                self.assertTrue((np.array(emission['variance']) > 0).all())


if __name__ == '__main__': unittest.main()

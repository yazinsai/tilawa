"""Small estimator contract checks; run on Modal, never on training or practice labels."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path = Path('packages/core/src/word-rhythm-fit.py')
spec = importlib.util.spec_from_file_location('fit', path)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


class Contracts(unittest.TestCase):
    def test_training_speaker_exclusion(self):
        rows = [dict(id='ok', speaker='a', split='train', kind='none'),
                dict(id='leak', speaker='b', split='train', kind='none'),
                dict(id='held', speaker='b', split='heldout', kind='none'),
                dict(id='unsplit', speaker='c', kind='none'),
                dict(id='repeat', speaker='d', split='train', kind='none', non_errors=[{}]),
                dict(id='event', speaker='e', split='train', kind='none', events=[{}])]
        self.assertEqual([r['id'] for r in fit.training_labels(rows)], ['ok'])

    def test_nucleus_structure_matches_runtime(self):
        st = fit.structure('بَااتِكدُ', [(0, 9)])
        self.assertEqual(st['starts'], [1, 5, 8])
        self.assertEqual(st['links'], [-1, 4, 1])

    def test_augmented_graph_preserves_identity_mass(self):
        ref, obs = 'بَتِدُ', list('بَكِد')
        closure, ins, mat, alpha, beta, scales = fit.graph(ref, obs)
        st = fit.structure(ref, [(0, 6)])
        states = [{(-1, -1): v} for v in alpha[0]]
        for t in range(len(obs)):
            states = fit.advance(states, t, st, ins[t], mat[t], closure)
            for cell in states:
                for key in cell:
                    cell[key] /= scales[t]
            np.testing.assert_allclose([sum(c.values()) for c in states], alpha[t+1], atol=1e-12)

    def test_full_timing_partition_with_independent_marginals(self):
        # With equal duration classes and rho zero the timing multiplier is shared by every identity path.
        fit.HORIZON = 3
        ref, obs = 'بَتِ', list('بَت')
        closure, ins, mat, alpha, beta, scales = fit.graph(ref, obs)
        r = dict(st=fit.structure(ref, [(0, 4)]), x=np.array([-.5, .1, .4]),
                 initial=alpha[0], closure=closure, ins=ins, mat=mat)
        got = fit.likelihood(r, np.zeros(8), np.ones(8), np.zeros(8))
        expected = sum(np.log(scales))-.5*sum(r['x']**2)
        self.assertAlmostEqual(got, expected, places=10)


if __name__ == '__main__':
    unittest.main()

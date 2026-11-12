"""Exhaustive CTC checks and safeguards for native angular supervision."""
import itertools
import unittest
import numpy as np
from student_angular_phone import occupancies, prepare, provenance, reconstruct, sequence_score, speaker_folds


def collapse(path):
    return [p for i, p in enumerate(path) if p and (i == 0 or p != path[i-1])]


class AngularTest(unittest.TestCase):
    def test_soft_occupancies_and_scores_match_every_ctc_path(self):
        probs = np.array([[.4, .5, .1], [.3, .2, .5], [.2, .4, .4], [.3, .3, .4]])
        for ids in ([1, 2], [1, 1]):
            expected = np.zeros_like(probs)
            z = 0.
            for path in itertools.product(range(3), repeat=4):
                if collapse(path) != ids:
                    continue
                mass = np.prod([probs[t, p] for t, p in enumerate(path)])
                z += mass
                for t, p in enumerate(path):
                    expected[t, p] += mass
            targets = occupancies(np.log(probs), ids)
            np.testing.assert_allclose(targets, expected/z, atol=1e-7)
            self.assertAlmostEqual(sequence_score(np.log(probs), ids), np.log(z))

    def test_genuine_repetition_requires_blank(self):
        self.assertIsNone(occupancies(np.log(np.full((2, 3), 1/3)), [1, 1]))
        targets = occupancies(np.log(np.full((3, 3), 1/3)), [1, 1])
        np.testing.assert_array_equal(targets, [[0, 1, 0], [1, 0, 0], [0, 1, 0]])

    def test_conditional_targets_retain_soft_occupancy_and_speech_weights(self):
        lp = np.log(np.full((4, 3), 1/3))
        rows, skipped = prepare([dict(id='a', features=[[1, 2]]*4, logProbs=lp, pronunciation=[1, 2])])
        self.assertFalse(skipped)
        full = occupancies(lp, [1, 2])
        np.testing.assert_allclose(rows[0]['y'].sum(1), 1)
        np.testing.assert_allclose(rows[0]['mass'], full[:, 1:].sum(1))
        np.testing.assert_allclose(rows[0]['y'] * rows[0]['mass'][:, None], full[:, 1:])

    def test_blank_and_nonblank_mass_survive_reconstruction(self):
        with np.errstate(divide='ignore'):
            primary = np.log([[.7, .1, .2], [1., 0., 0.], [0., .5, .5]])
        identity = np.log([[.8, .2]]*3)
        rows = reconstruct(identity, primary)
        np.testing.assert_array_equal(rows[:, 0], primary[:, 0])
        np.testing.assert_allclose(np.exp(rows[:, 1:]).sum(1), [0.3, 0., 1.], atol=1e-15)

    def test_speakers_cannot_cross_folds_or_enter_from_excluded_data(self):
        rows = [dict(speaker=s) for s in ['a', 'd', 'g']]
        folds = speaker_folds(rows, list('abcdefghi'))
        self.assertEqual({folds[r['speaker']] for r in rows}, {0, 1, 2})
        self.assertEqual(set(folds), set('abcdefghi'))
        with self.assertRaises(ValueError):
            speaker_folds(rows[:2], list('abcdefghi'))
        r = dict(id='a', source='help-train', split='train', excluded=False, speaker='a', kind='none',
                 features=[[1, 2]], logProbs=[[-1, -1, -1]])
        provenance([r], set(), 2, 2)
        with self.assertRaises(ValueError):
            provenance([r], {'a'}, 2, 2)
        with self.assertRaises(ValueError):
            provenance([dict(r, split='test')], set(), 2, 2)


if __name__ == '__main__':
    unittest.main()

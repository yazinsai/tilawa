"""Small exhaustive checks of soft occupancies and cross-recording mixup."""
import itertools
import unittest

import numpy as np
from student_vicinal_phone import occupancies, interpolation_pairs, sequence_score, speaker_folds


def collapse(path):
    return [p for i, p in enumerate(path) if p and (i == 0 or p != path[i-1])]


class VicinalTest(unittest.TestCase):
    def test_soft_occupancies_match_all_ctc_paths(self):
        probs = np.array([[.4, .5, .1], [.3, .2, .5], [.2, .4, .4], [.3, .3, .4]])
        ids = [1, 2]
        expected = np.zeros_like(probs)
        z = 0.
        for path in itertools.product(range(3), repeat=4):
            if collapse(path) != ids:
                continue
            mass = np.prod([probs[t, p] for t, p in enumerate(path)])
            z += mass
            for t, p in enumerate(path):
                expected[t, p] += mass
        targets, _ = occupancies(np.log(probs), ids)
        np.testing.assert_allclose(targets, expected/z, atol=1e-7)
        np.testing.assert_allclose(targets.sum(1), 1)
        self.assertTrue(np.any((targets > .01) & (targets < .99)))

    def test_genuine_repetition_requires_blank(self):
        self.assertIsNone(occupancies(np.log(np.full((2, 3), 1/3)), [1, 1]))
        targets, _ = occupancies(np.log(np.full((3, 3), 1/3)), [1, 1])
        np.testing.assert_array_equal(targets, [[0, 1, 0], [1, 0, 0], [0, 1, 0]])

    def test_donors_are_other_recordings_in_identical_context(self):
        rows = [dict(id='a', contexts=[(1, 2), (2, 3), (1, 2)]),
                dict(id='b', contexts=[(1, 2)]), dict(id='c', contexts=[(3, 4)])]
        a, b = interpolation_pairs(rows)
        contexts = [c for r in rows for c in r['contexts']]
        origins = [r['id'] for r in rows for _ in r['contexts']]
        self.assertEqual(len(a), 3)
        for i, j in zip(a, b):
            self.assertEqual(contexts[i], contexts[j])
            self.assertNotEqual(origins[i], origins[j])

    def test_neutral_identity_scores_condition_on_completion(self):
        lp = np.full((8, 3), -1.)
        self.assertAlmostEqual(sequence_score(lp, [1]), -8)
        self.assertAlmostEqual(sequence_score(lp, [1, 1, 2]), -8)
        self.assertEqual(sequence_score(lp[:1], [1, 1]), -np.inf)

    def test_preservation_only_speakers_cannot_empty_an_anchor_fold(self):
        rows = [dict(speaker=s) for s in ['a', 'd', 'g']]
        folds = speaker_folds(rows, list('abcdefghi'))
        self.assertEqual({folds[r['speaker']] for r in rows}, {0, 1, 2})
        self.assertEqual(set(folds), set('abcdefghi'))
        with self.assertRaises(ValueError):
            speaker_folds(rows[:2], list('abcdefghi'))


if __name__ == '__main__':
    unittest.main()

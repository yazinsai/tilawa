"""Small synthetic checks only; no recorded audio is fitted locally."""
import unittest
import numpy as np
from student_training_support import chain, comparison, fit, prepare, score


class TrainingSupportTests(unittest.TestCase):
    def test_complete_marginal(self):
        e = np.array([[-.2, -2], [-.7, -.3], [-2, -.4]])
        expected = np.logaddexp(e[0, 0]+e[1, 0]+e[2, 1], e[0, 0]+e[1, 1]+e[2, 1])-np.log(2)
        value, gamma = chain(e, True)
        self.assertAlmostEqual(value, expected)
        np.testing.assert_allclose(gamma.sum(1), 1)
        with self.assertRaises(ValueError):
            chain(e[:1])

    def test_whole_recording_deletion(self):
        # Removing one complete utterance matches an independent fresh refit.
        rows = [dict(id=str(i), speaker=str(i), x=np.full((4, 8), i*.2), ids=[1, 2]) for i in range(3)]
        full = fit(rows, 'بت', 2, epochs=2)
        deleted = [fit(rows[:i]+rows[i+1:], 'بت', 2, epochs=2) for i in range(3)]
        self.assertEqual(deleted[1], fit([rows[0], rows[2]], 'بت', 2, epochs=2))
        result = comparison(full, deleted, rows[1]['x'], [1], [2])
        effects = [(score(d, rows[1]['x'], [1])-score(d, rows[1]['x'], [2])-
                        score(full, rows[1]['x'], [1])+score(full, rows[1]['x'], [2]))/4 for d in deleted]
        expected = max(effects, key=abs)
        self.assertAlmostEqual(result['influence'], expected)

    def test_provenance(self):
        r = dict(source='help-train', split='train', excluded=False, speaker='s', id='r', kind='none',
                 pronunciation='بت', features=np.zeros((4, 8)).tolist())
        self.assertEqual(len(prepare([r], 'بت', set())), 1)
        for changes in [dict(split='exam'), dict(source='practice'), dict(kind='substitution'), dict(excluded=True)]:
            with self.assertRaises(ValueError):
                prepare([dict(r, **changes)], 'بت', set())
        with self.assertRaises(ValueError):
            prepare([r], 'بت', {'s'})
        with self.assertRaises(ValueError):
            prepare([r, r], 'بت', set())


if __name__ == '__main__':
    unittest.main()

"""Synthetic exact-CTC and recording-preserving grouping fixtures."""
from itertools import product
import unittest

import numpy as np

from student_confusion_batches import (batch_plan, batch_quality, ctc_occupancies,
    mine_confusion_pairs, recording_descriptors, verify_matched_plans)


def collapse(path):
    return [p for i, p in enumerate(path) if p != 0 and (i == 0 or p != path[i-1])]


class Occupancies(unittest.TestCase):
    def test_matches_enumerated_complete_ctc_paths_including_repeated_labels(self):
        rng = np.random.default_rng(17)
        p = rng.uniform(.05, 1., size=(5, 3))
        p /= p.sum(1, keepdims=True)
        for target in [[], [1], [1, 2], [1, 1], [2, 1, 2]]:
            expected = np.zeros_like(p)
            total = 0.
            for path in product(range(3), repeat=len(p)):
                if collapse(path) != target:
                    continue
                prob = np.prod(p[np.arange(len(p)), path])
                total += prob
                expected[np.arange(len(p)), path] += prob
            expected /= total
            np.testing.assert_allclose(ctc_occupancies(np.log(p), target), expected, atol=1e-12)

    def test_rejects_impossible_or_invalid_supervision(self):
        with self.assertRaisesRegex(ValueError, 'completely realized'):
            ctc_occupancies(np.log(np.full((2, 3), 1/3)), [1, 1])
        with self.assertRaises(ValueError):
            ctc_occupancies(np.log(np.full((2, 3), 1/3)), [0])

    def test_recording_descriptor_sums_soft_views_without_winning_label_substitution(self):
        lp = np.log(np.asarray([[.7, .1, .2], [.7, .2, .1], [.7, .1, .2]]))
        found = recording_descriptors({'a': [(lp, [1]), (lp, [1])]})['a']
        self.assertEqual(found[2], 0.)
        self.assertGreater(found[1], 0.)  # blank wins every frame
        self.assertAlmostEqual(float(found.sum()), 1.)

    def test_pair_mining_requires_bidirectional_training_decode_errors(self):
        first = np.log(np.asarray([[.05, .05, .9]]))
        reverse = np.log(np.asarray([[.05, .9, .05]]))
        self.assertEqual(mine_confusion_pairs([(first, [1])], [1, 2], 8), [])
        pairs = mine_confusion_pairs([(first, [1]), (reverse, [2])], [1, 2], 8)
        self.assertEqual(pairs[0]['phones'], [1, 2])
        self.assertEqual(pairs[0]['score'], 1.)


class Batches(unittest.TestCase):
    def fixture(self, count=32):
        keys = [str(i) for i in range(count)]
        lengths = {key: i+10 for i, key in enumerate(keys)}
        descriptors = {key: np.asarray([.5, .5, 0.] if i < count//2 else [.5, 0., .5])
                       for i, key in enumerate(keys)}
        return keys, lengths, descriptors

    def test_balancing_places_both_alternatives_together_without_resampling(self):
        keys, lengths, descriptors = self.fixture()
        schedules = {mode: batch_plan(keys, lengths, descriptors, [[1, 2]], 4, 42, mode)
                     for mode in ('balanced', 'shuffled', 'length')}
        verify_matched_plans(keys, schedules, 4)
        balanced = batch_quality(schedules['balanced'], descriptors, [[1, 2]])
        shuffled = batch_quality(schedules['shuffled'], descriptors, [[1, 2]])
        self.assertEqual(balanced['both_sides'], 1.)
        self.assertEqual(balanced['imbalance'], 0.)
        self.assertLess(balanced['imbalance'], shuffled['imbalance'])
        self.assertLess(balanced['imbalance'], batch_quality(schedules['length'], descriptors, [[1, 2]])['imbalance'])

    def test_each_seed_preserves_partial_batch_sizes_and_recording_multiplicity(self):
        keys, lengths, descriptors = self.fixture(19)
        for seed in range(12):
            plans = {mode: batch_plan(keys, lengths, descriptors, [[1, 2]], 4, seed, mode)
                     for mode in ('balanced', 'shuffled', 'length')}
            verify_matched_plans(keys, plans, 4)
            self.assertEqual([len(batch) for batch in plans['balanced']], [4, 4, 4, 4, 3])
            self.assertEqual(plans['balanced'], batch_plan(keys, lengths, descriptors, [[1, 2]], 4, seed, 'balanced'))
        with self.assertRaises(ValueError):
            verify_matched_plans(keys, {'duplicate': [[keys[0]]*4]}, 4)

    def test_soft_fractional_occupancies_drive_grouping(self):
        keys, lengths, descriptors = self.fixture()
        for v in descriptors.values():
            v[1:3] += .001
        shuffled = batch_plan(keys, lengths, descriptors, [[1, 2]], 4, 42, 'shuffled')
        balanced = batch_plan(keys, lengths, descriptors, [[1, 2]], 4, 42, 'balanced')
        self.assertLess(batch_quality(balanced, descriptors, [[1, 2]])['imbalance'],
                        batch_quality(shuffled, descriptors, [[1, 2]])['imbalance'])

    def test_refuses_absent_alternatives_and_duplicate_recordings(self):
        keys, lengths, descriptors = self.fixture()
        for v in descriptors.values():
            v[2] = 0.
        with self.assertRaisesRegex(ValueError, 'lacks an observed alternative'):
            batch_plan(keys, lengths, descriptors, [[1, 2]], 4, 42, 'balanced')
        with self.assertRaises(ValueError):
            batch_plan(keys+keys, lengths, descriptors, [], 4, 42, 'shuffled')


if __name__ == '__main__':
    unittest.main()

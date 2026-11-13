"""Checks of uncertain CTC boundaries and counterfactual feature composition."""
import importlib.util
import itertools
import unittest

import numpy as np
from student_crossed_context import occupancies, alignment, sample_path, compose, donors, temporal, sequence_score, speaker_folds, fit
from student_crossed_probe import exchange, disputed_pairs


def collapse(path):
    return [p for i, p in enumerate(path) if p and (i == 0 or p != path[i-1])]


class CrossedTest(unittest.TestCase):
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

    def test_uncertain_paths_are_complete_and_match_posterior_mass(self):
        probs = np.array([[.4, .5, .1], [.3, .2, .5], [.2, .4, .4], [.3, .3, .4]])
        found = alignment(np.log(probs), [1, 2])
        exact, _ = occupancies(np.log(probs), [1, 2])
        np.testing.assert_allclose(found['targets'], exact, atol=1e-7)
        rng = np.random.default_rng(1729)
        counts = np.zeros_like(probs)
        distinct = set()
        for _ in range(3000):
            path = sample_path(found, rng)
            labels = found['labels'][path]
            self.assertEqual(collapse(labels), [1, 2])
            distinct.add(tuple(path))
            for t, p in enumerate(labels):
                counts[t, p] += 1
        np.testing.assert_allclose(counts/3000, exact, atol=.04)
        self.assertGreater(len(distinct), 2)

    def test_crossed_join_preserves_core_and_changes_supervision_with_material(self):
        def item(record, phone, context, value):
            return dict(recording=record, phone=phone, context=context, parts=[
                (np.full((2, 32), value), np.array([context[0], 0])),
                (np.full((1, 32), value+1), np.array([phone])),
                (np.full((2, 32), value+2), np.array([0, context[1]]))])
        a, b, c = item('a', 2, (1, 3), 10), item('b', 2, (4, 5), 20), item('c', 6, (1, 3), 30)
        items = [a, b, c, item('a', 2, (4, 5), 40)]
        self.assertEqual(donors(items, a, 'context'), [1])
        self.assertEqual(donors(items, a, 'central'), [2])
        self.assertEqual(donors(items, a, 'unstructured'), [1, 2])
        x, y = compose(a, b, 'context')
        np.testing.assert_array_equal(x[2, 96:128], a['parts'][1][0][0])
        np.testing.assert_array_equal(y, [4, 0, 2, 0, 5])
        x, y = compose(a, c, 'central')
        np.testing.assert_array_equal(x[2, 96:128], c['parts'][1][0][0])
        np.testing.assert_array_equal(y, [1, 0, 6, 0, 3])
        self.assertNotEqual(list(y), [1, 0, 2, 0, 3])
        np.testing.assert_array_equal(a['parts'][1][0], np.full((1, 32), 11))

    def test_temporal_padding_does_not_wrap_around(self):
        x = np.array([[1., 2.], [3., 4.]], dtype='float32')
        values = temporal(x)
        np.testing.assert_array_equal(values[0], [0,0,0,0,0,0,1,2,3,4,0,0,0,0])

    def test_probe_keeps_the_disputed_interval_exact(self):
        features = np.arange(9*32, dtype='float32').reshape(9, 32)
        donor = dict(parts=[(np.ones((4,32)), None), (np.full((1,32), 99), None), (np.zeros((4,32)), None)])
        changed, n = exchange(features, 3, 5, donor)
        self.assertEqual(n, 6)
        np.testing.assert_array_equal(changed[3:5], features[3:5])
        self.assertFalse(np.array_equal(changed[:3], features[:3]))
        self.assertEqual(disputed_pairs([1,2,3], [1,4,3]), [(1,2,4)])
        self.assertEqual(disputed_pairs([1,2,3], [1,3]), [(1,2,0)])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch training smoke runs on Modal')
    def test_matched_training_builds_all_three_temporal_heads(self):
        rng = np.random.default_rng(4)
        rows = []
        for i, pronunciation in enumerate([[1,2,3], [1,4,3], [5,2,6]]):
            lp = np.log(np.full((12,7), 1/7))
            rows.append(dict(id=str(i), speaker=str(i), pronunciation=pronunciation,
                features=rng.normal(size=(12,32)), aligned=alignment(lp, pronunciation)))
        models, center, scale, count = fit(rows, 7, 'بتثجحخ', 1)
        self.assertEqual(len(models), 3)
        self.assertTrue(count['central'] > 0 and count['context'] > 0)
        self.assertEqual(list(models[0][0].weight.shape), [48,224])
        self.assertTrue(np.isfinite(center).all() and (scale > 0).all())

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

"""Numerical and training-boundary checks; run in the authorized train image."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from student_boosted_residuals import boost, ctc, log_probs, observations, predict_tree, speaker_folds, tree_json


class BoostedResidualTests(unittest.TestCase):
    def test_forward_backward_matches_paths_and_finite_difference(self):
        z = np.array([[.2, -.1, .4], [-.3, .8, .1], [.6, -.1, .3], [.4, .7, -.2]])
        for ids in ([1], [1, 2], [1, 1]):
            total = 0.
            occupancy = np.zeros_like(z)
            p = np.exp(log_probs(z))
            for path in itertools.product(range(3), repeat=len(z)):
                collapsed = [v for i, v in enumerate(path) if v and (i == 0 or v != path[i-1])]
                if collapsed == ids:
                    probability = np.prod([p[t, v] for t, v in enumerate(path)])
                    total += probability
                    for t, v in enumerate(path):
                        occupancy[t, v] += probability
            score, gradient = ctc(z, ids)
            self.assertAlmostEqual(score, np.log(total), places=12)
            np.testing.assert_allclose(gradient, occupancy/total-p, atol=1e-12)
            for t in range(len(z)):
                for v in range(3):
                    delta = np.zeros_like(z)
                    delta[t, v] = 1e-5
                    numeric = (ctc(z+delta, ids)[0]-ctc(z-delta, ids)[0])/2e-5
                    self.assertAlmostEqual(gradient[t, v], numeric, places=8)
        self.assertFalse(np.allclose(ctc(z, [1])[1], ctc(z+gradient, [1])[1]))

    def test_shared_vector_tree_export_and_threshold(self):
        from sklearn.tree import DecisionTreeRegressor
        x = np.array([[-2], [-1], [1], [2]], dtype=np.float32)
        y = np.array([[1, -1, 0], [1, -1, 0], [-2, 0, 2], [-2, 0, 2]])
        estimator = DecisionTreeRegressor(max_depth=1).fit(x, y)
        tree = tree_json(estimator)
        probes = np.r_[x, [[tree['threshold'][0]]]]
        np.testing.assert_allclose(predict_tree(tree, probes), estimator.predict(probes))
        self.assertEqual(len(tree['value'][1]), 3)

    def test_boosting_recomputes_current_transcript_gradient(self):
        x = np.zeros((24, 1), dtype=np.float32)
        z = np.tile([0., -.5, .8], (24, 1))
        row = dict(speaker='learner', role='transcript', x=x, z=z, ids=[1, 2])
        trees, _ = boost([row], [], depth=1, rate=.3, rounds=2)
        expected = ctc(z, row['ids'])[1].mean(axis=0)
        np.testing.assert_allclose(predict_tree(trees[0], x)[0], expected, atol=1e-12)
        current = z + .3*expected
        updated = ctc(current, row['ids'])[1].mean(axis=0)
        np.testing.assert_allclose(predict_tree(trees[1], x)[0], updated, atol=1e-12)
        self.assertFalse(np.allclose(expected, updated))

    def test_speaker_groups_and_mistake_supervision_rejection(self):
        rows = [dict(speaker=f's{i}', role=role) for i in range(6) for role in ('transcript', 'preservation')]
        folds = speaker_folds(rows)
        self.assertEqual(set.union(*folds), {r['speaker'] for r in rows})
        self.assertTrue(all(not folds[i] & folds[j] for i in range(3) for j in range(i)))
        meta = dict(tokens='ab', width=1, dimensions=3)
        invalid = dict(source='help-train', split='train', speaker='s0', role='transcript', kind='substituted_word',
            pronunciation='a', features=[[1]], logits=[[0, 0, 0]])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'rows.jsonl'
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, 'Mistake'):
                observations(path, meta, set())
            invalid.update(role='preservation', kind='substituted_word')
            path.write_text(json.dumps(invalid))
            self.assertEqual(observations(path, meta, set())[0]['role'], 'preservation')
            with self.assertRaisesRegex(ValueError, 'Unauthorized'):
                observations(path, meta, {'s0'})


if __name__ == '__main__':
    unittest.main()

"""Small synthetic checks for complete scoring and causal channel effects."""
import itertools
import unittest

import numpy as np
from student_feature_lesion import ctc_scores, lesioned_logits, log_softmax, partition_summary


class FeatureLesionTest(unittest.TestCase):
    def test_complete_ctc_matches_enumeration_in_every_arm(self):
        probabilities = np.array([[.2, .5, .3], [.4, .2, .4], [.3, .6, .1]])
        for sequence in [[], [1], [1, 1], [1, 2], [1, 2, 1]]:
            mass = 0.
            for path in itertools.product(range(3), repeat=3):
                collapsed = [p for t, p in enumerate(path) if p and (not t or p != path[t-1])]
                if collapsed == sequence:
                    mass += np.prod([probabilities[t, p] for t, p in enumerate(path)])
            result = ctc_scores(np.stack([np.log(probabilities)] * 2), sequence)
            np.testing.assert_allclose(np.exp(result), [mass, mass], atol=1e-12)
        self.assertTrue(np.isneginf(ctc_scores(np.log(probabilities[:1]), [1, 1])[0]))

    def test_mean_lesion_is_exactly_the_unchanged_head_on_modified_features(self):
        rng = np.random.default_rng(1729)
        features, weight, bias, mean = rng.normal(size=(7, 4)), rng.normal(size=(4, 3)), rng.normal(size=3), rng.normal(size=4)
        logits = features @ weight + bias
        masks = [[], [1], [1, 3]]
        result = lesioned_logits(features, logits, weight, mean, masks)
        for arm, mask in enumerate(masks):
            modified = features.copy()
            modified[:, mask] = mean[mask]
            np.testing.assert_allclose(result[arm], modified @ weight + bias, atol=1e-12)
        np.testing.assert_allclose(np.exp(log_softmax(result)).sum(-1), 1.)
        np.testing.assert_array_equal(result[0], logits)

    def test_preservation_counts_both_correct_and_scripted_distinctions(self):
        comparisons = [{'role': 'false', 'originalMargin': -2},
                       {'role': 'correct', 'originalMargin': 2},
                       {'role': 'scripted', 'originalMargin': -3}]
        report = partition_summary(comparisons, np.array([[1, -1], [2, -2], [-1, 1]]))
        self.assertEqual(report, {'reversals': [1, 0], 'preservationLosses': [0, 2]})


if __name__ == '__main__':
    unittest.main()

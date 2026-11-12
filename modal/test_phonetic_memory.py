"""Run with `python -m unittest discover -s modal -p 'test_phonetic_memory.py'`."""

import itertools
import unittest

import numpy as np

from phonetic_memory import balanced_indices, covariance_losses, occurrence_occupancy, pool_occurrences
from student_decorrelation import Pronunciations, partition

try:
    import torch
except ImportError:
    torch = None


def enumerate_occupancy(probabilities, target):
    mass = 0.0
    occupancy = np.zeros((len(probabilities), len(target)))
    for path in itertools.product(range(probabilities.shape[1]), repeat=len(probabilities)):
        collapsed, owners, previous = [], [], 0
        for label in path:
            if label and label != previous:
                collapsed.append(label)
            owners.append(len(collapsed) - 1 if label else -1)
            previous = label
        if collapsed != target:
            continue
        p = np.prod([probabilities[t, label] for t, label in enumerate(path)])
        mass += p
        for t, owner in enumerate(owners):
            if owner >= 0:
                occupancy[t, owner] += p
    return occupancy / mass


class AlignmentTests(unittest.TestCase):
    def test_canonical_word_boundaries_separate_duration_from_rearticulation(self):
        reference = Pronunciations({"ayahs": [{"s": 1, "a": 1, "p": "aabbba", "o": [0, 3]}]})
        self.assertEqual(reference.span("aabbba"), "abba")
        self.assertEqual(reference.passage(["1:1"]), "abba")
        self.assertIsNone(reference.span("untranscribed"))

    def test_ambiguous_occurrence_layout_is_excluded(self):
        reference = Pronunciations({"ayahs": [
            {"s": 1, "a": 1, "p": "aabb", "o": [0, 2]},
            {"s": 1, "a": 2, "p": "aabb", "o": [0, 1]},
        ]})
        self.assertIsNone(reference.span("aabb"))

    def test_partition_depends_only_on_speaker(self):
        groups = [partition(f"speaker-{i}") for i in range(100)]
        self.assertEqual(set(groups), {"fit", "development", "confirmation"})
        self.assertEqual(partition("same-speaker"), partition("same-speaker"))

    def test_forward_backward_matches_all_paths_including_repeated_labels(self):
        p = np.random.default_rng(3).dirichlet([1, 1, 1], size=5)
        for target in ([1], [1, 2], [1, 1], [1, 2, 1]):
            actual = occurrence_occupancy(np.log(p), target)
            np.testing.assert_allclose(actual, enumerate_occupancy(p, list(target)), atol=1e-6)

    def test_impossible_alignment_is_not_fabricated(self):
        self.assertIsNone(occurrence_occupancy(np.log(np.full((2, 3), 1 / 3)), [1, 1]))
        self.assertIsNone(occurrence_occupancy(np.array([[0, -np.inf]]), [1]))

    def test_blank_is_rejected_and_empty_transcript_has_no_occurrences(self):
        with self.assertRaises(ValueError):
            occurrence_occupancy(np.zeros((2, 3)), [0])
        self.assertEqual(occurrence_occupancy(np.zeros((2, 3)), []).shape, (2, 0))

    def test_balanced_sampling_is_reproducible_and_caps_frequent_classes(self):
        labels = [1] * 100 + [2] * 4 + [3] * 2
        a = balanced_indices(labels, 4, 8)
        np.testing.assert_array_equal(a, balanced_indices(labels, 4, 8))
        self.assertEqual([sum(np.array(labels)[a] == c) for c in (1, 2, 3)], [4, 4, 2])
        with self.assertRaises(ValueError):
            balanced_indices([0, 1], 4, 8)


@unittest.skipIf(torch is None, "Feature-gradient tests require the training image's PyTorch")
class FeatureTests(unittest.TestCase):
    def test_duration_merge_is_explicit_and_genuine_repeats_stay_separate(self):
        x = torch.tensor([[1., 0.], [2., 0.], [8., 0.]], requires_grad=True)
        occupancy = np.eye(3, dtype=np.float32)
        e, y = pool_occurrences(x, occupancy, [1, 1, 1])
        self.assertEqual(len(e), 3)
        merged, labels = pool_occurrences(x, occupancy, [1, 1, 1], [0, 0, 1])
        torch.testing.assert_close(merged[:, 0], torch.tensor([1.5, 8.]))
        self.assertEqual(labels.tolist(), [1, 1])
        merged.sum().backward()
        torch.testing.assert_close(x.grad[:, 0], torch.tensor([.5, .5, 1.]))
        with self.assertRaises(ValueError):
            pool_occurrences(x, occupancy, [1, 2, 1], [0, 0, 1])

    def test_pooling_excludes_blank_and_requires_real_frame_slice(self):
        with self.assertRaises(ValueError):
            pool_occurrences(torch.zeros(4, 2), np.ones((3, 1)), [1])
        with self.assertRaises(ValueError):
            pool_occurrences(torch.zeros(3, 2), np.ones((3, 1)), [0])

    def test_class_weights_ignore_duplicates_and_covariance_has_feature_gradients(self):
        x = torch.tensor([[-1., -1.], [1., 1.], [-2., -2.], [2., 2.]], requires_grad=True)
        labels = torch.tensor([1, 1, 2, 2])
        scale = torch.ones(2)
        var, cov = covariance_losses(x, labels, scale)
        duplicate = torch.cat([x[:2].repeat(5, 1), x[2:]])
        y = torch.tensor([1] * 10 + [2, 2])
        v2, c2 = covariance_losses(duplicate, y, scale)
        torch.testing.assert_close(var, v2)
        torch.testing.assert_close(cov, c2)
        cov.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_variance_floor_penalizes_collapsed_dimensions(self):
        x = torch.tensor([[-.01, 0.], [.01, 0.]], requires_grad=True)
        var, cov = covariance_losses(x, torch.tensor([1, 2]), torch.ones(2))
        self.assertGreater(var.item(), .98)
        self.assertEqual(cov.item(), 0)
        var.backward()
        self.assertLess(x.grad[1, 0].item(), 0)

    def test_matched_control_has_identical_loss_except_covariance(self):
        x = torch.tensor([[-1., -1.], [1., 1.]], requires_grad=True)
        variance, covariance = covariance_losses(x, torch.tensor([1, 2]), torch.ones(2))
        ordinary = x.square().mean()
        control = ordinary + .1 * variance + 0 * covariance
        treatment = ordinary + .1 * variance + .1 * covariance
        torch.testing.assert_close(treatment - control, .1 * covariance)
        gc = torch.autograd.grad(control, x, retain_graph=True)[0]
        gt = torch.autograd.grad(treatment, x, retain_graph=True)[0]
        expected = torch.autograd.grad(.1 * covariance, x)[0]
        torch.testing.assert_close(gt - gc, expected)


if __name__ == "__main__":
    unittest.main()

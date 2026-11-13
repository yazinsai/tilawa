import unittest
import numpy as np
from student_segment_units import acoustic_segments, correspondence, descriptor, fit_transducer, forced_segments


class SegmentUnitsTest(unittest.TestCase):
    def test_normalized_trajectory(self):
        np.testing.assert_allclose(descriptor(np.array([[0.], [6.], [12.], [99.]]), 0, 3), [2, 6, 10])
        np.testing.assert_allclose(descriptor(np.array([[7.]]), 0, 1), [7, 7, 7])

    def test_boundary_learning_needs_only_audio(self):
        x = np.array([[0.]]*4+[[5.]]*4)
        boundaries = acoustic_segments(x, np.array([[0., 0., 0.], [5., 5., 5.]]), .1)
        self.assertIn((0, 4), boundaries)
        self.assertIn((4, 8), boundaries)

    def test_complete_correspondence_is_marginalized(self):
        p = correspondence(np.zeros((5, 3)), [1, 2])
        np.testing.assert_allclose(p.sum(1), 1)
        self.assertGreater(p[1, 1], 0)
        self.assertGreater(p[1, 2], 0)
        self.assertIsNone(correspondence(np.zeros((2, 3)), [1, 1]))

    def test_forced_control_keeps_phone_boundaries(self):
        log_probs = np.full((6, 3), -20.)
        log_probs[:3, 1] = 0; log_probs[3:, 2] = 0
        self.assertEqual(forced_segments(log_probs, [1, 2]), [(0, 3), (3, 6)])

    def test_one_acoustic_unit_can_train_multiple_phones(self):
        log_probs = np.full((6, 3), -20.)
        log_probs[:3, 1] = 0; log_probs[3:, 2] = 0
        row = dict(canonical=True, pronunciation='ab', log_probs=log_probs)
        arm = dict(centers=[[0., 0., 0.]]+[[10., 10., 10.]]*31, variance=.1)
        learned = fit_transducer([row], [np.zeros((6, 1))], arm, 'independent', 'ab', 1.)
        self.assertGreater(learned['phones'][1][0], .5)
        self.assertGreater(learned['phones'][2][0], .5)
        self.assertGreater(learned['counts'][1], 0)
        self.assertGreater(learned['counts'][2], 0)


if __name__ == '__main__': unittest.main()

"""Mechanism checks use tiny synthetic arrays; audio fitting runs only on Modal."""
import unittest
import numpy as np
from student_alignment_noise import chain, latent_weights, fit_heads, prepare


class AlignmentNoiseTest(unittest.TestCase):
    def test_complete_boundary_sum(self):
        # Three placements of the boundary, all equally likely under zero emissions.
        self.assertAlmostEqual(chain(np.zeros((4, 2))), 0)
        value, gamma = chain(np.zeros((4, 2)), True)
        np.testing.assert_allclose(gamma.sum(1), 1)
        self.assertEqual(gamma[0, 0], 1)
        self.assertEqual(gamma[-1, -1], 1)

    def test_marginalizes_assignment_rather_than_dropping_observation(self):
        nominal = np.array([[0., 1., 0.]])
        neighbour = np.array([[0., 0., 1.]])
        result = latent_weights(np.log([[.1, .1, .9]]), nominal, neighbour, np.array([0., .5, 0.]))
        np.testing.assert_allclose(result, [[0., .1, .9]], atol=1e-12)
        np.testing.assert_allclose(result.sum(1), 1)

    def test_without_disagreement_noise_and_control_are_identical(self):
        x = np.array([[0.]*8, [1.]*8, [3.]*8, [4.]*8])
        nominal = np.eye(3)[[1, 1, 2, 2]]
        rows = [dict(x=x, nominal=nominal, neighbours=nominal[:, [0, 2, 1]])]
        control, noise = fit_heads(rows, 3, np.zeros(3), 2)
        for key in ('mean','variance','count'):
            np.testing.assert_allclose(control[key], noise[key])

    def test_boundary_noise_corrects_a_misassigned_adjacent_phone(self):
        x = np.array([[0.]*8]*8+[[4.]*8]*9)
        nominal = np.eye(3)[[1]*9+[2]*8]
        neighbour = np.eye(3)[[2]*9+[1]*8]
        control, noise = fit_heads([dict(x=x, nominal=nominal, neighbours=neighbour)],
            3, np.array([0., .2, .2]), 2)
        self.assertLess(abs(noise['mean'][1, 0]), abs(control['mean'][1, 0]))
        self.assertAlmostEqual(noise['count'].sum(), len(x))

    def test_excludes_nontraining_and_scripted_recordings(self):
        row = dict(source='help-train',kind='none',split='train',excluded=False,speaker='a',
            pronunciation='a',features=[[0.]*8])
        for patch in [dict(kind='wrong_word'),dict(split='eval'),dict(excluded=True),dict(source='practice')]:
            with self.assertRaises(ValueError):
                prepare([{**row, **patch}], 'a', set())
        with self.assertRaises(ValueError):
            prepare([row], 'a', {'a'})


if __name__ == '__main__':
    unittest.main()

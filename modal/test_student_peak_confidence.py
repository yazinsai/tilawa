"""Small exhaustive checks of the training soft labels; run on the Modal CPU."""
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from student_peak_confidence import ctc_graph, labelled_runs, occurrence_label


def collapsed(path):
    return [phone for i, phone in enumerate(path) if phone and (i == 0 or phone != path[i - 1])]


class TranscriptMarginalTest(unittest.TestCase):
    def test_matches_all_ctc_paths_including_repeated_occurrences(self):
        p = np.array([[.2, .6, .2], [.5, .3, .2], [.2, .4, .4], [.3, .1, .6]])
        lp = np.log(p)
        for transcripts in ([[1, 2]], [[1, 1]], [[1], [1, 2]], [[]]):
            graphs = [ctc_graph(lp, labels) for labels in transcripts]
            mass = 0
            occupied = 0
            for path in itertools.product(range(3), repeat=len(p)):
                multiplicity = sum(collapsed(path) == labels for labels in transcripts)
                weight = multiplicity * np.prod([p[t, phone] for t, phone in enumerate(path)])
                mass += weight
                if 1 in path[1:3]:
                    occupied += weight
            self.assertAlmostEqual(sum(np.exp(g[-1]) for g in graphs), mass, places=12)
            self.assertAlmostEqual(occurrence_label(graphs, lp, 1, 3, 1), occupied / mass, places=12)

    def test_absent_class_has_zero_correctness(self):
        lp = np.log(np.full((4, 3), 1 / 3))
        graph = ctc_graph(lp, [2])
        self.assertAlmostEqual(occurrence_label([graph], lp, 0, 4, 1), 0, places=12)

    def test_invalid_occurrences_and_empty_training_cache_are_rejected(self):
        lp = np.log(np.full((4, 3), 1 / 3))
        graph = ctc_graph(lp, [1])
        for a, b, phone in ((2, 2, 1), (-1, 2, 1), (0, 5, 1), (0, 2, 0)):
            with self.assertRaises(ValueError):
                occurrence_label([graph], lp, a, b, phone)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no emitted"):
                labelled_runs(Path(directory))


if __name__ == "__main__":
    unittest.main()

"""Lightweight estimator checks, run in the training environment before fitting real recordings."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("transition_train", Path(__file__).parents[1] / "src/acoustic_transition_train.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class TransitionTrainingTest(unittest.TestCase):
    def test_rejects_speaker_crossing_partitions(self):
        document = {"tokens": "ab", "examples": [
            {"speaker": "one", "fold": f, "choices": ["a", "b"], "rows": [[0.1, 0.9]]} for f in [0, 1]
        ]}
        with self.assertRaisesRegex(ValueError, "Speaker crosses"):
            training.tensor_examples(document, "rows")

    def test_latent_boundaries_are_contiguous_and_transcript_constrained(self):
        phones = [{"p": p, "start": i, "end": i, "conf": 0.9} for i, p in enumerate("abca")]
        document = {"clips": [{"speaker": "one", "fold": 0, "written": ["ab", "ca"], "starts": [0, 2],
                               "evidence": {"phonemes": phones}}]}
        prepared = training.boundaries(document)
        self.assertEqual({e["boundarySample"] for e in prepared["examples"]}, {0, 1, 2, 3})
        for e in prepared["examples"]:
            self.assertIn(e["written"], "abca")
            frames = [p["start"] for p in e["phones"]]
            self.assertEqual(frames, list(range(frames[0], frames[-1] + 1)))


if __name__ == "__main__":
    unittest.main()

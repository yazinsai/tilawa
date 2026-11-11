"""Small estimator invariants; run with Python on Modal where Torch is available."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("acoustic_run_train", Path(__file__).parents[1] / "src/acoustic_run_train.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CrossEncoderTests(unittest.TestCase):
    def test_speaker_cannot_cross_folds(self):
        examples = [{"speaker": "same", "fold": fold, "choices": ["a", "b"], "rows": [[0.6, 0.3, 0.1]]} for fold in [0, 1]]
        with self.assertRaisesRegex(ValueError, "Speaker crosses"):
            module.tensor_examples({"tokens": "ab", "examples": examples})

    def test_complete_distributions_reach_the_scorer(self):
        model = module.CrossEncoder(3, 8)
        rows = module.torch.tensor([[0.6, 0.3, 0.1], [0.6, 0.1, 0.3]], requires_grad=True)
        choices = [module.torch.tensor([1, 2]), module.torch.tensor([2, 1])]
        output = model(rows, choices)
        self.assertEqual(tuple(output.shape), (2,))
        output[0].backward()
        self.assertTrue(module.torch.isfinite(rows.grad).all())
        self.assertGreater(float(rows.grad[:, 1:].abs().sum()), 0)

    def test_reference_shapes_match_javascript(self):
        exported = module.CrossEncoder(3, 8).export()
        self.assertEqual(len(exported["acoustic"]), 8)
        self.assertEqual(len(exported["acoustic"][0]), 3)
        self.assertEqual(len(exported["acoustic"][0][0]), 3)
        self.assertEqual(len(exported["joint"][0]), 24)


if __name__ == "__main__":
    unittest.main()

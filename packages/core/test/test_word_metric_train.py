"""Small training-numerics checks; heavy fitting remains on Modal."""
import importlib.util
import random
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("word_metric_train", Path(__file__).parent.parent / "src/word_metric_train.py")
metric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric)


class WordMetricTrainingTests(unittest.TestCase):
    def test_duration_charges_are_separate_from_identity(self):
        self.assertEqual(metric.delete("نننن", 1), 0.2)
        self.assertEqual(metric.delete("ن", 0), 1)
        self.assertEqual(metric.identity_edit("نننن", "ن"), 0)
        self.assertEqual(metric.identity_edit("ق", "ك"), 5)

    def test_latent_boundaries_are_reproducible(self):
        one = metric.latent_words(["بتد", "بتد"], "بتدبتد", random.Random(42))
        two = metric.latent_words(["بتد", "بتد"], "بتدبتد", random.Random(42))
        self.assertEqual(one, two)
        self.assertTrue(one)
        self.assertTrue(all(r and o for r, o in one))
        self.assertTrue(all(set(r + o) <= set("بتد") for r, o in one))

    def test_batch_padding_does_not_change_string_composition(self):
        metric.torch.manual_seed(17)
        inventory = {p: i for i, p in enumerate("بتد")}
        encoder = metric.Encoder(len(inventory), 4).eval()
        solo = encoder(*metric.batch(["ب"], inventory))[0]
        padded = encoder(*metric.batch(["ب", "بتدبتد"], inventory))[0]
        self.assertTrue(metric.torch.allclose(solo, padded, atol=1e-6))

    def test_control_is_zero_even_with_different_encoders(self):
        inventory = {p: i for i, p in enumerate("بتد")}
        model = metric.Dual(len(inventory), 4)
        with metric.torch.no_grad():
            model.observed.bias.add_(1)
        cost = metric.distance(model, [("بتد", "بتد")], inventory)
        self.assertEqual(cost.item(), 0)


if __name__ == "__main__":
    unittest.main()

import json
import importlib.util
import os
import random
import time
import unittest
from pathlib import Path

import numpy as np

from location_margin import FEATURES, Reference, corrupt, corruption_channel, edits, inner_search

spec = importlib.util.spec_from_file_location("location_fit", Path(__file__).with_name("location-fit.py"))
fitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fitter)


def fixture():
    phones = "ابتثجح" * 8
    words = [{"surah": 1, "ayah": i // 2 + 1, "word": i % 2 + 1, "start": i * 3, "end": i * 3 + 3}
             for i in range(16)]
    return Reference({"phonemes": phones, "words": words})


class LocationMarginTest(unittest.TestCase):
    def test_source_supported_corruption_only(self):
        rng = random.Random(3)
        self.assertEqual(corrupt("ابت", ("sub", "ب", "ث"), rng), "اثت")
        self.assertEqual(corrupt("ابت", ("delete", "ب", ""), rng), "ات")
        self.assertEqual(corrupt("ابت", ("insert", "ب", "ج"), rng), "اجبت")
        self.assertIsNone(corrupt("ابت", ("sub", "ح", "ث"), rng))
        self.assertEqual(edits("ابت", "اثت"), [("sub", "ب", "ث")])

    def test_channel_is_training_only_and_excludes_duration_repetitions(self):
        self.assertNotIn(("delete", "ن", ""), edits("ننننَ", "نَ"))
        reference = fixture()
        clean = [{"path": [0], "phones": "اثت", "split": "train"}] * 3
        channel = corruption_channel(clean, reference)
        self.assertEqual(channel["patterns"], [(("sub", "ب", "ث"), 3)])
        with self.assertRaises(ValueError):
            corruption_channel([{**clean[0], "split": "heldout"}], reference)

    def test_inner_search_worst_vs_random_with_equal_compute(self):
        class Margin:
            def margin(self, example, weights, phones):
                return float(phones.count("ث")), np.zeros(len(FEATURES))
        channel = {"patterns": [(("sub", "ب", "ث"), 1)], "rate": .1, "max_edits": 2}
        example = {"phones": "ابت"}
        adv, ca = inner_search(Margin(), example, np.zeros(len(FEATURES)), channel, 17, True)
        control, cr = inner_search(Margin(), example, np.zeros(len(FEATURES)), channel, 17, False)
        self.assertEqual(ca, cr)
        self.assertEqual(ca, 9)
        self.assertGreaterEqual(adv[0], control[0])

    def test_true_location_and_restart_margin_gradient(self):
        reference = fixture()
        example = {"phones": "ابتثجح", "start": 0, "target": [4, 7]}
        weights = np.linspace(-.013, .017, len(FEATURES))
        result = reference.margin(example, weights)
        self.assertIsNotNone(result)
        gap, grad = result
        direction = np.linspace(.7, -.2, len(FEATURES))
        plus = reference.margin(example, weights + 1e-6 * direction)[0]
        minus = reference.margin(example, weights - 1e-6 * direction)[0]
        self.assertAlmostEqual((plus - minus) / 2e-6, float(grad @ direction), places=5)
        self.assertTrue(np.isfinite(gap))

    def test_optimizer_is_bounded_and_compute_matched(self):
        reference = fixture()
        example = {"phones": "ابتثجح", "start": 0, "target": [4, 7], "source": "synthetic", "kind": "clean"}
        channel = {"patterns": [(("sub", "ب", "ث"), 2)], "rate": .1, "max_edits": 2}
        a, ac = fitter.fit(reference, [example], channel, .1, True, 3, time.monotonic() + 60)
        b, bc = fitter.fit(reference, [example], channel, .1, False, 3, time.monotonic() + 60)
        self.assertEqual(ac["evaluations"], bc["evaluations"])
        self.assertEqual(ac["updates"], 3)
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertTrue(np.all(np.isfinite(b)))
        self.assertLessEqual(np.linalg.norm(a), 1 + 1e-9)
        with self.assertRaises(TimeoutError):
            fitter.fit(reference, [example], channel, .1, True, 3, time.monotonic() - 1)

    def test_sampling_is_deterministic_and_balanced_without_model_feedback(self):
        rows = [{"speaker": str(s), "kind": k, "source": str(i), "phones": "ابت", "start": 0, "target": [1, 4]}
                for s in range(3) for k in ["clean", "jump", "passage-jump"] for i in range(4)]
        a, b = fitter.selected_examples(rows), fitter.selected_examples(list(reversed(rows)))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 18)
        self.assertEqual({r["speaker"] for r in a}, {"0", "1", "2"})

    def test_parity_with_actual_typescript_search(self):
        path = os.environ.get("LOCATION_PARITY")
        if not path:
            self.skipTest("Run location-parity.ts and set LOCATION_PARITY for the cross-language check")
        data = json.loads(Path(path).read_text())
        reference = Reference(data["reference"])
        for case in data["cases"]:
            for n, expected in enumerate(case["states"], 1):
                actual = reference.replay(case["phones"][:n], 0, np.array(case["weights"]))
                self.assertEqual(actual["lo"], expected["lo"])
                self.assertEqual(actual["best"], expected["best"])
                for i, c in enumerate(expected["cost"]):
                    if c is None:
                        self.assertFalse(np.isfinite(actual["cost"][i]))
                    else:
                        self.assertAlmostEqual(actual["cost"][i], c, places=8)
                # Equal-cost histories can have different witnesses after floating-point accumulation.
                chosen = actual["best"] - actual["lo"]
                self.assertEqual(int(actual["origin"][chosen]), expected["origin"][chosen])


if __name__ == "__main__":
    unittest.main()

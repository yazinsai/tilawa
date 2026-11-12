"""Training calibration must use exactly the inference marginal, including its scale."""
import importlib.util
import math
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("edit_order_fit", Path(__file__).parents[1] / "src/edit-order-fit.py")
estimator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimator)


class CalibrationTest(unittest.TestCase):
    def test_scale_precedes_ambiguous_script_marginal(self):
        features = torch.zeros((1, 2, 2, 25), dtype=torch.float64)
        features[0, 0, 0, 0] = 2
        features[0, 0, 1, 0] = -2
        acoustic = torch.full((1, 2, 2), -math.log(2), dtype=torch.float64)
        weights = torch.zeros(25, dtype=torch.float64)
        weights[0] = 1
        tensors = (features, acoustic, torch.ones(1, dtype=torch.float64))
        self.assertAlmostEqual(estimator.differences(weights, tensors, 0.5).item(), math.log(math.cosh(1)))
        self.assertNotAlmostEqual(estimator.differences(weights, tensors, 0.5).item(),
                                  0.5 * estimator.differences(weights, tensors).item())
        self.assertAlmostEqual(estimator.differences(weights, tensors, 0).item(), 0)

    def test_shuffled_matched_pairs_have_zero_discriminative_gradient(self):
        features = [torch.randn((2, 25), dtype=torch.float64), torch.randn((3, 25), dtype=torch.float64)]
        acoustic = [torch.log_softmax(torch.randn(2, dtype=torch.float64), 0),
                    torch.log_softmax(torch.randn(3, dtype=torch.float64), 0)]
        example = ("training-speaker", features, acoustic, torch.randn(25, dtype=torch.float64))
        parameters = estimator.fit([example], 0.01, True)
        self.assertEqual(parameters.tolist(), [0.0] * 25)
        self.assertEqual(estimator.differences(parameters, estimator.batch([example], True)).item(), 0)


if __name__ == "__main__":
    unittest.main()

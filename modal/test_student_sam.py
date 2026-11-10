"""Run with python -m unittest discover -s modal -p test_student_sam.py."""

import unittest
import random

import torch
import numpy as np

from student_sam import sharpness_step


class SharpnessStepTest(unittest.TestCase):
    def test_relative_radius_and_adverse_gradient_update(self):
        model = torch.nn.Linear(2, 1, bias=False).double()
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1., 3.]]))
        original = model.weight.detach().clone()
        seen = []

        def closure():
            seen.append(model.weight.detach().clone())
            return {"loss": model.weight.square().sum() / 2}

        rho, lr = .05, .1
        result = sharpness_step(model, torch.optim.SGD(model.parameters(), lr=lr), closure, rho)
        scale = original.square().mean().sqrt()
        self.assertAlmostEqual(float(((seen[1] - original) / scale).square().mean().sqrt()), rho)
        self.assertGreater(float(result["perturbed_loss"]), float(result["loss"]))
        torch.testing.assert_close(model.weight, original - lr * seen[1])

    def test_zero_radius_is_exact_ordinary_update(self):
        model = torch.nn.Linear(2, 1)
        other = torch.nn.Linear(2, 1)
        other.load_state_dict(model.state_dict())
        x = torch.tensor([[2., -1.]])
        sharpness_step(model, torch.optim.Adam(model.parameters(), lr=.01),
                       lambda: {"loss": model(x).square().sum()}, 0)
        opt = torch.optim.Adam(other.parameters(), lr=.01)
        other(x).square().sum().backward()
        opt.step()
        for a, b in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_same_dropout_and_one_buffer_update(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Dropout(.5), torch.nn.Linear(4, 1))
        x = torch.randn(8, 4)
        draws = []

        def closure():
            draws.append((torch.rand(6), random.random(), np.random.random()))
            return {"loss": model(x).square().mean()}

        sharpness_step(model, torch.optim.SGD(model.parameters(), lr=.01), closure, .05)
        torch.testing.assert_close(draws[0][0], draws[1][0], rtol=0, atol=0)
        self.assertEqual(draws[0][1:], draws[1][1:])
        self.assertEqual(int(model[0].num_batches_tracked), 1)

    def test_failed_second_pass_restores_weights_and_optimizer(self):
        model = torch.nn.Linear(1, 1)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        opt = torch.optim.Adam(model.parameters())
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            loss = model(torch.ones(1, 1)).square().sum()
            return {"loss": loss if calls == 1 else loss * float("nan")}

        with self.assertRaises(FloatingPointError):
            sharpness_step(model, opt, closure, .05)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, original[k], rtol=0, atol=0)
        self.assertFalse(opt.state)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_invalid_radius_rejected(self):
        model = torch.nn.Linear(1, 1)
        for rho in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                sharpness_step(model, torch.optim.SGD(model.parameters(), lr=.1), lambda: None, rho)

    def test_backtracks_when_nonlinearity_lowers_actual_loss(self):
        model = torch.nn.Linear(1, 1, bias=False).double()
        with torch.no_grad():
            model.weight.fill_(1.)
        result = sharpness_step(model, torch.optim.SGD(model.parameters(), lr=.1),
                               lambda: {"loss": -model.weight.cos().sum()}, 5.)
        self.assertEqual(float(result["effective_rho"]), 2.5)
        self.assertGreater(float(result["perturbed_loss"]), float(result["loss"]))
        self.assertAlmostEqual(float(model.weight.detach()), 1. - .1 * np.sin(3.5))


if __name__ == "__main__":
    unittest.main()

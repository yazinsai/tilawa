import unittest

import torch

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

spec = spec_from_file_location('belief_fit', Path(__file__).with_name('belief-fit.py'))
fit = module_from_spec(spec)
spec.loader.exec_module(fit)


def frames():
    return [dict(x=torch.full((3, fit.FEATURES), x), predecessor=torch.tensor([-1, 0, 1]),
                 valid=torch.tensor([True, True, True]), costs=torch.tensor([1., 0., 1.]),
                 target=torch.tensor([.1, .8, .1])) for x in (.2, .4, .6)]


class BeliefFitTest(unittest.TestCase):
    def test_equal_capacity_and_finite_gradients(self):
        rnn, control = fit.Student(), fit.Student(True)
        self.assertEqual(sum(p.numel() for p in rnn.parameters()), sum(p.numel() for p in control.parameters()))
        for model in (rnn, control):
            objective = fit.loss(frames(), model(frames()))
            objective.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_prefix_causality_and_reset_ablation(self):
        model, f = fit.Student(), frames()
        self.assertTrue(torch.equal(model(f[:1])[0], model(f)[0]))
        self.assertFalse(torch.equal(model(f)[1], model(f, reset=True)[1]))

    def test_control_discards_prior_state(self):
        model, f = fit.Student(True), frames()
        self.assertTrue(torch.equal(model(f)[1], model(f, reset=True)[1]))

    def test_export_matches_runtime_equations(self):
        model = fit.Student()
        m = model.export(.5)
        x = frames()[0]['x'][0].tolist()
        import math
        hidden = [math.tanh(sum(w * v for w, v in zip(row, x)) + bias)
                  for row, bias in zip(m['input'], m['bias'])]
        expected = sum(w * math.tanh(h + b) for w, h, b in zip(m['output'], hidden, m['outputBias']))
        self.assertAlmostEqual(float(model(frames())[0][0]), expected, places=6)


if __name__ == '__main__':
    unittest.main()

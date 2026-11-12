"""Checks the differentiable graph against exhaustive tiny path enumeration and finite differences."""
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('fit', Path(__file__).with_name('reference-edit-fit.py'))
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


def test_graph_partition_and_gradient():
    # One observed token, two reference cells: insertion, match, lexical jump, and reference-only deletion.
    g = fit.prepare(dict(ins=[[1., 1.]], sub=[[0., 0.]], fixed=[[0., 0.]], source=[0], jump=[[-1, 3]],
                         initial=0, entry=[0., 0.], owners=[0, 1], allowed=[True, True], **{'del': [0., 1.]}), 2)
    theta = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    z, _ = fit.forward(g, theta)
    # Insertion at cell 0, then optionally deletion; match from cell 0; jump from cell 0.
    expected = torch.logsumexp(torch.tensor([-1., -2., 0., -2.], dtype=torch.float64), dim=0)
    assert torch.allclose(z, expected)
    z.backward()
    for feature in range(6):
        shift = torch.zeros(6, dtype=torch.float64)
        shift[feature] = 1e-5
        difference = (fit.forward(g, shift)[0] - fit.forward(g, -shift)[0]) / 2e-5
        assert torch.allclose(theta.grad[feature], difference, atol=1e-7)


def test_identical_pair_has_zero_js_and_location_penalty():
    g = fit.prepare(dict(ins=[[1., 1.]], sub=[[0., 0.]], fixed=[[0., 0.]], source=[0], jump=[[-1, 3]],
                         initial=0, entry=[0., 0.], owners=[0, 1], allowed=[True, True], **{'del': [0., 1.]}), 2)
    theta = torch.zeros(6, dtype=torch.float64)
    assert abs(float(fit.objective(dict(original=g, edited=g), theta))) < 1e-12


if __name__ == '__main__':
    test_graph_partition_and_gradient()
    test_identical_pair_has_zero_js_and_location_penalty()
    print('structured graph and gradients pass')

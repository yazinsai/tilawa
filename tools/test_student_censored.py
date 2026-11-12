"""Censored top-k supervision: analytic, numerical and gradient contracts."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import censored_soft_target_loss, censored_target_projection, soft_target_loss


def fixture(probabilities, q, ids=None):
    logits = torch.tensor(probabilities, dtype=torch.float64).log().requires_grad_()
    ids = torch.tensor([ids or list(range(len(q)))])
    q = torch.tensor([q], dtype=torch.float64)
    return logits, ids, q


def loss(logits, ids, q):
    return censored_soft_target_loss(logits.log_softmax(-1)[None, None], ids[None], q[None],
                                     torch.ones((1, 1), dtype=torch.bool))


def test_consistent_nonzero_tail_has_zero_loss_and_no_suppression_gradient():
    logits, ids, q = fixture([.48, .32, .12, .08], [.6, .4])
    value = loss(logits, ids, q)
    value.backward()
    assert abs(float(value.detach())) < 1e-12
    assert logits.grad.abs().max() < 1e-12
    logits.grad = None
    ordinary = soft_target_loss(logits.log_softmax(-1)[None, None], ids[None], q[None],
                                torch.ones((1, 1), dtype=torch.bool))
    ordinary.backward()
    assert (logits.grad[2:] > 0).all()  # Existing KD suppresses both omitted phones.


def test_retained_ratios_and_mass_bounds_hold_when_large_tail_is_capped():
    logits, ids, q = fixture([.15, .05, .7, .1], [.75, .25])
    p = censored_target_projection(logits.log_softmax(-1)[None], ids, q)[0]
    alpha = p[:2].sum()
    assert torch.allclose(p[:2]/alpha, q[0])
    assert alpha >= 1/(1+2*.25)-1e-12
    assert (p[2:] <= alpha*.25+1e-12).all()
    assert p.sum() == pytest.approx(1.)
    value = loss(logits, ids, q)
    value.backward()
    assert value > 0 and logits.grad[2] > 0


def test_uncapped_tail_keeps_student_ratios_without_inventing_teacher_identities():
    logits, ids, q = fixture([.3, .5, .12, .08], [.6, .4])
    p = censored_target_projection(logits.log_softmax(-1)[None], ids, q)[0]
    assert p[2]/p[3] == pytest.approx(1.5)
    assert p[0]/p[1] == pytest.approx(1.5)


def test_projection_matches_independent_constrained_optimizer():
    rng = np.random.default_rng(16)
    for _ in range(20):
        student = rng.dirichlet(np.ones(6))
        q = np.sort(rng.dirichlet(np.ones(3)))[::-1].copy()
        ids = [4, 1, 3]
        logits, retained, observed = fixture(student, q, ids)
        projected = censored_target_projection(logits[None], retained, observed)[0].numpy()
        omitted = [i for i in range(6) if i not in ids]

        def distribution(z):
            out = np.zeros(6)
            out[ids] = z[0]*q
            out[omitted] = z[1:]
            return out

        def objective(z):
            target = distribution(z)
            positive = target > 0
            return (target[positive]*np.log(target[positive]/student[positive])).sum()

        lo = 1/(1+len(omitted)*q[-1])
        result = minimize(objective, [lo, *([lo*q[-1]]*3)], method='SLSQP',
            bounds=[(lo, 1), *([(0, 1)]*3)], constraints=[
                {'type': 'eq', 'fun': lambda z: z.sum()-1},
                {'type': 'ineq', 'fun': lambda z: z[0]*q[-1]-z[1:]}],
            options={'ftol': 1e-12, 'maxiter': 1000})
        assert result.success, result.message
        np.testing.assert_allclose(projected, distribution(result.x), atol=2e-6, rtol=0)


def test_envelope_gradient_matches_finite_difference_in_capped_and_free_regions():
    for student in ([.15, .05, .7, .1], [.3, .5, .12, .08]):
        logits, ids, q = fixture(student, [.6, .4])
        assert torch.autograd.gradcheck(lambda x: loss(x, ids, q), (logits,), eps=1e-6, atol=1e-5)


def test_full_support_and_zero_smallest_entry_have_ordinary_kd_gradient():
    for student, target in [([.2, .5, .3], [.6, .3, .1]), ([.2, .5, .3], [1., 0.])]:
        logits, ids, q = fixture(student, target)
        censored = torch.autograd.grad(loss(logits, ids, q), logits)[0]
        ordinary = soft_target_loss(logits.log_softmax(-1)[None, None], ids[None], q[None],
                                    torch.ones((1, 1), dtype=torch.bool))
        assert torch.allclose(censored, torch.autograd.grad(ordinary, logits)[0], atol=1e-12)


def test_absolute_retained_mass_is_not_used():
    logits, ids, q = fixture([.3, .4, .2, .1], [.6, .4])
    assert loss(logits, ids, q) == loss(logits, ids, q*.125)


def test_mask_excludes_invalid_padding_and_empty_batch_retains_zero_gradient():
    logits = torch.tensor([[[1., 2., 0.], [float('nan'), 0., 0.]]], requires_grad=True)
    ids = torch.tensor([[[1, 0], [0, 0]]])
    q = torch.tensor([[[.6, .4], [0., 0.]]])
    lp = logits.log_softmax(-1)
    value = censored_soft_target_loss(lp, ids, q, torch.tensor([[True, False]]))
    assert torch.isfinite(value)
    clean = torch.randn(1, 2, 3, requires_grad=True)
    empty = censored_soft_target_loss(clean, ids, q, torch.zeros(1, 2, dtype=torch.bool))
    empty.backward()
    assert empty == 0 and clean.grad.abs().sum() == 0


def test_rejects_duplicate_unordered_and_out_of_inventory_observations():
    for ids, q in [([0, 0], [.6, .4]), ([0, 1], [.4, .6]), ([0, 4], [.6, .4]),
                    ([0, 1], [0., 0.]), ([0, 1], [float('nan'), .4])]:
        logits, ids, q = fixture([.2, .5, .3], q, ids)
        with pytest.raises(ValueError):
            loss(logits, ids, q)


def test_extreme_log_probabilities_stay_finite_and_normalized():
    logits = torch.tensor([-1000., -2000., 0., -1500.], dtype=torch.float64, requires_grad=True)
    ids, q = torch.tensor([[0, 1]]), torch.tensor([[.8, .2]], dtype=torch.float64)
    projected = censored_target_projection(logits.log_softmax(-1)[None], ids, q)
    assert projected.sum() == pytest.approx(1.)
    value = loss(logits, ids, q)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(logits.grad).all()

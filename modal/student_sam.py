"""Two-pass sharpness-aware updates for a single streaming acoustic model.

Each parameter tensor has perturbation RMS at most ``rho`` times its weight RMS.
The gradient direction maximizes the linearized loss within these layer balls.
The perturbation is discarded before the ordinary optimizer updates the
original weights. Neither perturbations nor extra branches are exported.
"""

import math
import random


def sharpness_step(model, optimizer, closure, rho):
    """Run a loss-dict closure at original and adverse weights on one draw.

``rho=0`` is the matched ordinary-optimizer control. Forward buffers retain
exactly one pass's changes. A training-loss backtrack shrinks the proposed
radius when nonlinearity makes the adverse direction lower the actual loss.
Exceptions/nonfinite losses or gradients restore
weights and clear gradients without updating the optimizer.
    """
    import torch
    import numpy as np

    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and nonnegative")

    def rng_state():
        return (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                random.getstate(), np.random.get_state())

    def set_rng(state):
        torch.set_rng_state(state[0])
        if state[1]:
            torch.cuda.set_rng_state_all(state[1])
        random.setstate(state[2])
        np.random.set_state(state[3])

    def backward(out):
        if not torch.isfinite(out["loss"]):
            raise FloatingPointError("nonfinite acoustic loss")
        out["loss"].backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("nonfinite acoustic gradient")

    original = []
    perturbations = []
    buffers = []
    before_buffers = [(b, b.detach().clone()) for b in model.buffers()]
    after_first = None
    optimizer.zero_grad(set_to_none=True)
    before_first = rng_state()
    try:
        first = closure()
        backward(first)
        if rho == 0:
            optimizer.step()
            return {k: v.detach() for k, v in first.items()}

        after_first = rng_state()
        buffers = [(b, b.detach().clone()) for b in model.buffers()]
        active = [p for p in model.parameters() if p.grad is not None]
        with torch.no_grad():
            for p in active:
                scale = p.square().mean().sqrt().clamp_min(1e-3)
                grad_rms = p.grad.square().mean().sqrt()
                if grad_rms > 0:
                    original.append((p, p.detach().clone()))
                    delta = p.grad * (rho * scale / grad_rms)
                    perturbations.append(delta)
                    p.add_(delta)
            for b, saved in before_buffers:
                b.copy_(saved)
        optimizer.zero_grad(set_to_none=True)
        fraction = 1.
        for attempt in range(9):
            set_rng(before_first)
            with torch.no_grad():
                for b, saved in before_buffers:
                    b.copy_(saved)
            second = closure()
            if not torch.isfinite(second["loss"]):
                raise FloatingPointError("nonfinite perturbed acoustic loss")
            if second["loss"] >= first["loss"] or fraction == 0:
                break
            fraction = fraction / 2 if attempt < 7 else 0.
            with torch.no_grad():
                for (p, saved), delta in zip(original, perturbations):
                    p.copy_(saved).add_(delta * fraction)
        backward(second)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        with torch.no_grad():
            for p, saved in original:
                p.copy_(saved)
            for b, saved in buffers:
                b.copy_(saved)
        if after_first is not None:
            set_rng(after_first)

    optimizer.step()
    result = {k: v.detach() for k, v in first.items()}
    result["perturbed_loss"] = second["loss"].detach()
    result["effective_rho"] = first["loss"].new_tensor(rho * fraction)
    return result

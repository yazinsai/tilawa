"""Execution audit of the trainer's pinned ScaledAdam (no audio inputs).

Set TILAWA_OPTIMIZER_SOURCE to the pinned icefall optim.py. The AST extraction
omits only its unused lhotse test-seeding import, not optimizer definitions.
"""

import ast
import hashlib
import os
from pathlib import Path

import pytest
import torch

from student_contrast import ContrastHead


def pinned_optimizer():
    path = os.environ.get("TILAWA_OPTIMIZER_SOURCE")
    if not path:
        pytest.skip("Pinned icefall source must be provided for the execution audit")
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == "7282cd6db1d234e086110feb1f39f076a76095a6b026b939cebc81f2ed0f5d45"
    tree = ast.parse(data)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "lhotse.utils")]
    scope = {"__name__": "contrast_optimizer_audit"}
    exec(compile(tree, str(path), "exec"), scope)
    return scope["ScaledAdam"]


def test_scaled_adam_ordinary_updates_depend_on_an_invisible_common_shift_but_contrast_updates_do_not():
    optimizer = pinned_optimizer()
    torch.manual_seed(53)
    native = torch.nn.Linear(7, 5)
    shifted = torch.nn.Linear(7, 5)
    with torch.no_grad():
        shifted.weight.copy_(native.weight + .5)
        shifted.bias.copy_(native.bias + .5)
    heads = [native, shifted, ContrastHead(native), ContrastHead(shifted)]
    x = torch.randn(13, 7)
    target = torch.tensor([1, 2, 3])
    initial = [h(x).detach().log_softmax(-1) for h in heads]
    for lp in initial[1:]:
        torch.testing.assert_close(lp.exp(), initial[0].exp(), atol=2e-6, rtol=2e-5)
    for head in heads:
        opt = optimizer(head.named_parameters(), lr=.045, clipping_scale=2.)
        lp = head(x).log_softmax(-1)
        loss = torch.nn.functional.ctc_loss(lp[:, None], target, torch.tensor([len(x)]),
            torch.tensor([len(target)]), blank=0, reduction="sum") / len(x)
        loss.backward()
        opt.step()
    ordinary_a, ordinary_b, contrast_a, contrast_b = [h(x).detach().log_softmax(-1) for h in heads]
    # Compare identifiable phone-minus-blank preferences, not common logits.
    pref_a, pref_b = ordinary_a[:, 1:] - ordinary_a[:, :1], ordinary_b[:, 1:] - ordinary_b[:, :1]
    assert (pref_a - pref_b).abs().max() > 1e-4
    torch.testing.assert_close(contrast_a.exp(), contrast_b.exp(), atol=2e-6, rtol=2e-5)

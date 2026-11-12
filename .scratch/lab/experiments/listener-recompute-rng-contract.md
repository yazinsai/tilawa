# listener-recompute-rng-contract

Status: killed on the premise; no engine change or replacement model.

## Prediction and mechanism

The card predicted one recovered added-word mistake, no losses, listener −1
and overall −1. It proposed that activation recomputation resamples stochastic
masks during backward, so weak-consonant gradients differ from those of the
original forward execution. The proposed intervention was RNG replay followed
by a bounded upper-block refit against an identical-data current-recomputation
control. The forecast was below the standalone fold threshold.

## Source audit

The checked-in trainer, `modal/student_train.py`, pins Icefall to
`3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`. Its `Student.forward` directly
executes subsampling, the causal Zipformer encoder and the CTC head. The training
loop invokes that forward once through `losses`, then `loss.backward()` and
`ScaledAdam.step()`. Neither it nor the other `modal/student_*.py` files or
`tools/student.py` enables activation checkpointing, compilation with
recomputation, or a custom-generator replay path. Saving `ckpt.pt` persists
training state; it does not recompute activations. Dataset augmentation runs
before the acoustic forward and is not called again by backward.

The pinned upstream implementation does contain memory-saving custom backward
functions, so absence of `torch.utils.checkpoint` alone was insufficient. The
audit downloaded and inspected the actual pinned `zipformer.py`, `scaling.py`
and `subsampling.py`, including every custom `backward` in those files.

| Operation | Forward/backward behavior | RNG replay implication |
|---|---|---|
| `ActivationDropoutAndLinearFunction` | Forward samples dropout and saves `x`, weight, bias and **the dropout mask**. Backward recomputes the deterministic Swoosh activation, then multiplies both the weight-gradient activation and input gradient by that saved mask. | The original stochastic draw is already retained. |
| `BiasNormFunction` | Backward reconstructs input if needed and recomputes normalization from saved input/output, scales, bias and log scale. | No stochastic operation is recomputed. |
| `Dropout3` / `MulForDropout3` | Mask is sampled in forward; backward uses saved forward output, without resampling. | No new random draw in backward. |
| Sequence, feature, ConvNeXt and bypass layer skipping | Masks and branch choices are sampled in the ordinary encoder forward and applied using ordinary tensor operations. | Backward does not invoke these forward methods again. |
| Attention, positional and CTC-head dropout | Ordinary PyTorch dropout in the direct forward graph. | No enclosing activation-checkpoint region re-executes them. |
| Balancer / whitening / parameter-limit helpers | Gradient modifications use saved tensors. Python randomness selects some helpers in forward or controls diagnostic logging. | These are not re-executions of a stochastic acoustic forward. |

The ConvNeXt `caching_eval(...)` call is commented out in this pinned revision;
the active path calls `forward_internal(x, mask)` directly. No explicit
`generator=` argument appears in the audited acoustic implementation.

The separate Swoosh implementations save forward-sampled quantized derivatives
for backward. Their randomized derivative encoding is not a dropout mask
resampled by recomputation. Replacing that approximation or altering the
gradient regularizers would change a different mechanism and training recipe.
Supplementary review of k2's native `swoosh.cu` found forward-only random draws,
a saved encoded derivative, and deterministic `SwooshForwardAndDeriv` kernels.
That file's last upstream change was
`47fecdd7ac6c889cfa34a6488a87e011b7bc3fad` (2023-01-30), before the trainer's
2026-06-25 wheel build. This source review does not substitute for executing
the trainer's actual k2 binary.

Pinned upstream source references:

- [Fused activation/dropout/linear forward and backward](https://github.com/k2-fsa/icefall/blob/3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1/egs/librispeech/ASR/zipformer/scaling.py#L1513).
- [BiasNorm recomputation](https://github.com/k2-fsa/icefall/blob/3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1/egs/librispeech/ASR/zipformer/scaling.py#L369).
- [Dropout3 saved-output backward](https://github.com/k2-fsa/icefall/blob/3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1/egs/librispeech/ASR/zipformer/scaling.py#L1305).
- [ConvNeXt active forward](https://github.com/k2-fsa/icefall/blob/3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1/egs/librispeech/ASR/zipformer/subsampling.py#L101).
- [Native Swoosh saved-derivative backward and deterministic recomputation](https://github.com/k2-fsa/k2/blob/47fecdd7ac6c889cfa34a6488a87e011b7bc3fad/k2/python/csrc/torch/v2/autograd/swoosh.cu#L147).

Downloaded source SHA-256 values were
`d5a5db141cd00b4beaa990fe6827599b3248b2fddd87e8dc7809b0dbd9d6633b`
(`scaling.py`),
`4d5365ed19dee07f5a51fcaaa26e18310f7fafbdf2a391e24a7cc01d1b2e8cfe`
(`zipformer.py`), and
`e04de281f136560e2d44b5f7df174d0364b5feb2c22608173b60de02630c2a66`
(`subsampling.py`). Diagnostic downloads remain ignored under
`.autolab/recompute-audit/`.

## Validation and limits

An ignored diagnostic script was prepared to compare the pinned fused
activation/dropout/linear function, BiasNorm and Dropout3 against retained
activations, checking input/weight/bias gradients and random-state stability.
Its syntax was checked, but it did not execute. The authorized supervisor's
environment probe (run 1) failed with `NotFoundError: No such file or directory`.
A training-image GPU request (run 2) was rejected because GPUs are available
only to bet cards. The CPU training-image retry (run 3) failed with the same
`NotFoundError`. Request and error records remain in ignored `.autolab/`
files. No numerical pass, gradient-error tolerance result, or audio-based
gradient comparison is claimed.

The source audit falsifies the proposed resampling premise independently of
which authorized recording or consonant enters the graph. No engine change was
made: adding replay would either preserve the already-saved masks or introduce
activation checkpointing solely to test a new hypothetical defect. Accordingly
the experiment stops before refitting either arm. No training examples,
help-train recordings or practice incidents were used to optimize a model or
select a checkpoint.

This does not establish exact equivalence of every custom gradient to a
mathematically exact retained-activation implementation. Quantized derivatives,
mixed precision and explicit gradient regularization remain distinct questions.
It establishes that the audited source has no stochastic forward operation
recomputed with a different draw for this intervention to correct.

Practice-score calls: 0. Targets won: 0 (no changed listener); listener and
overall practice deltas were not measured. Core/SDK tests were not run because
their behavior and source were unchanged. `lab-experiment.json`, scoring,
tracker, checker, inventory, clock, verification and supplied assets were left
unchanged. No `.autolab/model.json` is declared.

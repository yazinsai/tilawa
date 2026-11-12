# BiasNorm backward contract

Status: killed on the premise. The pinned student implementation already
differentiates the exact input-dependent normalization denominator. No engine
change, training refit, model export or practice score is justified by this card.
The experiment card is unchanged.

## Prediction and mechanism

The prediction was one recovered added word, no losses, listener −1 and overall
−1. The proposed mechanism was an incorrect fused BiasNorm derivative that treats
the normalization scale as fixed and therefore sends weak-consonant training
gradients in the wrong directions. The planned intervention preserved forward
values, froze the acoustic stem and compared a bounded upper-block refit with an
identical-data, identical-update original-backward control. Authorized transcripts
and help-train training speakers' none-labelled takes would supply supervision;
release-held-out speakers and practice labels would not select checkpoints.

The early criterion required reproducible derivative discrepancies beyond normal
floating-point error in at least two target windows. The user also requires
stopping when the mechanism is wrong. Source inspection establishes that the
claimed missing denominator derivative is absent before any refit or target
checkpoint selection.

## Source validation

`modal/student_train.py` pins icefall commit
`3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1` and constructs the upstream
`Conv2dSubsampling` and `Zipformer2`. There is no repository-local BiasNorm
override in `modal/student_*.py` or `tools/student.py`.

The exact pinned [scaling.py](https://github.com/k2-fsa/icefall/blob/3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1/egs/librispeech/ASR/zipformer/scaling.py#L369)
has SHA-256
`d5a5db141cd00b4beaa990fe6827599b3248b2fddd87e8dc7809b0dbd9d6633b`.
`BiasNormFunction.backward` enables autograd, marks the saved input, bias and
log-scale as requiring gradients, recomputes

```python
scales = mean((x - bias)**2, channel_dim, keepdim=True)**-0.5 * log_scale.exp()
ans = x * scales
ans.backward(gradient=ans_grad)
```

and returns gradients for all three inputs. The denominator remains connected to
both `x` and `bias`; it is not detached during this recomputation. Detaching the
saved tensors makes them local leaves for this vector-Jacobian product and does
not remove that dependence.

For `C` channels, `z = x − bias`, `m = mean(z²)`,
`s = exp(log_scale)/sqrt(m)` and upstream gradient `g`, the input gradient is

```text
dx = s*g − s*z*sum(g*x)/(C*m)
```

The existing backward includes the second term through autograd. Replacing it
with reference differentiation of this same expression would retain its current
first-order derivative.

Both actual student construction sites use `BiasNorm(channels)` with the default
`store_output_for_backprop=False`: pinned `subsampling.py:286` and
`zipformer.py:660`. The original input is saved, so these sites do not reconstruct
it by dividing the output by a saved scale. The optional output-saving branch
does that reconstruction, then differentiates the same complete expression; its
rounding behavior is separate from the claimed missing derivative. The module's
existing log-scale parameter limiter is also separate from this normalization
operator and provides no reason to remove the denominator correction.

## Numerical validation and limits

The numerical probe was designed to compare the original operator with an
independent ordinary-autograd expression for input, bias and log-scale gradients:
72 fp32/fp64 cases cover channel axes −1, 1 and 0, both storage modes, activation
amplitudes 0.01, 1 and 100, and two fixed seeds. It also checks the analytic input
gradient and fp64 symmetric directional differences. Another 18 cases cover bf16
inputs with fp32 parameters in the student's actual input-saving mode. A radial
test with zero bias must reject a deliberately fixed-scale derivative, providing
a sensitivity check for the claimed defect. These are generic operator probes,
not practice-specific model or acceptance rules.

This probe did **not execute successfully**. Supervisor requests 1 and 2 returned
`NotFoundError: No such file or directory` without downloaded outputs. Request 3,
using the supported training image and a non-ignored temporary diagnostic script,
returned the same error without outputs. Consequently there are no numerical
parity measurements, finite-difference results or target activation traces to
claim. The conclusion is based on the exact pinned source and its derivative,
not synthetic results that were merely planned. The diagnostic script passed
local Python syntax compilation; that is not a numerical check.

The existing core/SDK suite was requested through the supplied six-shard harness.
Its first request (2) and separate harness retry (4) both returned the same
`NotFoundError` without outputs. No successful core/SDK build or test run can be
claimed. No behavior was changed and no replacement normalization test was added
to the engine. Diagnostic scripts and fetched upstream sources remain in ignored
artifacts rather than the engine diff.

No target activation gradients or warning recoveries have been measured. In the
student's input-saving mode, backward recomputes the full expression on the
original saved input. This establishes that it differentiates the denominator,
but does not explain the five missed additions or exclude other training defects.
There is no changed listener to score: reported target wins are zero
because the experiment stops without an intervention, not because a practice
replay measured zero gain. No overall practice delta is claimed. Zero of the
three practice-score allowances were used.

Only this write-up is committed. No engine changes needed reverting; no model
bundle or `.autolab/model.json` was created. The ignored `.autolab/outcome.json`
records a premise kill with zero targets won. Local `git diff --check` passed,
and `lab-experiment.json` matches the branch's original version exactly.

# Stable CTC training gradients

Status: opt-in training intervention implemented; target diagnosis, matched head
refit and export blocked by the supplied supervisor runtime. The numerical
premise is unresolved. This is not a killed experiment or a completed listener
model. The experiment card is unchanged.

## Prediction and mechanism

Predict listener -1 and overall -1, with substantial zero-gain risk. The proposed
cause is disappearance or sign reversal of legitimate alternative-alignment
gradients through underflow or cancellation despite a finite sequence loss.

The premise requires at least two registered targets with a missing planted
consonant whose transcript-conditioned gradient vanishes or changes sign under
the original arithmetic relative to a high-precision reference on identical
logits. Target transcripts and gradients are diagnostics only: never updates,
regularization selection or checkpoint selection. This premise has not been
measured. Infrastructure failure is not evidence against it.

## Implemented intervention

modal/student_ctc.py implements CTC directly on logits, with float64 log-softmax,
transcript-conditioned forward/backward log accumulators and log-space aggregation
of repeated-label state occupancy. Skip transitions preserve ordinary CTC
semantics. State occupancy is normalized per frame in log space before gradient
combination. A signed expm1 difference combines acoustic probability and
occupancy before rounding to the logit dtype, avoiding an intermediate rounded
occupancy or separate rounded log-softmax backward operation.

Float64 gradients are retained through upstream recording weights before
conversion at the head boundary. Float32 callers receive float32 losses. Only
first derivatives are supported. Computation is explicitly outside autocast.

modal/student_train.py exposes ctc_arithmetic="stable" in training configuration
and its CLI, includes the new module in the Modal image, and requests raw logits
from the same model without changing architecture or weights. CTC remains summed
over recordings and divided by the existing sum of valid output lengths, with
blank zero and zero_infinity=True. KD retains its original autocast log-softmax,
validity mask, teacher gathering, delay, reduction and configured weight.

The default and ctc_arithmetic="native" preserve the original arithmetic as the
matched control. Export and ordinary model calls still return original log
probabilities. This is a training intervention, not an inference probability
conversion or a warning-bar change. No trained head or model bundle is claimed.

## Source findings

The original loss already casts returned log probabilities to float32 before
native F.ctc_loss outside bfloat16 CUDA autocast. The source does not establish a
bfloat16 CTC accumulator. The previous audit checked seven loss-contract
properties. Previously inspected upstream PyTorch 2.8.0 CPU/CUDA CTC sources use
max-shift log-space sums but subtract probability and occupancy in backward.
That motivates the new stable combination; it does not identify the unavailable
supervisor runtime or demonstrate a defect on any target.

Simply promoting already-rounded log probabilities would leave the separate
log-softmax gradient combination unresolved, which is why the new API takes
logits. It cannot recover information already rounded in a reduced-precision
head projection or guarantee that parameter-gradient accumulation preserves it.

## Validation and limits

tools/test_student_ctc.py adds small CPU checks against independently enumerated
CTC paths and double-precision native PyTorch. Cases cover empty/repeated
transcripts, alternate blank IDs, impossible alignments, valid lengths, recording
weights, zero padding gradients, longer sequences, finite differences, autocast
boundaries and gradients through linear-head parameters. A dispatch check uses
the actual losses function with a tiny model to verify the frame divisor,
unchanged KD and weighted loss composition.

A synthetic two-frame case excludes an almost impossible all-blank path and
produces a logit gradient around 3.8e-11, retained at float64, float32 and bf16
logit boundaries. On identical float32 logits, CPU native CTC returns a zero loss
and a correctly signed gradient of 3.5470e-11; stable CTC returns loss 3.7751e-11
and gradient 3.7751e-11, matching the double reference. This exercises complete
CTC gradients rather than subtracting two chosen scalars. The native gradient
does not vanish or reverse sign in this example; it is not evidence satisfying
the target premise. All 10 numerical tests passed locally on CPU with PyTorch 2.8.0 in an
isolated environment under .autolab/. Python syntax and git diff --check pass.
CUDA execution and performance have not been validated. The explicit Python
time-step loops prioritize inspectable arithmetic over throughput; a bounded
refit still needs a measured runtime before choosing its update budget.

The first attempt's supervisor requests 1–3 failed before execution. This attempt
used the same authorized service:

| Request | Requested work | Result |
|---|---|---|
| 4 | PyTorch/runtime and read-only supplied-data layout | NotFoundError: No such file or directory, before execution |
| 5 | Six shards of the supplied core/SDK build/test harness | Same infrastructure error; no build or tests ran |
| 6 | New CTC unit tests in the supervisor container | Same infrastructure error; tests did not execute there |

No alternative Modal access was attempted. Requests/results and disposable
diagnostics remain ignored under .autolab/. Core/SDK tests are unverified, not
passing. No scorer, lab tooling, supplied manifest, evidence or test data was
edited.

## Outstanding model work

The supervisor must execute before acoustic target diagnosis can establish the
premise. If fewer than two targets satisfy it, revert the opt-in engine changes,
record a premise kill and write the killed outcome. Otherwise cache frozen
encoder features from authorized transcribed recordings and help-train training
speakers' none-labelled takes only, excluding release-held-out speakers and
without treating mistake scripts as performed transcripts.

Run stable/native head refits with identical data, initial head, updates,
objective weights and training-speaker partitions. Retain ordinary speech and
real pauses; select checkpoints solely within those training-speaker partitions.
Export and verify the complete primary listener bundle under this experiment's
persistent scratch directory, preserving vocabulary, frame clock, tracker,
checker and folded insertion verification. Only then declare .autolab/model.json.
No model is declared; scoring the unchanged model would not evaluate this loss.

There have been zero practice-score calls. Targets evaluated, targets won and
practice deltas are unknown, not measured zero. Three or more built-change target
recoveries require a practice score before a gate decision; fewer than three
require rollback and a recorded kill. Neither condition can currently be tested.
No fold or killed outcome is recorded.

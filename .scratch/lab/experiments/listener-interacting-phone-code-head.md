# listener-interacting-phone-code-head

Status: implementation prepared; blocked before runtime tests, training and
acoustic validation by the supplied supervisor's infrastructure error.
No evaluated primary model is declared. This is neither a scored result nor a
killed acoustic experiment.

## Prediction

The unchanged card predicts two recovered additions and one collateral wrong
call: listener −1 and overall −1. Its six registered targets are p07 at
144.12 s, p33 at 156.16 s, p48 at 39.98 s, p56 at 106.76 s, and p57 at
95.26 s and 112.26 s. Recovery must retain previously missing planted
consonants in order in exported full-session decoding, relative to both the
shipped head and the equal-parameter nonlinear control. The user's later
instruction requires at least three target wins before practice scoring;
the card's two-target acoustic criterion alone would not meet that instruction.

## Mechanism and implementation

Freeze the shipped int8 ONNX encoder and its existing classifier. Fit an
additive phone-score residual on its native 256-channel representation:

```text
delta[c,x] = sum_i C[c,i] (W_single x)[i] / sqrt(16)
           + sum_(i<j) C[c,i] C[c,j]
               (W_pair tanh(W_hidden x + b_hidden))[i,j] / sqrt(120)
log p[c,x] = log_softmax(shipped_log_probs[x] + delta[:,x])[c]
```

The signed 43-by-16 code is fixed using inventory size and a fixed seed, with
eight positive bits per phone and 21 or 22 positive labels per column. The
generator explicitly excludes complementary rows: complements would have
identical pair products and therefore could not be distinguished by the
nonlinear interaction branch. All 43 pair-product patterns are distinct.
Both score branches start at zero, preserving the initial shipped
distribution. This uses acoustic-conditioned joint code coefficients rather
than independent Bernoulli likelihoods.

The interaction bottleneck has 52 units. Its exact-parameter conventional
control is a tanh dense residual with 79 hidden units. Each has 23,700
trainable parameters, with no unused padding parameters. The code dictionary
is a fixed output factorization; it does not imply independently supervised
error correction. Its nonlinear score can be written as a factored categorical
nonlinear head plus a linear skip branch. The comparison therefore tests this
geometry and constraint against the conventional dense control, rather than
claiming that code products alone create a new probability family.

`modal/student_interacting_head.py` implements these heads, grouped partitions
and CTC plus KL preservation. `modal/student_interacting_train.py` prepares
frozen full-utterance features and selects preservation from 0.1, 1 and 10
using three speaker-grouped training partitions and a fixed 18-epoch schedule.
The selection criterion is held-out-speaker macro phone error plus KL, with
the strongest preservation breaking ties. Each arm receives identical
original and two shipped-augmentation views. Feature normalization is fitted
only on the training side of each partition. Final heads use all admitted
training speakers after their regularization is selected.

Admission accepts help-train training speakers' none-labelled complete takes,
rejecting events, repeats, unsupported or excluded recordings. Additional
recordings require explicit authorization, verified performed phone
transcripts, audio SHA-256 and train split. Any speaker seen in another help
split and the explicit supplied exclusion list are excluded. Scripted-mistake
labels are never converted into performed transcripts. Shipped noise/RIR
banks are required and hashed. Their availability, the exclusion-list path,
additional transcribed recordings and the admitted training-speaker count
could not be established because the data probe never executed.

`modal/student_interacting_export.py` exposes the native feature tensor and
grafts the selected residual into the primary ONNX graph. The existing SDK
can consume it through its ordinary directory loader. The ordered alphabet,
state tensors, frame clock and frozen cache computation are inherited.
Tracker, checker and insertion rules have no edits. Export checks finite
normalization and weights, prohibits overwriting the source bundle, copies
adjacent files recursively and verifies checksums. Training requires the
existing SDK insertion teacher and its external data; their identity must
survive preparation and both exports. No inference fallback is declared.

## Validation performed

Read the shipped training/export implementation, SDK stream/drain geometry,
reference schema, help-label schema and adjacent insertion-teacher loader.
All four new Python files pass `py_compile`. No tracked existing engine file,
card, scorer, lab tool, supplied manifest, test data or evidence was modified.

A five-second, dependency-free audit executes the actual codebook and
parameter-width functions with a list-only replacement for Torch's tensor
constructor. It verifies 43 unique codes, 43 unique pair patterns, constant
row weight, the balanced column counts, and exactly 23,700 parameters per
arm. An analytic mixed acoustic derivative of an isolated nonlinear
pair-score contrast is nonzero. The script and its result remain untracked
at `.autolab/interacting-geometry-check.py` and
`.autolab/interacting-geometry.json`. This establishes inventory geometry and
the score formula only; it does not execute Torch training or ONNX inference.

Added `modal/test_student_interacting_head.py` for zero residual initialization,
nonlinear phone odds, CTC gradients, speaker admission, primary ONNX/Torch
parity, unchanged cache outputs, stream draining, recursive auxiliary copying
and rejection of missing insertion assets. These tests have **not run**.
Preparation and fitting also include source/extracted-output parity, prepared
data hashes, finite/shape/phone checks and full-utterance exported-output parity.
Those checks likewise remain unexecuted.

Every supplied supervisor request failed before its command executed:

| Request | Requested work | Result |
| --- | --- | --- |
| 1 | Read-only runtime/storage probe | `NotFoundError: No such file or directory` |
| 2 | Required core/SDK suite, six shards | Same error |
| 3 | New Python head/export tests, training image | Same error |
| 4 | Read-only supplied-data/training-input probe, training image | Same error |

The failures were independent of the requested command and occurred on both
default and training images. No alternative route to Modal or external data
was attempted. The local Python has no Torch or ONNX runtime; heavy work
remains reserved for Modal as instructed.

## Limits and remaining work

No training input was read and no head was fitted or exported. Targets
acoustically evaluated: **0 of 6**. Target recoveries, listener/overall deltas,
clean-session warnings and studio drift are **unmeasured**, not measured zero.
Practice scores requested: **0 of 3**. No `.autolab/model.json` exists, and no
killed outcome or fold is recorded. Infrastructure failure does not establish
the kill premise, so the implementation is retained as unvalidated work.

Once the supplied supervisor works, establish the authorized training inputs,
run the new Python tests and core/SDK suite, prepare the frozen observations,
fit/select both heads exclusively on grouped training partitions, and finish
and verify both complete bundles under the experiment scratch directory.
Compare exported full-session SDK decoding of all registered targets against
both controls before requesting practice scoring. If fewer than three targets
recover, revert the engine implementation and record the killed acoustic
result. If at least three recover, declare only the complete verified
interacting bundle and run the authorized practice score before a gate
decision. Ordinary-reading KL preservation alone cannot establish safety.

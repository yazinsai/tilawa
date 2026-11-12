# Listener contrast-space head training

Status: implementation prepared; acoustic validation is blocked by the supplied
supervisor. No replacement listener has been trained or declared. The card is
unchanged, and the premise has not been killed.

## Prediction and mechanism

The registered prediction is one recovered added-word miss with no losses:
listener −1 and overall −1. This forecast is below the fold threshold and within
the stated two-call noise band. It is not an assertion that the branch earns a
fold.

Softmax ignores a feature-dependent value added to every class logit. A phone
head therefore has a redundant common weight vector and bias. The proposed
failure is that ScaledAdam uses those invisible parameters when scaling useful
contrast updates, so equivalent classifiers need not learn identically.

The source audit supports the optimizer part of this premise. The trainer pins
icefall revision `3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1` and uses ScaledAdam
with clipping scale 2. Its `scaling_step` computes RMS over complete parameter
tensors and multiplies the adaptive update by that RMS. Its scale update also
uses the complete parameter/gradient inner product. A common-logit shift can
alter the former while leaving starting probabilities unchanged. The pinned
optimizer source SHA-256 is
`7282cd6db1d234e086110feb1f39f076a76095a6b026b939cebc81f2ed0f5d45`.
This is an optimizer finding, not evidence that the five target consonants
benefit from a head refit.

## Implementation

`modal/student_contrast.py` constructs a deterministic Helmert basis of the
entire class inventory, including blank. Its columns are orthonormal and
perpendicular to the all-classes vector. The trainable head has C−1 weight rows
and biases. Initialization projects the centered source parameters, and export
folds the trained coordinates back into an ordinary C-class affine projection.
The basis is fixed rather than learned.

Three ordinary-coordinate controls start from the original head, its centered
equivalent, and a centered head with a common-logit shift. The shift direction
uses a fixed seed; its RMS comes from the source head, not practice evidence.
All four arms share frozen features, complete transcripts, fixed augmented
views, objective weights, ScaledAdam settings and Eden schedule. Each arm
evaluates one recording objective and backward pass per update, averaging its
three views. Dropout draws, recording order and the total number of backward
evaluations are identical. The intended parameter-dimension difference remains;
wall time and optimizer arithmetic are not asserted to be bit-for-bit equal.

One shared epoch, including epoch zero, is selected by the mean objective across
the four arms on separate training speakers, averaging recordings within each
speaker and speakers equally. The horizon is twelve epochs. Controls are not
selected or restarted according to target results. The report records the
selected update count and every arm's complete training evaluation count.

`modal/student_contrast_train.py` provides `prepare` and `fit`. Preparation
requires authorized performed-transcript enrollment, help-train training
speakers' none-labelled takes, an explicit held-out-speaker exclusion list,
shipped augmentation banks and source optimizer/objective configuration. It
does not discover another corpus or guess transcripts. Any speaker appearing
in a non-training split in either corpus is excluded before audio is read.
None takes must have empty events, non-errors and unsupported lists; mistake
scripts and repeats are not admitted as canonical transcripts. Audio hashes
prevent duplicates, and cached-view hashes protect matched inputs. Whole
speakers are assigned deterministically to fit or selection, and both
partitions must contain both training sources.

The objective is the complete recording's frame-normalized CTC, configured KD
when present, and shipped-output KL preservation with unit weight. Preservation
includes all frames, including ordinary speech and real pauses. The source
configuration supplies learning rate, Eden parameters and CTC/KD weights.
Every arm uses the same objective; no loss weight is chosen on practice.
KD requires matching complete clocks rather than silently cropping targets.
Preservation limits drift but does not guarantee loss-free recognition.

`modal/student_contrast_export.py` reuses the native graph extraction/export
infrastructure from `exp/listener-curvature-aware-head-refit`, with a material
correction for this card: a dynamically quantized head's rounded activations
are retained. Using dequantized weights on raw features would change the
starting probabilities and confound the experiment. Native MatMulInteger
activations are reconstructed after their existing quantizer, and all arms use
those same inputs. Unsupported projection layouts fail rather than guessing.

Initialization is mathematically probability-equivalent, subject to float32
projection/accumulation rounding. Every cached view must pass probability
parity with the shipped classifier (absolute tolerance 2e-6, relative tolerance
2e-5) before any optimizer is constructed. Feature exposure must preserve the
original outputs and caches exactly. No real source model passed these checks
here because the supervisor never executed. The synthetic tests do not replace
that prerequisite.

Export retains the encoder, its activation rounding, ordered alphabet, blank,
frontend, windows, states and posterior clock. SDK decoding, tracker, checker
and insertion verification are unchanged. The teacher graph and every external
weight file must exist and be copied with identical checksums. Each of the four
exported bundles must match its fitted head on all selection views, with export
probability tolerance 2e-4. `training-report.json` is written only after all
bundles pass. Existing manifests are never rewritten; newly generated manifests
describe the new graphs.

## Validation and limits

Fifteen small local unit tests passed without skips using generated features,
transcripts and ONNX graphs. The command was:

```sh
TILAWA_OPTIMIZER_SOURCE=.scratch/contrast-probe/optim.py \
  .autolab/contrast-venv/bin/python -m pytest -q \
  --basetemp=.autolab/contrast-pytest modal/test_student_contrast*.py
```

The pinned optimizer source was downloaded for the audit and is not vendored
into engine code. Local dependencies were Torch 2.15.0, ONNX 1.23.2, ONNX Runtime
1.31.1 and NumPy 2.5.4. The production trainer pins Torch 2.8.0; real training
and its dependency compatibility remain unverified.

Tests cover basis completeness including blank, common-shift initialization,
folded trained-head parity, control probability identity and different RMS,
speaker grouping and cross-corpus exclusion, duplicate audio and transcript
identity checks, corpus path containment, identical update counts and schedules,
no encoder gradients, complete pause preservation, impossible CTC transcripts,
and rejection of a mismatched starting classifier. Native float and dynamic-int8
fixtures check unchanged source outputs/states, activation rounding, fitted-head
export, manifest contracts, verifier copying and missing-asset rejection.

The actual pinned ScaledAdam execution audit shows different phone-minus-blank
updates after one identical CTC step from shifted ordinary heads with identical
initial probabilities. Contrast heads initialized from those classifiers remain
equivalent after that step. This audit omits only the optimizer file's unused
lhotse test-seeding import; optimizer definitions are unchanged. It establishes
the optimization dependence on synthetic features, not target sensitivity.

Five supervisor requests failed before execution with
`NotFoundError: No such file or directory`, each reporting USD 0.00:

1. Initial supplied-data inspection.
2. A second supplied-data/model/dependency inspection.
3. Small mathematical and optimizer tests.
4. The required six-shard core/SDK build and tests.
5. The completed mathematical/optimizer/native-export tests.

No core/SDK pass or test failure was obtained. No training recordings, practice
audio, labels or evidence were read remotely. No feature cache was prepared,
no matched head was fitted, no complete real model was exported, and no target
was replayed. Practice-score calls: 0 of 3. Target wins, losses, component delta
and overall delta are unknown. Python syntax compilation and final whitespace
checks pass.

## Remaining experiment work

Once the supplied supervisor works, inspect its available source bundle and
authorized enrollment, prepare the shared training-only cache, and run the
matched refit with the pinned optimizer and verified source configuration.
The runner requires `prepare --model --help --transcribed --exclusions
--augmentation --config --out`, then `fit --prepared --optimizer --out`, with
cache/export directories under `/scratch/listener-contrast-space-head-training`.

First test the registered kill criterion on exported full-session target
evidence: fewer than two targets changing missing-consonant-versus-blank
preference across probability-equivalent ordinary controls kills the premise.
If it survives, measure actual confirmed additions from the selected contrast
bundle. The user's final rule requires at least three target wins before
practice scoring. Below that floor, revert the engine implementation and record
a measured kill. At three or more wins, run the supplied practice score before
narrowing or deciding; the fold gate owns collateral-loss decisions.

Service failure does not falsify the premise. No killed outcome or
`.autolab/model.json` is written, and scoring the shipped listener would not
evaluate this implementation. The implementation remains on this branch for
completion; it is not a trained candidate ready for the fold gate. No scorer,
lab tooling, existing manifest, test data, evidence or card was changed, and no
fold was performed.

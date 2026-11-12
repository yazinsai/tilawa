# Listener class-local update scales

Status: needs-info — implementation and synthetic validation complete; the
supplied supervisor fails before execution. No trained listener or acoustic
result is available. The experiment card is unchanged.

## Prediction and premise

Prediction: one recovered added-word miss, listener −1 and overall −1, with
no losses predicted. This is within the two-call noise band and below the
standalone fold threshold; neither precision nor a fold is established.

The proposed failure is direct coupling through ScaledAdam's parameter-scale
statistics: a weak consonant's update uses the magnitude of unrelated phone
rows. This differs from changing consonant gradient weights, separating
supervision-regime moment histories, or centering softmax's common-logit
component. Those mechanisms are not part of this experiment.

The source premise survives the audit. `modal/student_train.py` constructs one
linear classifier weight matrix and bias vector, and passes the ordinary
parameter groups to ScaledAdam with clipping scale 2. The trainer pins icefall
revision `3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`. Its `batched_params`
introduces an extra leading parameter-batch dimension. `scaling_step` computes
RMS and parameter/gradient inner products over every dimension after that
batch dimension. The weight matrix therefore has one complete-matrix scale;
the bias vector has one complete-vector scale. Output classes do not already
have independent scale statistics. The pinned optimizer source SHA-256 is
`7282cd6db1d234e086110feb1f39f076a76095a6b026b939cebc81f2ed0f5d45`.

This establishes an optimizer dependency, not that it caused any target miss.

## Implementation

`modal/student_class_scales.py` represents every class weight row, including
blank, as an individual parameter and each class bias as an individual scalar
parameter. Forward evaluation stacks them into the same affine projection;
there is no extra head, changed inventory, common-logit projection or added
capacity. Folding restores an ordinary linear classifier for export.

The pinned ScaledAdam definitions are used unchanged. All class parameters
remain in **one** optimizer group, preserving the complete-head clipping
procedure, its scale-2 median threshold and its norm-history schedule. There
is no class-by-class clipping. Weight scale statistics become row-local.
Individual biases use ScaledAdam's existing scalar learning-rate multiplier
and scalar clamp, with no RMS statistic; that is an explicit consequence of
the new parameter boundaries. Per-coordinate second-moment estimation and
momentum procedures, coefficients and epsilon are unchanged. Both arms start
with fresh zero moments. Global clipping can still couple rows; the isolated
scale test disables clipping to test the proposed dependency directly.

The tensor-grouped and class-local arms share the complete frozen feature
cache, performed transcripts, recording order, three views per recording,
dropout draws, acoustic objective, optimizer settings and Eden schedule.
Every recording contributes one backward evaluation per arm per epoch,
averaging its views. The horizon is twelve epochs with a fixed seed of 1729
and a persisted one-day preparation/refit deadline. The tensor control's
equal-speaker objective on separate training speakers selects one epoch for
both arms, including epoch zero. Target incidents cannot select checkpoints.
Matched compute means identical gradient-evaluation counts, not identical
wall time or optimizer arithmetic.

The objective is the shipped frame-normalized transcript CTC plus its
configured renormalized top-five teacher cross-entropy, retaining the original
clamped teacher-frame mapping and delay. Teacher targets are generated from
the unmodified recording and reused on its augmented views, as in the shipped
ClipDataset. No auxiliary loss or new gradient weight is added. All ordinary
speech and real pause frames are retained.

`modal/student_class_scales_train.py` enrolls only authorized, verified,
hash-matched performed transcripts and help-train training speakers' scored
none takes with empty error, repeat and unsupported lists. An explicit
release-speaker exclusion list is required. Every speaker appearing in a
non-training split in either source is excluded globally before reading audio.
Audio hashes reject duplicates. Whole speakers are deterministically assigned
to fit or selection, and both partitions require both sources. The unchanged
view and two fixed views use the shipped noise/RIR/gain/SpecAugment chain.
Source optimizer/objective configuration and shipped augmentation banks must
be supplied explicitly, rather than inferred from practice.

The native extraction/export scaffolding is adapted from the retained
`exp/listener-contrast-space-head-training` engine implementation. Its
contrast parameterization, extra preservation loss and four-arm selection
are not used. `modal/student_class_scales_export.py` retains the shipped
encoder and dynamic activation rounding. Dequantizing weights while feeding
unrounded encoder features would change initial probabilities; the driver
rejects that mismatch on every cached view before creating an optimizer.
The two refit arms start with exactly identical affine logits. Their native
source probability tolerance is 2e-6 absolute, 2e-5 relative, accounting for
float32 accumulation; no real model passed this prerequisite here.

Export changes only the ordinary affine phone head. It retains the alphabet,
blank, frontend, sample rate, streaming inputs/caches, windows and 40 ms clock.
All existing insertion-teacher files and external weights must be present,
copied and checksum-verified. Existing manifests are never rewritten; new
manifests describe the exported graphs. Both bundles must pass selection-view
native inference parity before the final training report is written. SDK
decoding, tracker, checker and insertion verification source are unchanged.

## Validation and limits

Fifteen synthetic tests passed locally with Torch 2.8.0, NumPy 2.2.6, ONNX
1.19.0, ONNX Runtime 1.22.1 and the exact pinned optimizer source:

```sh
TILAWA_OPTIMIZER_SOURCE=.autolab/icefall-optim.py \
  .autolab/unit-venv/bin/python -m pytest -q \
  --basetemp=.autolab/unit-pytest modal/test_student_class_scales*.py
```

The tests establish exact classifier and CTC/KD gradient parity, frozen encoder
and teacher inputs, unchanged objective reduction/frame mapping, identical
per-coordinate second moments on an identical gradient, local RMS shapes,
scalar bias semantics, one clipping factor applied to every coordinate,
optimizer writeback and folded-head parity, matched update/dropout/schedule
counts, training-only selection, initial-probability and deadline rejection,
cross-corpus speaker exclusions, duplicate/hash checks and corpus containment.
Generated fp32 and dynamic-int8 ONNX fixtures establish unchanged source
outputs/caches, activation rounding, fitted-head inference, manifest geometry,
complete verifier copying and missing-asset rejection.
The preparation and fit argument parsers also pass. The reused preparation
runner's `--help` collision was corrected to `--help-train` before commit.

Executing the pinned optimizer on synthetic gradients shows that multiplying
unrelated row/bias magnitudes changes the tensor-grouped first class's update,
while class-local updates for that class remain exactly equal through eight
steps, including periodic scale updates. This does not measure acoustic
consonant recovery or warning behavior.

All five supplied-supervisor requests returned
`NotFoundError: No such file or directory`, each with USD 0.00 and no shard
execution logs: the available-data/dependency probe, the required six-shard
core/SDK tests, a second probe using the supervisor's training image, final
training-image synthetic validation, and final six-shard core/SDK validation.
No core/SDK pass or test failure was obtained. Python syntax, CLI help and
staged whitespace checks pass locally. Small local tests supply only the
optimizer file's unused lhotse test-seeding import; optimizer definitions and
their checksum are unchanged. No training was moved onto the Mac.

No training recording, practice audio, target label or scored evidence has been
read remotely. No cache was prepared, head fitted, checkpoint selected, real
bundle exported or target decoded. Practice-score calls: **0 of 3**. Target
wins, component/overall changes, studio regression and clean-session warnings
are unknown. The infrastructure failure does not falsify the premise, so no
killed outcome or fabricated zero-win count is written.

## Remaining experiment work

Restore the supplied supervisor, inspect the supplied model/training inputs,
and run the authorized matched preparation/refit. The runner accepts
`prepare --model --help-train --transcribed --exclusions --augmentation --config
--out`, then `fit --prepared --optimizer --out`. All real outputs belong under
`/scratch/listener-class-local-update-scales/`. The optimizer file must match
the audited checksum.

After both complete bundles pass export parity, compare their training-selected
full-session target decodes, distinguishing missing planted-consonant recovery
from actual confirmed added-word wins. Follow the user's final rule: absent
premise or fewer than three target wins means reverting the engine changes,
recording a killed outcome and committing that result. Three or more wins
requires the supplied practice score before deciding or narrowing; the fold
gate prices collateral losses. Target results never choose a checkpoint.

No `.autolab/model.json` is declared because no real verified bundle exists.
Scoring the main listener would not evaluate the refit. This branch is an
implemented, synthetically validated experiment awaiting compute, not a
trained candidate ready for the fold gate. The card, scorer, lab tooling,
existing manifests, test data and evidence remain untouched. No fold occurred.

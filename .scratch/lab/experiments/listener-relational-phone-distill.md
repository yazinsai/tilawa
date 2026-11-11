# Relational phone distillation

Status: training implementation and 19 numerical unit tests complete; matched pilot, model
export and acoustic evaluation are unmeasured. The supplied execution service
has not produced container output. The experiment card is unchanged.

## Prediction

The card predicts two recovered additions and one collateral wrong call:
listener −1 and overall −1. Its hypothesis is that compression preserves
individual outputs less reliably than the teacher's acoustic relationships
between consonants. The six registered incidents are evaluation targets only.
No practice recording, label, evidence or score has entered the training code
or checkpoint selection. No target-win count has been measured.

## Implemented mechanism

`modal/student_relational.py` computes exact transcript-conditioned CTC
forward-backward occupancy. Each transcript phone has its own occurrence state,
including adjacent identical phones with a mandatory separating blank. Blank
states are excluded. Each occurrence's marginal occupancy is normalized to unit
mass and pools its encoder features over possible boundaries. Student and teacher
use their own valid output grids. Boundary weights and all teacher features are
detached, so encoder gradients change features rather than move boundaries to
reduce the new objective.

For normalized occurrence features `z`, each anchor predicts a distribution
`softmax(cosine(z_anchor, z_candidate) / temperature)`. The loss is teacher-to-student
KL between these distributions. Every candidate from the same original recording
is masked, including other clips cut from that recording. Item keys and audited
parent recording IDs are separate. Anchors require both a same-phone occurrence
and a different-phone occurrence in another recording. All different-phone
candidates participate, including confusable negatives. An empty eligible batch
raises an error rather than falling back to within-recording imitation.

Features are normalized; student and teacher need neither matching coordinates
nor matching dimensions. Each eligible anchor has equal loss weight. The matched
`hint` control uses normalized projected-feature squared error on exactly the
same pooled occurrences and eligible anchors. Its projection is trained only in
the control and is omitted from the deployment checkpoint.

Both arms jointly update the encoder and phone head with transcript CTC,
renormalized top-five teacher cross-entropy using the existing edge-clamped
25 Hz correspondence, and KL preservation against a frozen copy of the initial
student. Teacher features and posteriors are freshly computed on each exact
augmented PCM item; cached clean-audio targets are not substituted. Student
SpecAugment remains the original feature-domain mask. Preservation covers every
valid frame, including blank and real recording pauses; entire recordings are
retained rather than cropped to speech. These examples can limit drift but do
not establish protection against extra emissions.

`modal/student_train.py` exposes the existing phone-head input through
`forward_features()`. Its ordinary forward still uses the same embedding,
encoder, padding mask, permutations and phone head. No architecture, parameter
layout, inventory, inference timing, tracker, checker or insertion verification
is changed. `modal/student_relational_train.py` provides the opt-in direct-call
training entry point for the supplied train image:

```sh
PYTHONPATH=/app/tools:$PYTHONPATH python modal/student_relational_train.py \
  --recipe /scratch/listener-relational-phone-distill/recipe.json \
  --mode relational --out /student/runs/relational
```

The control changes only `--mode hint` and the output directory. The recipe must
be generated after the real corpus and teacher feature tap are audited; no
recipe with invented speakers, recording IDs or hashes has been created.

## Matched training contract

The recipe fixes source items and audio hashes, parent recording IDs, speaker
partitions, initial checkpoint and teacher-state hashes, batch schedule, update
count, learning rate, temperature and positive loss weights. Every frozen
training item must appear in the batch plan. Training and validation are disjoint
subsets of original training speakers; release-excluded speakers are disjoint
from both. Both partitions contain authorized transcribed Quran recordings and
help-train none-labelled takes. Selection requires the original `train` split,
explicit `kind=none`, a scored/non-excluded row, and empty events, non-errors and
unsupported-label lists. The help transcript must match its canonical ayahs.

The loader accepts Quran audio only from supplied `/student` training input,
excluding `/student/eval`, and help audio only from supplied
`/lab/data/help-train`. It verifies source bytes and finite mono 16 kHz PCM.
Student augmentation must match the initial checkpoint's original recipe.
Per-step/item private RNG gives both arms identical audio augmentation and
SpecAugment masks, independent of projection initialization. Deterministic model
passes disable stochastic layers and restore individual module flags; both arms
retain the deployed 16/128 chunk/context geometry.

Checkpoint selection minimizes validation CTC on the frozen training-speaker
partition, using the same schedule and criterion for both arms. The output is
`ckpt.pt` with the validation-selected ordinary student state and original export
configuration. This entry point has no resumable final-step checkpoint; only an
improvement of the training-speaker validation criterion overwrites `ckpt.pt`.
The existing streaming exporter loads this file without a new phone head or
side asset. No checkpoint or export has actually been produced.

These checks detect missing records and obvious partition violations. They do
not prove the producer correctly identified every original Quran recording,
reader alias, release exclusion or real pause. The actual corpus audit, adapter
to the supplied packed Quran corpus, teacher feature-module verification,
teacher/student clock verification and training-only recipe remain outstanding.

## Validation

Read AGENTS.md and `docs/agents/audio-review.md`. Numerical unit validation uses
tiny synthetic tensors only, in an ignored local Python environment. It covers
an independent exhaustive CTC state-path oracle, repeated phones, impossible
alignments, unit-mass pooling, detached boundaries, encoder gradients, padding,
all same-recording exclusions, eligible anchors, rotation/scaling/dimension
invariance, confusable-negative gradients and gradcheck, hint matching, batch
permutation, teacher identity, mode restoration, original forward parity,
top-five acoustic KD parity, help eligibility and frozen batch coverage.

All 19 numerical tests pass with Python 3.13 and torch 2.8.0:

```sh
.scratch/relational-test-env/bin/python -m unittest discover -s modal -p test_student_relational.py -v
```

Compilation and `git diff --check` pass. Core and SDK regression suites were
requested remotely and have not run
successfully. No scorer, lab tooling, supplied manifest, evidence, test data or
experiment card was edited.

Actual Muaalem/Zipformer integration, corpus-schema compatibility, loss balance,
runtime/memory cost, clean-pause behavior and quantized full-session inference
remain unverified. The numerical tests cannot establish ordered consonant
recovery or warning safety.

## Execution service record

All requests use the user-supplied supervisor. CPU run 1 (storage/corpus probe),
CPU run 2 (six shards of `bash tools/autolab_tests.sh`), L4/train run 3
(torch/checkpoint/help inventory), and CPU run 4 (the six-shard regression retry) returned
`NotFoundError: No such file or directory` before any command output. The CPU
requests reported $0 GPU cost. Run 3 reported $0.1487 charged; the budget record
reports $0.15 spent and $99.85 remaining from $100. There are no pending runs.
No alternate Modal access or outside-worktree data search was attempted.

## Result and remaining work

There is no matched training result, target-win measurement, exported listener
bundle, `.autolab/model.json` or practice score. All three practice scoring
opportunities remain. Scoring with the main model would not evaluate this
training change.

An execution-service startup failure cannot establish the acoustic premise or
fire the pilot kill criterion. No killed outcome or fabricated zero-target
measurement is written. The implementation remains an incomplete experiment.

After service restoration, audit the supplied corpus, speaker aliases, original
recording IDs, true help schema and real pauses; freeze the training-only recipe;
run both matched pilots; select within training-speaker partitions; export and
verify complete primary-listener bundles through the unchanged SDK path. Then
compare the six full-session exported streams against the feature-hint control
for newly restored planted consonants in correct order and confirmed additions.

The card's matched pilot requires at least two consonant recoveries. The user's
final rule also requires at least three built-model target wins to proceed:
fewer kills the experiment and requires reverting engine changes and recording
the measured result. Three or more require a practice score before deciding or
narrowing; the fold gate weighs collateral wrong calls and release safeguards.

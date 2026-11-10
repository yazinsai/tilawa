# Whole-recording acoustic group training

Status: implementation draft; blocked before remote execution. No trained model,
exported bundle, target replay, or practice score exists. The acoustic hypothesis
has not been killed or supported. The experiment card is unchanged.

## Prediction

The card predicts two false wrong-word warnings removed while risking one missed
mistake: listener −1 and overall −1. It hypothesizes that average-loss training
sacrifices rare recording-level acoustic regimes. Its four target warnings are
used only for the eventual replay comparison, never to define training groups,
choose examples, fit weights, or select a checkpoint.

The task's final stopping instruction requires at least three target recoveries
before practice scoring, which is stricter than the card's two-warning criterion.
Neither criterion has been evaluated: zero targets have been tested. No gain,
loss, or zero-effect result is claimed.

## Implementation

`modal/student_group_dro.py` runs directly in the supplied training image. It
adds an `inspect` stage, paired `train` stage and `export` stage without changing
Modal orchestration, scoring, benchmark inputs, supplied manifests, decoding,
tracking, or checker decisions.

Training inputs are authorized original `clean-match` training recordings and
help-train's scored, none-labelled training takes. Mistake, repeat, unsupported,
excluded and release-held-out help recordings are rejected. Original recordings
retain the shipped spliced-reference CTC targets and teacher top-five KD; no
scripted mistake is turned into a performed transcript. The original source is
sampled once using a fixed seed and a compute-hours cap; every eligible help
none take is included. There is no fixed learner oversampling ratio. Selected original audio and
teacher arrays are cached once for random group batches; supplied data remains
read-only.

The original corpus provides coarse speaker clusters rather than verified
person identities. Each cluster stays in one training-only partition; clusters
containing excluded release-gold recordings are removed wholesale. Every
help-train speaker also stays in one partition, and any speaker seen with a
nontraining supplied split is rejected. This cannot prove that different
original clusters never contain the same person.

Descriptors use the complete recording, including pauses: median log energy,
energy quantile contrast, relative low-energy fraction, zero-crossing rate,
high/low spectral power ratio, and spectral flatness quantiles. Descriptor
scaling and six acoustic clusters are fitted exclusively on the training
partition. Validation recordings are assigned to those frozen clusters.
Missing groups, transcripts, eligible help takes or augmentation assets fail
explicitly. No verse, incident, session or reader identity controls a group.

Both arms start from the same supplied student checkpoint and use identical
rank-four encoder Linear adapters, with a zero-initialized output factor. The
frontend and phone head remain frozen. Both arms use the shipped room, noise,
gain and SpecAugment chain, identical item draws, identical augmentation seeds,
identical optimization steps and learning rate, and the same training-speaker
validation subset. Dropout is disabled and the inference chunk mask is fixed
for adaptation and its parent-preservation pass.

Each whole recording contributes CTC sequence loss divided by its valid output
length, top-five teacher KD on original data (parent KD on clean help takes),
and a KL preservation term to the parent. Only normalized CTC is adversarially
weighted; KD and preservation keep the empirical group-prevalence weights in
both arms, avoiding a dual weight driven by irreducible teacher entropy. Valid blank/pause frames participate
in KD and preservation; padding does not. No isolated consonant is mined or
upweighted. Group means give recordings equal weight within each regime.

Stratified batches sample uniformly inside every acoustic group. The mean-loss
control weights those group means by their empirical recording prevalence,
recovering the original recording mixture in expectation. The DRO arm starts
with uniform group weights and performs exponentiated-gradient adversarial
ascent on detached current group losses; model descent uses the resulting
weights. This changes the objective rather than assigning a fixed learner-data
ratio. Both arms select their checkpoint by minimum maximum normalized CTC group loss on the
same frozen validation recordings, entirely outside practice.

The selected adapters merge into ordinary Linear weights. Export uses the
existing streaming ONNX wrapper and dynamic MatMul int8 quantization. It checks
inventory, output timing and state geometry against the supplied primary bundle,
runs a streaming-interface smoke check, and copies the unchanged declared
insertion-teacher assets with hash equality checks. No bundle is declared in
`.autolab/model.json` until actual training, export and SDK replay succeed.

## Validation and limits

Local syntax compilation and CLI parsing passed. All 12 focused unit tests
passed in a workspace-local CPU environment (Python 3.13, numpy 2.5.4 and Torch
2.15.0), with no skips. They cover adversarial mass moving toward a difficult
rare group, normalization and offset invariance, fixed empirical mean weights,
identical balanced batch plans, whole-recording pause descriptors, frozen
cluster prediction, strict clean-take eligibility, stable speaker partitioning,
and CLI argument transport. Torch assertions verify zero-initialized parent
equality, frozen parameter boundaries, restoration of parent-only mode even on
exceptions, merge/reload parity, padding exclusion and pause-frame gradients.
The supplied training image uses Torch 2.8.0; testing the real Zipformer under
that version remains necessary. These local checks do not train a model.

Three supervisor requests failed before executing their commands, all with
`NotFoundError: No such file or directory`:

1. Run 1: training-image inventory of original recordings, prior checkpoints,
   help-train, exported models and scratch capacity.
2. Run 2: the supplied six-shard `bash tools/autolab_tests.sh` harness, unchanged.
3. Run 3: the new numerical tests in the training image followed by authorized
   input inspection and training-only group construction.

All three report $0 GPU spend and $120 remaining. These are infrastructure
failures; they do not establish missing training inputs, absent checkpoints,
failed acoustic premises, failed tests, or any target gain. No other route to
Modal or external data was attempted. No practice scoring attempt was made;
all three allowed practice scores remain available.

Consequently the full original-model/adapter integration, data coverage,
augmentation availability, actual checkpoint provenance, int8 parity, complete
bundle integrity, SDK full-session behavior, matched-control target comparison,
clean-session warnings, studio regression and overall attribution are still
unvalidated. The draft must not be treated as a deployable listener or evaluated
using the main model as a substitute for absent assets.

## Resume after the supervisor can execute

First rerun input inventory and the new unit tests in the training image, then
run `inspect` with the fixed defaults to build the training-only partitions and
coverage report. Use the exact shipped student source checkpoint, verifying its
provenance against the installed primary model before training. The paired
`train` command runs both arms with identical defaults and saves both selected
merged checkpoints. Export each with the installed shipped bundle supplied by
`--auxiliary`; keep the teacher graph and external data unchanged.

Then run both exports through the ordinary SDK over all four complete target
sessions and compare final warnings against the matched mean-loss control.
The target sessions select diagnostics only. Verify streaming parity and bundle
loading and run the supplied core/SDK tests. If fewer than three target warnings
disappear, roll back the engine draft, record the measured kill reason, and keep
this write-up. If three or more disappear, declare the complete DRO bundle in
the untracked `.autolab/model.json` and run the required supervisor practice
score before making any decision about losses. The fold gate decides adoption.

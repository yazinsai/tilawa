# Listener duration-stratified distillation

Status: implementation complete; training and acoustic evaluation blocked by the
supplied Modal supervisor. The mechanism is unmeasured, not killed. No trained
model has been declared, and the experiment card is unchanged.

## Prediction and mechanism

The card predicts two recovered additions and one collateral wrong call:
listener −1 and overall −1 on practice. That prediction is below the fold bar.
The hypothesis is that ordinary frame-weighted distillation gives sustained
phones more identity supervision than brief consonant occurrences.

`modal/student_duration.py` computes transcript-conditioned CTC forward-backward
state marginals from the frozen teacher's complete distribution. Each nonblank
state represents a distinct transcript occurrence, including repeated instances
of the same phone. For occupancy `q[t,j]`, occurrence duration is
`d[j] = sum_t q[t,j]`. The experimental measure is `q[t,j] / d[j]`; the control
uses `q[t,j]`. The alignment never depends on either trained head.

The identity objective scores the transcript phone conditional on nonblank.
It therefore has zero direct gradient on the blank logit. A separate, unchanged
teacher blank/nonblank cross-entropy covers all paired real frames in both arms;
ordinary transcript CTC and KL preservation of the shipped output also stay in
both arms, over every real student frame. There is no per-class weighting,
blank-probability adjustment, new vocabulary or warning rule.

Fit-only corpus denominators make identity weight equal across occurrences in
different-length recordings, as well as within a recording. Both arms have the
same aggregate identity budget, scaled by the fit corpus's mean teacher speech
mass. Each recording has one original and two fixed views of the shipped
augmentation chain. Both arms use those exact views, zero-initialized affine
head corrections, deterministic recording order, SGD, learning rate 0.02 and
12 epochs. These settings were fixed before acoustic evaluation.

The trainer caches the installed encoder's native pre-quantization features and
posteriors without changing any encoder operators or weights. The export adds
the affine correction to shipped log probabilities before normalization. The
installed quantized projection remains intact. Exposure verifies unchanged
original outputs and recurrent states; export checks prediction parity on
selection recordings, including an optimized ONNX runtime. The primary clock,
inventory, SDK decoder, tracker and checker are inherited. Existing insertion
teacher files and external weights are required, copied and checksum-verified.

## Enrollment and selection

Preparation requires both authorized, verified performed-transcript recordings
and help-train none-labelled takes. Additional transcripts must identify a
training speaker, audio path, authorization, verification and audio SHA-256.
None takes must explicitly be training, none, scored and free of events,
non-errors and unsupported labels. Scripted mistakes and repeats are excluded.
Explicit exclusions and speakers appearing in any non-training split in either
corpus are excluded before enrollment. Audio identity duplicates and paths
escaping an enrolled corpus fail preparation.

Speakers are partitioned by a stable hash into fit and selection groups within
the training corpus. Both groups must contain both enrolled corpora. Selection
uses speaker-balanced transcript CTC plus shipped-output preservation. A shared
epoch is selected from the mean of the two arms' selection objectives, including
epoch zero. Both exported checkpoints consequently have identical update counts.
Practice incidents and release-held-out speakers have no selection role.

## Validation and limits

The unit suite covers exhaustive emitted-path CTC marginals, repeated phones,
a one-frame versus five-frame occurrence, probability conservation on low-mass
long alignments, impossible transcripts, equal corpus identity budgets, unchanged
blank/CTC/preservation values and gradients, detached evidence and real-frame
tails. Enrollment and speaker partition tests cover held-out exclusion,
authorization, transcript/audio identity and duplicate audio. A synthetic matched
fit exercises shared checkpoint selection and complete exports. Export tests
cover state/clock preservation, zero-correction parity, quantized feature
extraction, optimized-runtime parity and missing external teacher weights.

Local validation: **25 tests passed** with
`.scratch/duration-venv/bin/python -m pytest -q --basetemp=.scratch/duration-pytest modal/test_student_duration.py`.
The environment used Python 3.13.7, torch 2.8.0, NumPy 2.5.4, ONNX 1.23.2,
ONNX Runtime 1.31.1 and pytest 9.2.0, on CPU. The synthetic fit selected a
nonzero shared epoch and verified both complete trained-head exports. This test
also caught and fixed a cache-loader bug that discarded filenames needed by the
export parity pass. Python compilation and git whitespace checks passed.

The first Modal requests, volume/compute inspection (run 1) and the required
six-shard core/SDK suites (run 2), both failed before execution with
`NotFoundError: No such file or directory`; their result files report zero
compute spend. A retry requested the same core/SDK suites (run 3) and this
experiment's Python unit tests (run 4); both retries failed before execution with
the same error and zero reported spend. The required core/SDK suites therefore
remain unrun. No workaround or alternate Modal access was attempted.

Consequently the actual enrollment schema, authorized recording availability,
augmentation banks, complete shipped bundle and teacher/student edge clocks
have not been checked on real data. Synthetic parity establishes the adapter's
behavior, not parity on the shipped graph or preservation of ordinary readings.
No training cache, matched trained heads or complete real model bundle exists.
`.autolab/model.json` is absent so an incomplete export cannot be selected.
There have been zero practice-score requests. Target recoveries and overall,
listener, studio and clean-session changes are unknown.

## Remaining acoustic experiment

Once the supplied supervisor runs, invoke `student_duration_train.py prepare`
with the complete pinned source bundle, enrolled performed-transcript JSONL,
help-train labels, excluded-speaker list, shipped noise/RIR banks and a fresh
scratch destination. Invoke `fit` with that prepared directory and a separate
output directory under `/scratch/listener-duration-stratified-distill/`.

Compare both verified exports using the ordinary SDK full-session streaming
decode. Target keys are diagnostic selectors only: p07 37:36:1 at 144.12 s,
p48 12:76:1 at 39.98 s, p56 32:21:1 at 106.76 s, and p57 51:18:1 at 95.26 s
and 51:20:1 at 112.26 s. Report regained previously missing planted consonants
relative to the control, and actual warning recoveries at the incident keys;
these are different measurements. Oracle crop recovery is insufficient.

Fewer than two consonant recoveries relative to the matched control fails the
card's premise criterion. The user's final instructions also require rollback
if fewer than three targets are won. If three or more targets are won, declare
only the complete verified occurrence bundle and request practice scoring before
deciding; the fold gate decides whether gains justify collateral losses. An
infrastructure error establishes neither criterion, so no killed outcome has
been written.

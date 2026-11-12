# Variance-reduced frozen phone-head refit

Status: implementation and synthetic validation complete; acoustic work blocked
before execution. No replacement model is declared. The card is unchanged.

## Prediction

Listener −1 wrong call and overall −1, with one recovered added-word mistake
and no predicted losses. That forecast is within the stated noise band and
below the fold floor. The premise is speculative: ordinary stochastic updates
may reverse useful weak-consonant improvements on frozen acoustic features.

## Mechanism

`modal/student_variance.py` implements a finite-data SVRG continuation of the
existing affine phone head. For reference parameters r, a full pass computes
mu = mean_i grad L_i(r). A uniformly sampled recording i then supplies
grad L_i(w) − grad L_i(r) + mu. Averaging that direction over recordings gives
the ordinary objective's gradient at w. Neither consonants nor particular
recordings receive extra weight. Both gradient terms consume the same cached
features, targets, teacher distributions and dropout mask. No encoder gradient
or random augmentation is generated during refitting.

Both arms use frame-normalized transcript CTC, full-distribution teacher
cross-entropy and shipped-output KL preservation. Views average within each
recording; recordings have equal sampling weight. CTC retains the shipped
trainer's valid-frame units, rather than normalizing by transcript length.
KD and preservation include real pause frames. This is a bounded head
continuation recipe, not a reproduction of the original ScaledAdam training
run. The controlled comparison keeps its objective, data and weights identical
and changes only the stochastic gradient estimator. Both arms use plain SGD,
learning rate 0.02, without momentum or arm-specific clipping.

A full reference pass costs N complete recording gradients; each paired
update costs two. Refresh candidates are N and 4N updates. Their cycle costs
are 3N and 9N, respectively. Four shared 9N compute blocks give each arm 36N
gradient evaluations: ordinary SGD takes 36N updates, the N-refresh arm 12N,
and the 4N-refresh arm 16N. Thus the ordinary control uses its additional
compute for real optimizer updates. Gradient evaluation includes all three
views of a recording. Fixed encoder/teacher cache preparation is shared.

Refresh frequency and checkpoint are selected by the minimum pair-mean
objective at equal compute, on held-aside training speakers only. Epoch zero
is eligible. Validation averages recordings within speakers and then speakers
equally. Practice labels, target margins and scores cannot enter this path.

## Training cache and native export

`modal/student_variance_train.py prepare` admits help-train training-speaker
`none` takes with empty events, non-errors and unsupported labels. Canonical
ayah phones supply their transcripts. Optional additional performed transcripts
must be explicitly enrolled with speaker, training split, authorization,
verification, audio path, phones and audio SHA-256. Scripted mistakes and
repeats are not silently converted into canonical transcripts. Any speaker in
a non-training split of either corpus, or in the explicit exclusion list, is
excluded globally. Audio hashes reject duplicate enrollment. Deterministic
speaker grouping keeps entire speakers in fit or selection partitions.

Preparation uses the source's frozen streaming ONNX encoder, one original
view and two fixed views from the shipped waveform/SpecAugment family. Noise
and RIR banks must be supplied; it does not substitute practice recordings or
invent replacement augmentation banks. Each view also caches a seeded 0.1
head-dropout mask with the original expected activation scale. Masks apply
only to training features; selection and inference use undropped features.
Cache hashes and execution semantics bind the source model, teacher, views,
alphabet, clock and masks.

The native feature/export helper is adapted from
`exp/listener-curvature-aware-head-refit`. Its projection extraction and bundle
contracts are retained. Both arms start from the shipped affine weights and
bias, dequantizing the projection weights where necessary. The encoder stays
quantized and frozen. Both exported heads use float32 projection, so activation
rounding removal is a shared intervention and cannot establish an SVRG gain.
Recovery shared with the ordinary control is not evidence for variance
reduction.

`modal/student_variance_export.py` replaces only that projection and prunes its
unused nodes. It inherits vocabulary, frontend, window, shift, state outputs
and 40 ms clock. SDK decoding, tracker, checker and insertion verification are
unchanged. All teacher ONNX/external-weight members must be present, copied
and checksum-verified. Source outputs/states must be exactly preserved when
exposing features. Each fitted export must reproduce its head probabilities
on every selection view within 2e-4; the final report is written only after
both complete bundles pass. Existing source manifests are never edited.

## Premise audit and execution sequence

Run `prepare --model --help --exclusions --augmentation --out`, optionally
`--transcribed`, through the supplied supervisor. All new caches and bundles
belong under `/scratch/listener-variance-reduced-phone-refit`.

Then run `fit --prepared --out --ordinary-only`. This saves ordinary checkpoints
every N gradient evaluations, before any variance arm is needed. The later
full comparison reproduces that control exactly from the same cache and seed.
Preparation has a one-hour cap, the premise continuation five hours, and the
full comparison seventeen hours, leaving one hour for export within the
one-day experiment budget. Supervisor invocations still obey its shorter run
timeout; these caps are ceilings, not requests for longer jobs.

The separate `audit --checkpoints --diagnostics --out` reads diagnostic NPZ
features, original shipped rows, teacher rows and annotated planted consonant
spans. Diagnostic JSON supplies source SHA-256 and distinct target IDs with
cache paths and `[startFrame, exclusiveEndFrame, phoneId]` occurrences. Those
inputs never reach fitting or selection. No target/session/verse identifiers
are embedded in the implementation.

For a consonant absent from the shipped argmax throughout its annotated span,
the audit picks one fixed teacher-supported frame, using the teacher's largest
correct-versus-competitor margin. On that same frame, it measures correct
logit minus the strongest other class at each ordinary continuation checkpoint.
Initialization is excluded from the crossing test. Both strictly positive and
strictly negative margins must occur. Multiple crossing consonants in one
target count as one target. Fewer than two crossing targets triggers the
card's premise kill. Teacher support is correlated acoustic evidence, not an
independent listening judgment; supplied labels and alignments remain essential.

If the premise passes, run the full `fit` and verify both bundles. Only then
declare the selected variance bundle in ignored `.autolab/model.json` and
measure full-session target recovery. The user's separate floor requires three
target wins before practice scoring. At three or more, run the supplied
practice score before deciding; the fold gate owns collateral-loss decisions.
An acoustic kill requires reverting engine changes and recording its actual
measurement, not interpreting infrastructure failure as zero wins.

## Validation and result

All 21 synthetic tests pass without dependency skips:

```
.scratch/variance-venv/bin/python -m unittest discover -s modal -p test_student_variance.py -v
```

They verify the full finite objective gradient with unequal recording lengths,
the paired estimator's exact expectation, zero estimator variance at the
reference, frozen feature/teacher/preservation inputs, matched gradient compute,
selection-only checkpoint choice, identical staged ordinary continuation,
impossible alignment rejection, fixed-frame premise crossings and target-level
aggregation. Enrollment tests cover global heldout exclusion, repeat/ambiguous
none rejection, optional additional corpus and path containment. Native ONNX
tests cover float/int8 head extraction, encoder/state preservation, projection
replacement and complete unchanged teacher bundles. These are tiny generated
examples; no acoustic training or heavy model inference ran on the Mac.

The local test environment uses Torch 2.15.0, ONNX 1.23.2, ONNX Runtime 1.31.1
and NumPy 2.5.4. Python compilation and whitespace checks pass.

Three supervisor requests all failed before command execution with
`NotFoundError: No such file or directory`, each reporting USD 0.00:

1. Run 1: inspect supplied data roots and scratch availability.
2. Run 2: required six-shard core/SDK suites, `bash tools/autolab_tests.sh`.
3. Run 3: retry supplied-data/model/dependency inspection and synthetic tests.

No remote audio, labels or source model were inspected. No real cache,
ordinary continuation, margin audit, SVRG fit or model export was produced.
Core/SDK suites were requested but did not execute. There were zero practice
score calls. Target wins, losses and component/overall change are unknown.

This service failure does not falsify the premise. The engine implementation
remains for resumed validation; no killed outcome or replacement model is
declared. Scoring the main listener would not evaluate this refit. Real bundle
compatibility, data/augmentation availability, transfer and recognition safety
remain unverified until the supplied supervisor can execute.

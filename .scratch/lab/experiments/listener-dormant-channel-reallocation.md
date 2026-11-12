# Dormant upper-channel reallocation

Status: blocked by the supplied supervisor execution route. No engine change,
trained model or target result is claimed. The experiment card is unchanged.

## Prediction and mechanism

Experiment `listener-dormant-channel-reallocation`, listener, explore. Predict
one recovered added word, listener −1 and overall −1, with substantial zero-gain
risk. The proposed explanation is insufficient usable capacity for weak
consonants within the existing model, despite redundant upper-block channels.

Audit channel activity and outgoing contributions on authorized training audio.
Recycle only persistently inactive upper-block channels: initialize incoming
weights from poorly represented acoustic examples and zero outgoing weights.
Freeze the acoustic stem and compare a bounded upper-block refit with ordinary
continuation and an equally sized random-channel recycling arm. Recordings,
objectives, preservation examples and compute must match across all three arms.
Width, alphabet, posterior clock, tracker, checker and insertion verification
must remain unchanged.

Use authorized Quran transcripts and help-train training speakers' none-labelled
takes. Scripted mistake labels are not verified performed transcripts. Exclude
release-held-out speakers and form speaker-grouped selection partitions within
training speakers; channels and checkpoints must not be selected using practice
targets. Include ordinary reading and real pauses in fitting and preservation.

The card requires a previously missing planted consonant in full-session
exported decoding relative to both matched controls. The user's final stopping
rule additionally requires at least three of the five registered targets to win
before practice scoring. Neither comparison has been performed.

## Source audit

The worktree's `modal/student_train.py` pins icefall revision
`3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1` and builds a causal Zipformer2.
The documented shipped `spliced-clean` recipe uses the full-size student,
spliced CTC targets, top-five KD weight 2 and the existing noise, RIR, gain and
SpecAugment chain. Existing `load_clips` excludes gold PER training clips but
does not itself create the training-speaker checkpoint-selection partitions
required by this experiment. `tools/student_manifest.py` preserves speaker
cluster metadata; `tools/help_import.py` preserves speaker, split, kind and
exclusion metadata. Their actual remote contents remain unverified.

The exact pinned `zipformer.py` has SHA-256
`4d5365ed19dee07f5a51fcaaa26e18310f7fafbdf2a391e24a7cc01d1b2e8cfe`.
Its `FeedforwardModule` at line 2018 has an incoming linear projection, a hidden
balancer, SwooshL activation plus dropout and an outgoing projection, followed
by output whitening. This internal feed-forward expansion is a plausible
recycling site: unlike residual-state dimensions, its columns can be edited
without changing the encoder width, acoustic caches or normalization geometry.
That structural suitability does not establish that any channels are dormant.

The pinned `scaling.py` has SHA-256
`d5a5db141cd00b4beaa990fe6827599b3248b2fddd87e8dc7809b0dbd9d6633b`.
SwooshL is

```text
log(1 + exp(x − 4)) − 0.08*x − 0.035
```

It has no ReLU-style zero region. The hidden balancer requests at least 0.3
positive preactivations and absolute magnitude of at least 0.75, through
training-gradient adjustments; these are not guarantees about trained channel
activity. `ActivationDropoutAndLinear` performs the activation before the
outgoing linear operation. A negative-preactivation census would therefore
misclassify some contributing channels as inactive.

A small standard-library scalar check of the exact activation expression passed:
preactivations −6, −2 and −1 give outputs approximately 0.4450, 0.1275 and
0.0517. The source and this diagnostic are retained in ignored
`.autolab/channel-audit/`, including `source-contract.json`. This check loads no
model or audio and cannot establish channel dormancy, recoveries or acoustic
parity.

For channel j, its actual outgoing contribution is `W[:, j] * activation_j`.
Zeroing outgoing columns removes the sum of their existing contributions.
Initializing their incoming weights does not cancel that removal. Consequently,
the initial-function-preservation claim needs contribution measurements and a
before/after probability and decode parity check on independent training-speaker
audio, including pauses. Near-zero activity alone does not prove exact function
preservation. The smooth activation is a reason to perform this audit, not
evidence that the dormant-capacity hypothesis is false.

## Execution and validation

Supervisor run 1 requested a default-image dependency, storage and authorized
data/checkpoint inventory. Run 2 requested the supplied six-shard
`bash tools/autolab_tests.sh` harness. Both returned
`NotFoundError: No such file or directory` on 2026-11-12, without command output
or downloaded shard logs. Both result files report $0 charged. No dependency,
checkpoint, audio inventory or core/SDK test result can be inferred from these
failures.

Run 3 requested the supported training image and its dependencies and training
paths. Run 4 requested only `pwd` in the default image, to distinguish general
execution failure from the inventory command. Both also returned the same
`NotFoundError` without command output or downloaded shard logs. Their result
files report $0 charged and no GPU allocation. All four requests failed on
2026-11-12; no successful remote execution is available to this experiment.
The error does not identify the missing supervisor resource, so no specific
image, volume or filesystem root cause is asserted.

Local package discovery finds no Torch, torchaudio, ONNX, ONNX Runtime or pytest.
There are no workspace, core or SDK node_modules. Heavy acoustic work must run
remotely; alternative Modal access and training files outside this worktree are
outside the user's authorization. A shell-only failure confirms that changing
the calibration script cannot resolve the available execution blocker.

No calibration, incoming-weight initialization, matched-arm refit, checkpoint
selection, export, SDK parity check, full-session target comparison or practice
score has run. Practice-score requests used: **0 of 3**. Target wins and overall
score change are unknown, not measured zero. No engine behavior changed, so no
replacement behavior tests were added. The required existing test harness was
requested but failed before producing any test result.

## Remaining work and disposition

Completion requires the supplied supervisor to execute the training image with
the shipped checkpoint and authorized training recordings. Then collect
padding-excluded, per-recording activation and contribution statistics; choose
persistently inactive channels using training-speaker partitions; initialize
them from poorly represented transcribed acoustic examples; and verify initial
function preservation. Train and select all three matched arms within the
one-day CPU budget, export complete bundles with the existing streaming geometry
and insertion-teacher assets, and compare the five targets in full-session SDK
decoding only after training-only selection is frozen.

The implementation remains incomplete. Actual checkpoint weights and authorized
recordings are needed to establish which channels contribute negligibly,
whether recycling preserves the initial function and which acoustic examples
should initialize them. A source-only patch cannot supply those measurements
or the matched continuation/random controls. No engine edits were made, so
there is no unvalidated training utility to revert.

No replacement model or `.autolab/model.json` is declared. An unused recycling
helper or an untrained checkpoint would not implement the requested experiment,
and decoding with the main model would not evaluate channel recycling. No killed
outcome is justified: unavailable execution has not disproved the premise or
measured fewer than three wins. No `.autolab/outcome.json` was written. Only this
experiment record is committed. Local `git diff --check` passed and the card
matches its original bytes. No scorer, lab tooling, supplied manifest, test data
or evidence has been edited. No fold has been performed.

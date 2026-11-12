# Sparse-factor acoustic readout

Status: implementation prepared; training and validation blocked before execution.
This is not a measured experiment result or a scientific kill. The experiment
card is unchanged.

## Prediction and premise

Predict two false warnings removed and one caught substitution lost: listener
−1, overall −1. The proposed defect is overlap between independently useful
acoustic factors in the frozen encoder's hidden directions. An overcomplete
sparse representation might separate weak consonant evidence from stronger
vowel or nasal activity.

The registered targets are the wrong-word false warnings at p11 / 26:106:1,
p36 / 55:48:1, p55 / 26:174:8, and realuser-v1-sob-049 / 92:13:1.
These references describe the experiment only; the runtime and trainer contain
no target, session or reader selectors. The user's stopping rule requires at
least three target wins before practice scoring, even though the card's original
kill criterion requires two. Neither criterion can be assessed without trained
readouts and real warning-window comparisons.

## Implementation

`modal/student_sparse_factor_export.py` exposes the effective deployed CTC
projection input in a copy of the shipped streaming ONNX. For dynamically
quantized projections this is the dequantized activation-code input, including
the actual scale and zero point. It retains the full feature width, rather than
selecting primary logits or a projection subspace. The exporter checks all
primary outputs and recurrent caches for exact equality over eight streaming
windows and verifies the extracted feature-to-logit projection numerically.
Those checks are implemented but have not executed against the shipped model.

The SDK training utility caches eligible help-train training-speaker features.
Any speaker with a non-training label is excluded globally; an additional
exclusion list can be supplied. Release-held-out speakers are absent from the
authorized help-train mount. Edited and repeat takes supply unlabelled
reconstruction observations only. Canonical CTC supervision is restricted to
usable none-labelled takes without events or non-errors. Scripted mistakes are
never treated as performed transcripts.

For feature width D, sparse and dense autoencoders both have a ReLU D → 2D
encoder and linear 2D → D reconstruction dictionary. They receive identical
initialization, recordings, frame orders, reconstruction losses and updates.
Only the sparse arm receives an activation L1 penalty. Dictionary columns are
normalized after each update so decoder scaling cannot evade the penalty.
Both encoders and dictionaries are frozen before fitting their CTC readouts.
The ordinary nonlinear control has the same D → 2D → vocabulary inference
capacity and trains directly with CTC.

The fixed training schedule is eight reconstruction epochs and thirty CTC
epochs, with 512-frame and four-recording batches. The activation penalties
0.01 and 0.1 are selected using sparse-arm CTC loss in three whole-speaker folds,
split before normalization or dictionary fitting. Every arm shares each trial's
recordings, initialization and update schedule. The selected recipe is then
refitted on all eligible training speakers. No practice input participates in
selection. Extraction uses a single inference thread with a two-hour limit;
training has a 7200-process-CPU-second cap. The combined planned work is bounded
below one CPU day. Reports retain reconstruction error, activity fractions,
matched capacity, frozen-encoder equality, speaker partitions, dictionaries and
Python readout parity probes.

`packages/sdk/src/sparse-factor.ts` discovers the actual complete written and
already-accused sequences and frame bounds from otherwise supported string-path
wrong-word confirmations. Each arm scores the same native feature rows and
complete pronunciation, marginalizing all legal CTC phone-boundary paths. A
completion-conditioned shared timing prior avoids a path-count advantage for
one pronunciation. Equal consecutive labels require blank separation; incomplete
or unsupported sequences abstain.

A certificate is issued only when the sparse arm strictly prefers the written
pronunciation and both matched controls strictly prefer the rival. Ties,
nonfinite scores, unavailable frames and out-of-inventory pronunciations
abstain. The certificate carries all six scores and the exact strings, word and
frame bounds through ordinary token metadata. The checker changes only the
public confirmation, preserving its internal ownership and candidate ledger.
Graph confirmations are exempt. Primary phonemes, confidence, posteriors,
tracker input and insertion decisions are preserved by construction and have
dedicated tests, but their actual model-backed parity remains unverified.

The normal Node loader validates descriptor sizes and SHA-256 values, all
readout dimensions, and the exact primary model hash before attaching the side
extractor. Missing or corrupt declared files fail loading. Packaging copies the
primary model, manifest and existing teacher assets unchanged, then declares
both side files. A separate SDK verification mode compares Python and SDK
readout log probabilities. No bundle has been published or declared here.

Audio sessions currently retain audio and perform the auxiliary review at
`end()`. A live warning can therefore precede its later downgrade. This buffers
session audio and adds a full-session encoder replay; streaming latency and
device memory budgets are not established.

## Validation and block

Local Python compilation passes for the trainer, exporter and tests. The
stdlib-only whole-speaker isolation test passes. The broader local data test
could not import NumPy because its installed native extension has an
incompatible architecture; Torch and ONNX are unavailable locally. No heavy
computation was run on the Mac.

Added tests cover complete-path marginalization against exhaustive enumeration,
repeated labels, both-control preference requirements, malformed and incomplete
evidence, primary-token JSON transport, exact occurrence matching, graph
exemptions, session downgrades, insertion/position preservation, declared-asset
failure, data isolation, equal inference capacity, fixed dictionaries and CPU
budget interruption. These numerical and runtime tests have not passed yet.

The authorized supervisor was invoked four times:

1. Five-minute filesystem and dependency probe.
2. Ten-minute filesystem and dependency retry.
3. The supplied six-shard `bash tools/autolab_tests.sh` build/test harness.
4. The complete sixty-minute Python-test, build, feature-export, help-train-cache,
   grouped-training, bundle-packaging and SDK-parity pipeline.

Every request returned `NotFoundError: No such file or directory` before any
command output, with reported compute spend zero. No container logs, cached
features, weights, target comparisons or test results were returned. The
supervisor environment needs repair; changing acoustic code cannot resolve this
pre-execution failure. The user was asked asynchronously to restore it.

No practice scores were used. Target wins, losses, component and overall changes
are unknown. The premise has neither been verified nor falsified, so no killed
outcome is written. No `.autolab/model.json` is declared: scoring the main model
without the side bundle would not evaluate this implementation. The source is
retained for continuation after infrastructure repair; this branch is not ready
for acoustic evaluation or folding.

## Continuation

Rerun the completed pipeline recorded in ignored `.autolab/modal-request-4.json`
after the supervisor is repaired, then run the current full tests. Verify the
normal SDK loader and Python/SDK parity before selecting the bundle. Inspect
all four targets with the generic SDK utility's `--audio` and `--evidence` mode;
its candidate windows come from the engine, never labels. Compare complete
before/after warnings and verify unchanged tracker and insertion events.

If fewer than three targets are won, revert the implementation, record the
measured premise failure and commit the requested killed outcome. If three or
more are won, declare the verified scratch bundle in ignored model selection
and run the authorized practice score before deciding. The fold gate, not this
unexecuted forecast, decides population safety and benefit. No fold was made.

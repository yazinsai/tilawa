# Chunk-phase observation channel

Status: calibration and required validation blocked before execution; prototype retained, not activated.

## Prediction

Card `tracker-chunk-phase-observation`, tracker/explore, practice `lab-v1.5u`, listener
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`: tracker −1, overall 0. The saved audits
reported the near continuation 0.3915 edits behind the rhyme restart. A better binding can
expose the documented nearby listener false warning; no standalone fold is predicted.

Target: `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.

The card's premise test is whether a training-frozen phase correction reverses the two
paths' ordering over their identical retained observation interval. That test has not run.

## Implemented prototype

The SDK listener attaches native inference coordinates to each emitted token: original
output frame, inference-call start frame, full graph output-frame count and model hash.
The output-frame count comes from the actual acoustic graph output, including when the
final inference call is only partially drained. This metadata leaves the greedy token,
confidence, timestamps, carried-run behavior and posterior rows unchanged.

`trackerInput` derives phase as `(frame − chunkStart) / chunkFrames`. Original inference
coordinates survive token-preserving serialization, cropping and evidence rechunking;
the evidence push size and posterior slice boundaries are never used. Invalid or absent
geometry supplies no phase. A frozen channel can bind its calibration to a model hash
and inference advance; another graph or advance contributes no phase correction.

`phase-channel.ts` fits normalized distributions of emitted identity conditional on
reference identity and phase, including a separate insertion row. Sparse rows back off
through reference-phone manner classes and each reference's phase-pooled distribution.
The local matcher adds the conditioned-versus-pooled contribution:

`−(10 / 3) log[P(emitted phone | reference phone, phase) / P(emitted phone | reference phone)]`.

Ten units and three nats remain one edit, using the tracker's existing units. The same
observation correction enters continuation and restart diagonals, and insertions have
their corresponding observation correction. Reference-only deletion costs, explicit
duration charges, movement edges and penalties, commitment margins, search horizon,
event handling and checker rules are retained. A one-bin channel is an exact pooled
control with zero phase contribution. No numerical default was invented or tuned.

The channel is currently optional through `TrackerOptions.phaseChannel`. No fitted
channel was returned, so none is installed as the product default. Consequently,
practice scoring this prototype would not test the registered hypothesis.

## Paired training and selection

`packages/sdk/src/training/phase-channel-fit.ts` reads only help-train labels/audio,
the reference asset and the supplied primary model. It selects scored, non-excluded
`none` recordings with no labelled edits or repeats, from training speakers only.
Any speaker appearing in a non-training split is excluded entirely. The runner's data
contract excludes release-held-out speakers; no release data was requested or read.

Each recording is resampled once and its identical audio is decoded at every legal
integer native-frame origin within one inference advance. A common silence prelude
fills the model's declared left context, with the origin offset added to that prelude.
Geometry comes from the manifest's inference advance, fbank stride, output stride and
encoder context, and is checked against the actual graph output. Audio and model hashes
key the decode cache. These paired decodes change computational placement while keeping
speaker and pronunciation fixed; they do not change the deployed listener decode.

Transcript-constrained latent edit alignments supply fractional counts. Reference
duration expansions are collapsed for identity supervision, while product duration
costs are retained. Each recording's views have total weight one. Pairing does not
establish gold sound boundaries or eliminate uncertainty in reference-to-token
alignment; the estimator marginalizes that uncertainty rather than claiming a sound
transcription verified by listening.

Up to five speaker-grouped folds, entirely within training, select 1, 2, 4 or 8 phase
bins and pooling masses 16, 64 or 256 by held-partition predictive log likelihood.
All origins of one recording stay in the same speaker fold. The one-bin candidate
tests whether training supports any phase dependence. The selected configuration is
refitted on all eligible recordings. The output includes the fitted table, model
geometry, selected-label hash, speaker folds and every selection result. No practice
incident, reader, word or verse identity is a model feature or selection criterion.

## Validation actually completed

Nine focused test bodies passed locally using Node 22's TypeScript transformation
and a temporary assertion adapter for Vitest:

- Native phase survives merging, splitting and cropping without mutating the source.
- Missing or invalid geometry, stale derived phase and mismatched model geometry are ignored.
- Emitted-identity rows normalize and a paired synthetic channel learns opposite phase preferences.
- The pooled control, unknown phones and missing phase contribute zero.
- Duplicating paired views does not overweight a recording.
- Unequal decoded lengths and indels remain representable through latent alignments.
- Local channel plumbing and save/restore preserve the expected state and cost.
- Deterministic SDK graph outputs carry identical coordinates under small and whole-clip pushes.
- A different graph advance and partially drained final call retain their true inference geometry.

The temporary loader and assertion adapter are removed from the final branch. These
are focused numerical and plumbing checks, **not** a TypeScript build or passing
required core/SDK Vitest suites. No actual acoustic model or training recording ran
locally. `git diff --check` passed.

## Supervisor requests and limits

Requests through the supplied runner:

1. Training/model availability inspection: `NotFoundError: No such file or directory` before execution.
2. Six-shard `bash tools/autolab_tests.sh`: the same pre-execution failure.
3. Core/SDK build and one-shard, 60-minute paired calibration: the same pre-execution failure.
4. Six-shard build/tests after the focused test additions: the same pre-execution failure.

No training availability listing, model manifest, paired decode, calibrated table,
selection result, build log or required suite result was returned. The error does not
identify a particular missing resource or establish that storage is exhausted.

Target ordering, target warnings won and overall change are unknown. Zero practice
scores were requested (0 of 3 used). No engine improvement, premise kill or fold-gate
verdict is claimed. Infrastructure failure alone is not a measured zero-target result.
The recorded outcome is `needs-human`, with no measured count. Engine code and unit
tests remain on this branch as an unactivated prototype; calibration and target
validation must resume before an experiment verdict can be recorded.

## Resume after the runner is restored

Run the authorized calibration and required suites:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && PHASE_CACHE=/scratch/tracker-chunk-phase-observation/paired-decodes pnpm --filter @tilawa/sdk exec tsx src/training/phase-channel-fit.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- \
  'bash tools/autolab_tests.sh'
```

Freeze the returned training-selected model and install its default only after the
build and tests pass. For archived evidence lacking native metadata, verify its
original zero-based acoustic clock against the supplying model's inference geometry
before adding an engine-side reconstruction contract; never infer phase from lab
chunks or edit saved evidence/scorer code. Compare both target paths over the same
retained interval, then replay the unchanged checker to count actual target wins.
If the premise fails or the built change wins fewer than three targets, follow the
user's kill rule: revert engine code/tests and commit the measured killed outcome.
If at least three targets are won, obtain a practice score before narrowing or
making any gate decision. The supervisor's fold gate remains responsible for folding.

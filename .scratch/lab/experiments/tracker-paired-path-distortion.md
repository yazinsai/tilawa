# Paired path-score distortion

Status: needs-human — supervisor Modal infrastructure failed before executing any worker command. The untrained prototype and its unit tests were reverted. This is not a premise kill.

## Prediction, before training or target replay

Tracker −1, overall 0 on practice phone. A better continuation binding might remove the false skipped-passage warning at `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84` while exposing the documented nearby listener false warning. The card's stated disadvantage of 0.3915 edits is prior evidence, not a measurement from this worktree.

## Mechanism and prototype

The proposed observation model learns relative changes in complete candidate path scores between original training recordings and phone-degraded copies. The unchanged tracker must run on both; their candidate union must be evaluated over corresponding complete heard intervals. Speaker-grouped model selection must use only authorized help-train speakers, include unchanged-input controls, and finish before any target evaluation. Neither scripted mistakes nor their omissions should be treated as degradation labels: they occur in both members of a pair.

An observation-only prototype retained local paths as persistent reference segments, preserving each jump's source endpoint and existing movement charge. It tracked complete-prefix edit composition and recent observed-token assignments without altering the underlying recurrence. Candidate features combined edit composition with group pronunciation overlap and shared-token usage. A joint Gaussian clean-score distribution was designed with shared coefficient uncertainty and residual covariance from common token assignments; this is a distribution over competing scores, rather than independent fitted phone costs or a destination classifier.

A proposed replacement candidate had to satisfy `inferred advantage − quantile × standard deviation >= commitMargin(n)`. Its advantage variance included the covariance of the competing scores, rather than summing two independent error variances. The available movement edges, checker behavior and commitment margin would remain unchanged. No trained coefficients, covariance, quantile or runtime model were produced, and no model was connected to tracker.ts.

## Validation attempted

Three commands used the authorized supervisor, with at most two requests outstanding:

1. Modal run 1: inspect `/scratch` capacity, list authorized help-train files, inspect training label structure and list the deployed model files.
2. Modal run 2: a minimal `pwd` command, with a five-minute worker timeout.
3. Modal run 3: `bash tools/autolab_tests.sh`, six shards, with a five-minute worker timeout.

Every request failed with exactly `NotFoundError: No such file or directory`. The supervisor returned no worker log or shard output. Even the minimal command failed, so the failure does not establish a defect in the prototype, training data, tests or disk capacity. In particular, the earlier experiments' storage failures were not reproduced or diagnosed here.

Lightweight local synthetic checks of the prototype passed: observation hooks preserved local positions on a short repeated-phone stream; complete-path evaluation charged unmatched prefix tokens and retained a jump's two-edit movement cost; and a synthetic joint model changed a candidate only when its inferred advantage cleared the requested margin. These checks used Node's built-in TypeScript stripping and assertions. They did not execute the full core/sdk suites, fit a model or replay any practice session. Unit tests covering snapshot restoration, shared-token covariance and the margin were drafted but could not be validated remotely; they were removed with the prototype.

## Limits and next step

No training recording was read, no paired observation was collected, no speaker partition was selected, and no model was frozen. No target was replayed and no target-win count is known. Practice scoring was used zero times. The training-frozen target comparison required by the kill criterion therefore remains untested; reporting zero wins or killing the mechanism would invent a measurement.

Restore the supervisor's Modal execution infrastructure, then collect paired observations, fit and freeze the model entirely on training-speaker partitions, validate the complete-path accounting and core/sdk tests, and only then compare the target's continuation with the rhyme restart over the identical interval. The experiment card, scorer, lab tooling, data, manifests and evidence remain untouched. No engine change remains on this branch.

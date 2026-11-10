# Occurrence-preserving temporal distillation

Status: blocked by the supplied execution service; unmeasured. The experiment
card is unchanged. No listener weights, engine behavior, scorer, lab tooling,
training data, supplied manifests or evidence have been changed.

## Prediction and mechanism

The card predicts three recovered added-word targets and one collateral loss:
listener −2 and overall −2 on lab-v1.5u. These are predictions, not results.

The proposed explanation is that frame-synchronous teacher supervision puts
blank at the time a bounded-lookahead student can emit a weak consonant.
Occurrence-preserving monotone temporal transport would marginalize the permitted
timing correspondence between teacher and student trajectories while keeping
separate transcript-phone occurrences and conserving each occurrence's teacher
probability mass. This changes timing correspondence, rather than phonetic
content or the KD weight. Transcript CTC, the original acoustic augmentation and
training mix, and ordinary-reading preservation examples must remain.

Training may use authorized transcribed Quran audio and help-train training
speakers' none-labelled takes, excluding release-held-out speakers. Scripted
mistake labels must not be treated as performed transcripts. The comparison arm
must continue the same initial checkpoint on identical data with ordinary
framewise distillation. Both arms must use the same schedule and training-only
checkpoint selection. A surviving primary model would use the existing SDK
inventory, 40 ms output clock, tracker, checker and insertion verification.

## Static inspection and premise test

Read AGENTS.md and the repository audio-review instructions first. The current
modal/student_train.py losses() computes teacher indices from student frame
indices and kd_delay, clamps the indices to available teacher rows, and applies
renormalized top-k cross-entropy. There is no latent timing correspondence or
transcript-occurrence assignment in that loss. This establishes the current
implementation, not that its timing causes a target miss.

The primary SDK manifest describes a 45-frame input window and 32-frame advance
at the 10 ms fbank clock, producing eight 40 ms output rows per call. A valid
lookahead restriction must derive the teacher's available input prefix from
that geometry and the analysis window. An attention-mask change alone would
need an audit of convolution and feature-normalization future dependence.

Before training, derive the ordinary teacher/student emission-delay range from
training-only, occurrence-matched phones. Then evaluate the six registered
practice incidents diagnostically, without using them to choose that range,
training examples, weights or checkpoints. A qualifying target needs a missing
planted consonant retained by the teacher whose emission timing changes beyond
that ordinary range when teacher lookahead is restricted to the student's.
Repeated phones must retain separate occurrence identities. A consonant that
simply vanishes under restriction does not establish a delayed emission.
Fewer than three qualifying targets kills the acoustic premise.

## Execution record

All compute was requested through the user-supplied supervisor; no alternate
Modal access, credentials, exam access or files outside this worktree were used.

- Run 1: CPU inventory of the supplied model, help-train and experiment scratch
  directories, timeout 10 minutes. Returned `NotFoundError: No such file or
  directory`, with no container log or inventory.
- Run 2: L4 training-image inventory of supplied checkpoints, training shards,
  training dependencies and help-train metadata, timeout 20 minutes. Pending
  when this record was first written.
- Run 3: explicit Node-image startup check, `pwd`, timeout 3 minutes. Returned
  the same `NotFoundError: No such file or directory`, with no container output.
  This command did not depend on any corpus path or diagnostic implementation.
- Run 4: the requested six-shard `bash tools/autolab_tests.sh` harness, explicit
  Node image, timeout 5 minutes. Returned the same `NotFoundError: No such file
  or directory`, before any test log or result was returned.

The ignored supervisor result files for runs 1, 3 and 4 report zero actual GPU
cost. Run 2 remains pending at the last status check (15:00:59 UTC on
10 November 2026); it is an inventory request, not training. No premise diagnostics, training, export or practice score has completed.
All three practice-score opportunities remain unused.

## Limits and next step

There is no measured target-win count, ordinary correspondence range, loss
comparison or evidence for or against the hypothesis. Infrastructure failure is
not the card's acoustic kill criterion. Do not interpret this record as a
zero-target result or a measured failed distillation experiment. No killed
outcome is written, because neither the premise nor the built-change kill
criterion has been measured.

Implementation cannot responsibly proceed to a scoreable model here until the
supplied execution service starts a container: the required training inputs and
compute are available only through it. No .autolab/model.json was declared and
no incomplete bundle was left for validation. After service restoration, first
rerun the inventories and the training-derived premise test. Only if three or
more targets qualify should the matched training arms, transport unit tests,
complete SDK export/parity validation and fresh target evaluation proceed. If
the built model wins three or more targets, request a practice score before any
narrowing; otherwise roll back the engine changes and record an acoustic kill.

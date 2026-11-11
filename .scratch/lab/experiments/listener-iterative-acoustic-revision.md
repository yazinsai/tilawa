# Iterative acoustic hypothesis revision

Status: ready-for-human — execution infrastructure unavailable; acoustic hypothesis untested.

## Prediction and mechanism

The unchanged card predicts two recovered added words and one collateral wrong
call: listener −1, overall −1. Intermediate CTC phone distributions would be
projected into subsequent encoder blocks, allowing later blocks to revise a
provisional identity while retaining the acoustic residual. This differs from a
frozen intermediate head because both the encoder and the feedback are trainable.

The prototype attached intermediate heads after encoder stacks 2 and 4, before
later ordinary blocks. Each head reads the restored 20 ms embedded time axis;
the final phone head retains the shipped 40 ms clock. Feedback projections start
at zero and use per-recording trajectory dropout with probability 0.5 during
training. The matched control has identical parameters and intermediate/final
transcript supervision, but receives constant-zero feedback. Both arms retain
the shipped augmentation implementation and KL preservation of ordinary readings.
Neither inference arm receives text or passage-specific instructions.

Training admission was designed to require speaker provenance and transcribed
authorized Quran audio, or none-labelled help-train training-speaker recordings.
Mistake scripts would not be treated as performed transcripts. Evaluation speakers
would be excluded before a deterministic speaker partition within the training
speakers is used for checkpoint selection. No supplied training data could actually
be inspected or admitted in this attempt.

The export prototype reused the existing recurrent streaming wrapper, compared
its outputs with fp32 ONNX, and used the shipped MatMul-only int8 conversion. It
was designed to preserve installed auxiliary files in the model bundle. None of
these runtime or export checks executed, and no checkpoint or bundle exists.

## Validation and limits

The supplied supervisor command was the only execution service used. Four
requests failed before returning any container output:

| Run | Request | Result |
| --- | --- | --- |
| 1 | Default-image storage and supplied-data availability probe, 1 shard, 5 minutes | `NotFoundError: No such file or directory` |
| 2 | Training-image storage, prior-checkpoint and pinned-encoder source probe, 1 shard, 5 minutes | Same error |
| 3 | Synthetic revision unit tests in the training image, 1 shard, 5 minutes | Same error |
| 4 | `bash tools/autolab_tests.sh`, 6 shards, 10 minutes | Same error |

Each supervisor result reported `usd: 0.0`, `gpuUsdSpent: 0.0`, and
`gpuUsdLeft: 80.0`. The error has no path or traceback identifying the missing
resource. In particular, these requests do not establish another storage-full
incident or any training-data deficiency; even the storage probe never ran.

Local Python syntax compilation passed for the prototype. Its synthetic tests
covered identity initialization, feedback gradients, whole-trajectory dropout,
intermediate CTC lengths, unchanged streaming-state shape, and native Zipformer
checkpoint/streaming round trips. They were submitted but did not execute, so
their behavior is unvalidated. The existing core and SDK suites likewise did
not execute. Syntax compilation is not evidence that the model works.

No training, acoustic target evaluation, full-session exported decoding, or
practice score occurred. All three allowed practice scores remain unused. No
targets were evaluated; a zero-recovery result was not measured. Neither the
card's two-target acoustic condition nor the task's three-target stop condition
can be decided. The hypothesis is not killed.

The engine prototype was reverted because the required complete, verified model
bundle could not be produced. An ignored local snapshot remains under
`.autolab/revision-prototype/` for retry; it is not a scored or shipped model.
There is no `.autolab/model.json` selection. The experiment card, primary
listener, tracker, checker, scorer, lab tooling, supplied manifests, evidence
and test data are unchanged. No fold was performed.

Retry requires restoring the supplied execution service, then inspecting actual
training-speaker provenance, completing paired training and exported decoding,
and measuring the registered targets before requesting any practice score.

# Progressive block distillation

Status: blocked by supervisor execution; no trained model or measured target result.
The experiment card is unchanged.

## Why the implementation cannot be completed here

The supplied supervisor is the only authorized path to the training runtime,
teacher checkpoints, enrolled training recordings and export environment. On
revalidation, even a default-image request containing only `pwd` (run 5) failed
with `NotFoundError: No such file or directory`. The training-image inventory
request (run 6) failed with the same error. Neither returned command output.
This establishes an execution blocker beyond an error in a proposed training
script; it does not identify which supervisor resource is missing.

Local package discovery confirms that Torch, Transformers, quran_muaalem and
pytest are unavailable; this worktree also has no node_modules. Heavy training
must run remotely, and the user prohibits looking for an alternative Modal
connection or training files outside this worktree. Consequently, neither the
progressive arm nor its required matched conventional control can be trained,
selected or exported through an available permitted execution path.

This matters to implementation as well as measurement. The teacher layer
implementation is supplied by quran_muaalem, while the student stages come from
icefall in the training image. The repository specifies how to construct and
call these models, but does not contain those dependencies' actual encoder
block implementations. Their interfaces and the available pretrained weights
could not be inspected. A generic replacement wrapper, an assumed block API,
or a different teacher cannot establish that compact blocks were trained
through the actual remaining teacher network. Even a completed training script
would still lack the trained, verified model bundle required for this experiment.

The engine remains unchanged. The missing work
is the real teacher/student block integration, audited training-speaker data
admission, matched training, compact streaming export, complete insertion-teacher
bundle verification and paired full-session target comparison. These require
the supplied supervisor to execute commands successfully. The requested
implementation remains incomplete until that execution environment is restored.

## Prediction and mechanism

The prediction is two recovered additions and one collateral wrong call:
listener −1, overall −1. The proposed mechanism changes the compression path:
replace encoder blocks progressively, initially optimizing each compact block
through the remaining frozen teacher network. Temporary channel adapters bridge
width differences. Transcript CTC and ordinary-reading preservation remain
active throughout replacement and final compact-model training.

The comparison must use a conventional distillation control with the same final
architecture, authorized recordings, augmentation and matched training compute.
Replacement schedule and checkpoint selection must use speaker-grouped
partitions within training speakers. Help-train inputs must be eligible
none-labelled training takes; release-held-out speakers must be excluded.
Practice targets cannot select schedules, losses or checkpoints.

The user’s final stopping rule requires at least three registered targets to
recover before practice scoring. The card’s two-target consonant criterion alone
does not authorize continuing past that stricter rule. Recovery must be shown
in full-session exported decoding relative to the matched control, with
previously missing planted consonants present in correct order. Crop diagnostics
alone cannot establish this result.

## Architecture investigation

The shipped primary is a streaming Zipformer2 with 80-dimensional Kaldi features
and 40 ms posterior frames. The original KD teacher is Muaalem’s
w2v-BERT/Conformer, with its own 160-dimensional stacked feature input. A faithful
implementation must bridge actual block interfaces and frontend geometry, then
remove the temporary teacher-width adapters for the final compact export.
Simply substituting a wider Zipformer teacher would change the stated premise.

The existing primary export also supplies recurrent attention/convolution
states and a fixed output-length convention. Preserving a nominal 40 ms clock
does not by itself establish SDK streaming parity. Final export must verify
state transport, frame count, alphabet and complete bundle loading, while
retaining the existing insertion verification, tracker and checker.

## Validation and limits

Training-only replacement primitives and unit tests were drafted. They cover
ordered replacement, gradients through frozen unreplaced blocks, channel-only
adapters, attention-mask and tuple metadata transport, removal of adapters during
compaction, positive terminal compact training, and padding exclusion from the
CTC/KD/preservation objective. Local AST parsing passed. No Torch test result is
available; these primitives do not constitute a runnable experiment. They were
removed from engine source and retained only as ignored drafts under
`.autolab/progressive-draft/`, so an incomplete training utility is not presented
as a deployable listener change.

Supervisor run 1 requested the six-shard baseline build/core/SDK test harness.
Supervisor run 2 requested a training-image inventory without a GPU. Both
returned `NotFoundError: No such file or directory`, without command output or
downloaded artifacts. Both reported GPU spend of $0. These are infrastructure
failures, not evidence that progressive distillation fails or that tests pass.

Supervisor run 3 retried the inventory with an explicit L4 allocation and also
returned `NotFoundError: No such file or directory` without command output. The
actual teacher module structure, available checkpoints and audited training
speaker metadata remain unverified. No training,
export, full-session target comparison or practice scoring has run. Target wins
and aggregate score changes are unknown, rather than zero.

Supervisor run 4 requests only `pwd` in the default image to distinguish a
general execution failure from a training-specific failure. It also returned
`NotFoundError: No such file or directory` without command output.

On revalidation, run 5 repeated `pwd` in the default image and run 6 retried
dependency/checkpoint discovery in the training image. Both failed with the same
error. Run 7 retried the six-shard build/core/SDK test harness with a five-minute
timeout and also returned `NotFoundError: No such file or directory` without
command output. Neither test request reached a reported build or test result.

All seven requests failed on 2026-11-11. No remote test, data inventory
or training command produced a result. Run 7 reports rounded cumulative spend
of $0.10 and $299.90 remaining; this is not a completed training run.

Static source inspection was repeated against `modal/student_train.py`,
`modal/student_teacher_export.py`, `modal/student_export.py` and
`packages/sdk/src/model.ts`. It confirms the distinct model construction and
feature inputs, native streaming export state requirements, and the SDK's
Zipformer-only primary architecture contract. It does not validate an actual
replacement forward pass, gradient path, training partition, trained checkpoint
or exported inference result. No engine behaviour was changed, so no new unit
test is offered as a substitute for those checks. The required existing test
harness was requested again; no passing test result is available.

The experiment has not been killed on its premise. Neither zero target wins nor
zero score change can be inferred from failed infrastructure requests. No
`.autolab/model.json` was written, and no complete replacement model is available
for supervisor validation. No killed outcome was written: that would falsely
attribute the inability to run to the mechanism. Resumption requires restoring
the supplied supervisor execution environment, then completing the actual
architecture integration, matched training, export and target comparison.

No scorer, supplied manifest, test data, evidence or experiment-card changes
were made. No fold was performed.

# Progressive block distillation

Status: blocked by supervisor execution; no trained model or measured target result.
The experiment card is unchanged.

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

All four requests failed on 2026-11-11. No remote test, data inventory or training
command produced a result. The supervisor reported $0.1022 for failed run 3,
rounded cumulative spend of $0.10 and $299.90 remaining; this is not a completed
training run.

The experiment has not been killed on its premise. Neither zero target wins nor
zero score change can be inferred from failed infrastructure requests. No
`.autolab/model.json` was written, and no complete replacement model is available
for supervisor validation. No killed outcome was written: that would falsely
attribute the inability to run to the mechanism. Resumption requires restoring
the supplied supervisor execution environment, then completing the actual
architecture integration, matched training, export and target comparison.

No scorer, supplied manifest, test data, evidence or experiment-card changes
were made. No fold was performed.

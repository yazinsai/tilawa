# Generated-history word-ear readout

Status: incomplete; blocked on the parent word ear and supervisor execution.
This is preparatory training code, not an exported listener or a measured lab
result. The experiment card is unchanged.

## Prediction and mechanism

The card predicts two retained additions and one collateral wrong call:
listener −1 and overall −1. Its premise is that an early output error changes
the word ear's decoder history and induces further invented words. Freeze the
parent ear, cache its decoder states under recorded and freely generated
histories on the same authorized audio, and fit a residual output readout to
recover the recorded continuation. Compare with an identical-data
teacher-forced readout. Selection belongs entirely to speaker-grouped training
partitions, excluding release-held-out speakers. The parent seam windows,
exclusions and original-audio teacher verification must remain unchanged.

## Implementation available

`modal/student_word_readout.py` provides global edit correspondence, paired
history caching through a frozen-ear adapter contract, provenance and speaker
checks, and matched residual-head fitting. An inserted output token leaves the
recorded continuation at the same position. A deleted recorded token is the
recovery target at the state preceding the next generated output. A prematurely
ended generation targets the first missing recorded token instead of EOS; no
states are fabricated for continuations the frozen decoder did not visit.
Repeated tokens use sequence positions and a deterministic diagonal-first
alignment tie break. Alignment cannot resolve acoustically ambiguous duplicate
occurrences by itself.

Both readouts start from the same zero-output rank-32 tanh residual head. They
use the same recordings, ordering, optimizer updates and preservation objective.
The rollout arm averages recorded-history and generated-history losses per
recording; the control fits recorded histories. Both select checkpoints using
the same speaker-averaged objective on paired histories from the fixed 20%
selection speaker partition. Cached features and the base output projection
remain frozen. Cache validation rechecks provenance, speaker exclusions,
artifact and generation-policy hashes, partition assignments, and exact
sequence targets before either arm is fitted.

There is no concrete parent-ear adapter, trained checkpoint, ONNX export or SDK
integration yet. A caller would still need to supply the genuine frozen ear,
its original generation policy, verified performed transcripts and the release
speaker exclusion list. No model bundle or `.autolab/model.json` is declared.
The current code therefore cannot affect primary decoding or warnings.

## Validation and limits

The local lightweight command
`python3 -m unittest discover -s modal -p test_student_word_readout.py -v`
passed 18 tests. These cover insertion and deletion recovery targets, distinct
repeated-token positions, early termination, shared audio, none-labelled
help-train eligibility, release-speaker exclusion before decoding, altered
cache targets, mixed checkpoints, duplicated recordings and partition changes.
The nineteenth test, which checks actual fitting and preservation of frozen
weights and states, was skipped because local Torch is unavailable. Heavy
fitting was not attempted on the Mac.

Supervisor runs 1 and 3 requested environment and asset diagnostics; run 2
requested the supplied six-shard core/SDK test harness. All three returned
`NotFoundError: No such file or directory` before execution, with no container
output and zero reported spend. Run 3 used a corrected heredoc command, so the
initial diagnostic's possible shell-quoting error does not explain this
repeated failure. Run 4 requested the readout unit tests through the supervisor
and returned the same pre-execution `NotFoundError`, also with zero reported
spend. No requested supervisor command executed.

The parent branches `exp/listener-word-ear-teacher-arc-adds`,
`exp/listener-word-ear-real-seam-negatives` and
`exp/listener-short-crop-word-ear-seams` are absent from the available local
refs; the queried origin branch names were also absent. The supplied SDK
contains the folded CTC insertion teacher, but no word-ear loader or seam
integration. The CTC teacher is not a substitute for the autoregressive word
ear whose histories this experiment must train on. The parent artifact and
integration recipe, and restoration of the supervisor runner, have been
requested from the user.

Neither speaker-held-out free-rollout correction nor any target spelling or
warning has been measured. Target wins are unknown. No practice score was
requested, leaving all three allowed scores available. The infrastructure and
artifact failures do not disprove the premise and do not justify a killed
outcome. Core/SDK tests, full Torch fitting tests, paired training, held-out
free-rollout diagnostics, target retention, complete asset export and SDK
validation remain outstanding before this can be evaluated as an experiment.

No scorer, lab tooling, supplied manifest, evidence or test data was changed.
No fold was performed.

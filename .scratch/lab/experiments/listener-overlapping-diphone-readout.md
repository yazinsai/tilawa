# Overlapping diphone acoustic readout

Status: infrastructure-blocked, unmeasured. This experiment cannot be completed
in the provided environment: every authorized supervisor execution failed with
`NotFoundError: No such file or directory`, including minimal commands on both
supported images and the required test harness. Training data and the primary
model are available only through that execution route, so no trained, verified
side-model bundle can be produced here. This explains why the final diff contains
no engine change; it does not establish a failed acoustic premise.

The draft implementation is preserved
in commits `e2006685` and `29714bfe`; the final working tree restores the engine
to its starting state. The experiment card is unchanged. No target gain or
practice result is established, and the acoustic premise has not been killed.

## Prediction and mechanism

The card predicts two false warnings removed and one caught substitution lost:
listener −1 and overall −1. Its premise is that independent phone evidence can
favor a rival whose actual ordered transitions are acoustically unsupported.
The task's stopping rule requires at least three registered targets won before
an official practice score; this supersedes the card's two-window reversal test.

The draft exposes frozen pre-CTC encoder features through a separate ONNX copy.
It checks all original outputs and caches for exact equality over eight successive
windows. A seeded 32-dimensional projection feeds both readouts. Canonical
transcripts of none-labelled help-train training speakers constrain a CTC
forward-backward alignment; greedy word boundaries never supply supervision.
Held-sound runs use the engine's existing articulation collapse, retaining genuine
doubled consonants. Evaluation speakers are excluded by speaker identity.

Each ordered diphone target represents two successive articulations. A latent
middle articulation contributes equally to its incoming and outgoing pair;
endpoints contribute to their sole neighboring pair. Blank occupancy remains
blank. Thus A-B-C supplies A-B and B-C rather than three independent phone labels.
Inference compiles the complete pronunciation into overlapping pairs and rejects
any pair chain whose neighboring labels disagree on their shared articulation.

Both heads use identical adjacent-frame features, a 48-unit tanh layer, the same
output matrix dimensions, initialization, recording-balanced minibatches and
18 epochs. The control groups output probabilities by the first phone and sums
over its unlabelled right partner. Every parameter participates, but supervision
is only single-phone occupancy. Two fixed regularization candidates are selected
through three speaker-grouped training folds using diphone soft-target likelihood;
no practice observation selects a setting. This control matches parameter count
and matrix computation; its grouped output is more flexible than a conventional
one-logit-per-phone head, which limits the interpretation of an eventual result.

Complete written and already-accused pronunciations use the same existing string
warning window. CTC placement paths are marginalized with a completion-conditioned
timing prior to avoid a pure path-count reward. Only a strict, finite reversal—
diphone prefers written, single-phone control prefers rival—supplies a certificate.
Certificates bind the exact word, both complete pronunciations and frame endpoints.
The checker retains its internal ownership ledger and changes only the public
wrong-word verdict. Graph confirmations and insertion decisions are exempt.

## Validation and limits

The draft includes tests for shared-articulation consistency, impossible complete
paths, uniform-emission length neutrality, the control's probability grouping,
exact-window certificate binding, JSON transport, graph exemption, neighboring
insertion preservation, declared-asset failure, and session completion behavior.
Python tests additionally cover non-greedy latent boundaries and the shared middle
articulation's targets. An exhaustive enumeration of all four-frame acoustic
paths independently verifies phone occupancy marginals, including doubled phones
that require a separating blank. All four tests passed locally using
`arch -x86_64 python3 modal/test_student_diphone.py`; Intel mode matches the installed
numpy binary. Native ARM Python failed to import that binary. Python syntax
compilation and whitespace checks also pass. These checks establish only the
small alignment/target mathematics, not TypeScript behavior or acoustic gains.

The auxiliary SDK path buffers session audio and reviews warnings at `end()`.
It preserves primary listener and tracker computation but does not prevent an
earlier live warning from being displayed. This is an acoustic lab experiment,
not evidence of immediate streaming warning readiness. A bundle must preserve
all existing teacher insertion assets and verify normal Node SDK loading before
it can be selected for supervisor scoring.

Supervisor run 1 requested a read-only data/model/package census and returned
`NotFoundError: No such file or directory`, with no container output and reported
cost zero. Run 2 requested the required six-shard core/SDK build/test harness
and failed with the same `NotFoundError`, again without test output. Run 3
requested a minimal working-directory, `/lab/data` and numpy check and failed
with the same error and zero reported cost. Run 4 repeats the census through
the supervisor's supported training image and also failed with the same
`NotFoundError`, no container output, and zero reported cost.

On continuation, the full source and unit-test draft was restored from
`29714bfe` while retrying the authorized execution route. Run 5 requested
`pwd`, `/lab/data` listing and a numpy import on the Node image at
2026-11-12 16:14:24 UTC. Run 6 requested `bash tools/autolab_tests.sh` with
six shards at 16:15:59 UTC. Run 7 explicitly selected the training image and
requested only the current directory and existence of the help-train labels
at 16:19:01 UTC. All three returned the same `NotFoundError`, no container
output and zero reported cost. Both supported images therefore failed again
before any data census, feature extraction, training,
Node SDK bundle verification or TypeScript test result could be obtained.
No alternative Modal route was attempted, no official practice score was
requested, and no model bundle has been declared. The restored draft was
rolled back again after all pending jobs returned, retaining its existing
source commits for recovery. All seven requests and returned errors remain
in the untracked `.autolab/modal-request-*.json` and `modal-result-*.json` files.

None of the four registered windows was evaluated. Target wins and the practice
change are unknown, not zero. Neither the card's two-reversal criterion nor the
task's three-target rule can be assessed. The engine and new source/test files
were rolled back because an untrained, unverified auxiliary model cannot be
submitted as a completed listener experiment. The draft remains recoverable
from git history once the supervisor's Modal execution is restored; this report
records an infrastructure failure rather than evidence against overlapping
diphone acoustics. There is no fold or premise-killed outcome.

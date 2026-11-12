# Overlapping diphone acoustic readout

Status: implementation drafted; remote training and validation pending. The
experiment card is unchanged. No target gain or practice result is established.

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
the supervisor's supported training image; its result is pending.
No alternative Modal route was attempted, no official
practice score was requested, and no model bundle has been declared.

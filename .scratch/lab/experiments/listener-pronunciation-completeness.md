# Pronunciation completeness

Status: implementation checkpoint; remote training and validation blocked. No
trained bundle, target measurement, practice score, or kill decision exists yet.
The experiment card is unchanged.

## Prediction and mechanism

The card predicts listener −1 and overall −1: individual plausible acoustic
sounds need not constitute a complete rival pronunciation. The task's final
instruction requires at least three registered target wins before a practice
score; fewer than three measured wins require rollback and a killed outcome.
Infrastructure failures do not establish that acoustic kill condition.

The implementation extracts frozen pre-CTC encoder activations into a separate
ONNX graph. Export verifies all original outputs and caches bit-for-bit over
eight streaming windows. A fixed 32-dimensional projection feeds a small
pronunciation-conditioned attention head, which aggregates mean and minimum
phone evidence to classify complete versus incomplete realization.

Only help-train training speakers' none-labelled takes supply fitting examples.
Full-take canonical CTC alignment estimates word and phone boundaries. Each
positive is a complete recorded word; its paired negatives are naturally
contiguous crops missing beginning or ending phones. Half extend the other edge
into real neighbouring speech; matching positives also contain that context.
Thus complete and incomplete examples overlap in duration without splicing
audio. Examples are balanced by pronunciation and class. Equal-weight original,
onset-trimmed and ending-trimmed views integrate one-frame boundary uncertainty.

Speakers are separated before normalization: three groups fit, one selects the
checkpoint, and one freezes calibration. Any speaker appearing in a non-train
split is excluded. Actual engine-caught scripted substitutions from calibration
training speakers are preservation controls, never inferred canonical positives.
The cutoff is at most one half and no greater than any complete calibration
anchor or substitution control probability. Equality preserves the warning.
No practice case selects the checkpoint, features or operating point.

The SDK judges every phone of the already accused pronunciation over the
checker's exact existing string-path support window. A certificate contains the
word, both pronunciations, frame bounds, probability and frozen cutoff. It can
only downgrade a matching existing wrong-word confirmation. Internal ownership
and confirmation bookkeeping remain intact; graph confirmations and insertion
decisions retain their existing paths. The primary decoder and tracker receive
the original evidence. Audio sessions retain an auxiliary copy of audio and
publish applicable uncertainty revisions at completion.

The Node loader checks all three declared auxiliary files by byte size and
SHA-256, validates their primary-model identity, and rejects incomplete bundles.
No auxiliary asset has been generated or declared for scoring yet.

## Validation and limits

Local checks passed: two crop-construction tests, Python compilation, shell
syntax, and `git diff --check`. Crop tests verify natural continuity, removal of
source onset/ending samples, paired class balance and uncertain views staying
inside the supplied interval. Core/SDK tests cover discovery, frozen-cutoff
equality, exact-window matching, graph preservation, evidence transport,
end-of-session publication, and missing/corrupt asset rejection. These
TypeScript tests have been written but have not returned a remote result.

Supervisor runs 1 (data inspection) and 2 (environment probe) returned
`NotFoundError: No such file or directory`, no container output and zero compute
charge. Run 3 submitted the required six-shard test harness; run 4 submitted
`bash modal/student_completeness_run.sh`. Both remain pending at this checkpoint.
The local NumPy installation cannot load its binary extension on this CPU;
construction checks were made independent of that dependency and passed.

No training-only discrimination, export parity, control count, frozen cutoff,
actual engine target requests, target wins, clean-session behavior, studio
regression, or aggregate score has been measured. In particular, training uses
word crops while some engine support windows contain neighbouring speech; the
head's generalization to those windows must be measured, not assumed. This
checkpoint is not a score-ready result and must not be evaluated by silently
falling back to the primary model without its proposed auxiliary assets.

Resume with remote tests and training, inspect the training-only report, then
measure the four registered targets at that frozen cutoff. If fewer than three
are won, remove the implementation and record a killed outcome. Otherwise verify
the complete primary-plus-auxiliary bundle under this experiment's scratch
directory, declare `.autolab/model.json`, and practice-score before deciding.
No practice scores have been used (three remain). No fold has been performed.

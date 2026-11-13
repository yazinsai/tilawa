# listener-vowel-conditioned-residual

Prediction: listener −1, overall −1, removing one wrong-word false warning with
no forecast losses. The premise is that predictable vowel structure occupies
the acoustic readout and hides distinguishing consonant evidence. Status:
implementation checkpoint; training and practice validation are blocked.

## Mechanism implemented

The separate frozen streaming encoder graph exposes pre-CTC features, projected
to 32 dimensions. Its exporter checks every primary output and cache against the
inherited graph over eight successive calls. Primary decoding remains on the
inherited graph.

An acoustic-only semi-Markov model sums possible vowel qualities and intervals
up to 75 native frames. Blank supplies neutral context evidence. For every
possible vowel interval it uses normalized position and its square; posterior
expectations produce one ten-dimensional context distribution for the whole
recording. Neither candidate pronunciation enters this inference. A regularized
linear predictor removes the expected vowel-conditioned features.

Three identically sized 32-hidden-unit phone readouts learn complete CTC
transcripts: residual features, original features, and features after subtraction
of a predictor fitted with shuffled contexts. Shuffling destroys alignment
within training speakers while preserving each speaker's context population.
The readouts share data, initialization, update order and search budget.
Normalization and both predictors use fitting speakers only. Common checkpoint
and regularization selection minimize average validation CTC across all three
arms in a deterministic speaker-grouped training partition. The fitter enforces
a one-CPU-day limit using recorded extraction/export CPU plus fitting CPU.

Training preparation accepts help-train training speakers' none-labelled takes
and their authorized mushaf transcripts. Repeats, scripted events, unsupported
labels and unscored takes are excluded. Any non-training or excluded row bans
its entire speaker group. Waveforms must remain inside the authorized source
root. No practice example participates in training or selection.

The checker exposes the complete hypotheses and exact window from its existing
support comparison, including unchanged neighbours where that comparison uses
them. The SDK compares both complete CTC chains on identical residual, original
and shuffled-control windows. A certificate requires written preference from
the residual and rival preference from both controls, all strictly finite. A
second complete-chain comparison neutralizes non-consonant identities and must
also prefer the written consonants; vowel agreement alone cannot veto.

Certificates bind both strings, word and native-frame bounds. Graph decisions
are exempt. Primary internal ownership remains intact, preserving insertion
merges and other decision classes. Audio sessions publish a matching final
uncertainty revision at completion. The verified bundle must retain all inherited
files, including the insertion teacher and its external data, byte for byte.
The Node loader rejects missing, corrupt or incompatible declared side assets.

## Validation and limits

Four Python unit tests passed locally under the installed x86_64 Python runtime:
speaker exclusion/partitioning, residual versus shuffled nuisance removal,
complete CTC consumption, and inherited-bundle preservation/missing-asset
rejection. The ordinary arm64 Python cannot import the installed NumPy binary.

Numerical source smoke checks passed for vowel posterior normalization and phase
moments, exhaustive small CTC path sums, both-control agreement, unsupported
phones, and the positive-consonant guard. All five new core certificate cases
also passed in a stripped-TypeScript Node assertion harness, including exact
window matching, public-verdict preservation and graph exemption. Python, shell
and changed TypeScript syntax checks and `git diff --check` passed. These smoke
checks do not establish TypeScript type correctness or replace the required
core/SDK Vitest suites. Added SDK tests cover session revisions, primary tracker
and insertion preservation, evidence transport and declared-asset loading.

The authorized supervisor requests for data inventory, the six-shard test suite
and the complete 60-minute extraction/training/bundle/parity pipeline returned
`NotFoundError: No such file or directory` before execution, with zero compute
reported and no container log or output bundle. The final six-shard test retry
returned the same error. All four requests failed before execution; request/result
details and `.autolab/blocker.json` are retained untracked in `.autolab`.

No real training recording or target audio was inspected, no readout was fitted,
and no complete scorer bundle exists. No practice score was attempted; all three
allowances remain. Target wins and losses are unknown. Infrastructure failure
does not establish the acoustic kill criterion, so no killed outcome or model
override is declared. Scoring the inherited listener would not evaluate this
implementation. The card, scorer, lab tooling, supplied manifests and evidence
remain unchanged.

## Resume validation

After supervisor execution is restored, run the final core/SDK suites and
`bash modal/student_vowel_residual_run.sh` through the authorized supervisor.
Verify the returned training audit, inherited outputs, complete bundle and
exported-runtime parity before declaring `.autolab/model.json`.

Then inspect the actual checker-requested comparisons for:

- `lab-v1.5-p11/phone/warning:wrong_word/26:106:1/104.32`
- `lab-v1.5-p36/phone/warning:wrong_word/55:48:1/282.64`
- `realuser-v1-sob-049/phone/warning:wrong_word/92:13:1/36.44`

Freeze selection before this inspection. If fewer than three targets are won,
roll back engine changes and record the measured killed result as instructed.
If three are won, practice-score before deciding; other losses belong to the
fold gate. No fold has been performed.

The residual may remove genuine consonant information, its context is correlated
with the shipped encoder, and small-data transfer remains unmeasured. Completion
revisions do not establish timely suppression of live warnings. Whole-session
audio retention and a second encoder pass are also not a measured device budget.
This checkpoint is not a completed or score-ready experiment.

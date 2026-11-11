# Reverse-predictive acoustic pronunciation evidence

Status: implementation present; training and target validation blocked by the
supervisor's Modal infrastructure. The experiment card is unchanged. No acoustic
premise result, target gain, or practice score is established.

## Prediction and mechanism

The card predicts listener −1 and overall −1, one removed false warning with
no predicted loss. Its premise is directional ambiguity: a weak consonant's
following vowel may retain a transition that supports the written pronunciation
even when a predictor looking only backward in the recording prefers the rival.

The SDK retains the native 80-dimensional, 25 ms window / 10 ms shift log-mel
trajectory. It does not use recognition logits, phoneme peak confidence, or an
existing edit backtrace as its acoustic observations or boundary supervision.
The CPU trainer fits two diagonal, phone-conditioned autoregressive Gaussian
models of identical capacity. Each predicts the current spectral observation
from the previous observation in its own direction; the reverse arm reverses
both the recording and the candidate pronunciation. Their regression intercepts,
coefficients and residual variances are learned separately on identical takes.

Both models sum over complete pronunciation paths with optional background
states between phones. Every prescribed phone must be consumed; repeated labels
require an intervening background state. Self loops integrate phone durations.
Scores are normalized by the probability of completing the same timing graph in
the available frames. The two arms have identical topology and duration treatment.
Uniform boundaries are used only to initialize EM; training, validation and
inference then marginalize boundaries.

The preparer accepts only help-train's none-labelled training-speaker recordings
and canonical ayah pronunciations. It excludes every speaker present in a
non-training split, plus an explicit excluded-speaker list, and rejects deviations
or repeats on purported none takes. The trainer separately enforces these guards.
Three deterministic speaker-disjoint folds select one shared shrinkage value
from 8, 32 and 128 by the mean of both arms' validation predictive likelihoods.
Normalization is fitted separately within each fold's training partition. Eight
EM iterations, the variance floor and the timing graph are fixed before any
practice inspection. The selected pair is then refitted on all eligible training
recordings. Mistake scripts are never treated as performed transcripts.

At an otherwise confirmed non-graph wrong-word warning, the checker exposes the
same complete written and already-accused pronunciations and the exact existing
string-path support window. A veto requires strictly higher reverse likelihood
for the written pronunciation **and** strictly higher forward-control likelihood
for the accused pronunciation. Ties, unsupported phones, unavailable audio,
impossible completions and non-finite comparisons abstain. Neither arm chooses a
new rival or changes a confirmation constant.

The veto changes only the public verdict to uncertain. The checker's internal
confirmation and ownership history remain intact, preserving insertion decisions
and subsequent candidate generation. The primary listener, posterior rows,
tracker events and graph confirmations are preserved. Exact pronunciation,
location and frame bounds bind each serialized certificate; it cannot be reused
after a window or hypothesis changes. Live audio sessions can compute the same
comparison once the existing warning window is available; SDK evidence enrichment
also discovers comparisons generically for ordinary stored-evidence replay.

## Validation and limits

Both core and SDK TypeScript builds and all five new core certificate tests passed locally.
They cover baseline-equivalent candidate discovery, finite strict reversals,
exact frame/pronunciation matching, unchanged internal issue creation, JSON
transport, and preservation of graph confirmations.

The full core suite ran locally: 753 tests passed and eight hit execution
deadlines, with two worker RPC timeout errors. Re-running the five affected files
with one worker and a 30-second default deadline passed seven of those eight
tests. The remaining startup-persistence case has an explicit 15-second deadline
that the CLI cannot override; it took approximately 18 seconds in both runs.
No assertion failure was reported. The required Modal test run remains blocked.
Running both startup-persistence fixture cases directly through the same
tracker/checker functions and assertions, without Vitest's deadline, passed in
15.3 and 18.6 seconds. Thus all existing assertions exercised passed, but the
full Vitest invocation did not finish cleanly on this host.

SDK unit tests cover exhaustive complete-path sums with optional background
states and repeated phones, reversal of observation and pronunciation order,
native frontend equality under different audio push sizes, missing coverage,
model validation, and missing/corrupt declared assets. Python unit tests cover
the forward/backward recursion, posterior occupancy normalization, the expanded
Gaussian residual formula, training eligibility, equal parameter capacity and
fold-local normalization. The full local SDK suite passed: 34 tests, with one
optional help integration test skipped. All four Python trainer tests passed in
an isolated worktree-local ARM NumPy environment. These are implementation checks,
not training or measurements of the target acoustics.

Supervisor requests 1 (training-data inventory), 2 (a minimal `pwd` health check),
3 (the supplied six-shard `bash tools/autolab_tests.sh` command), and 4 (a second
`pwd` health check during final review) all failed
before command execution with `NotFoundError: No such file or directory` and
reported zero spend. This is an infrastructure failure, not a training-data
finding or an acoustic kill. The service failure was reported to the user.

No help-train recording or registered target audio was accessible through the
authorized service. Models were therefore not fitted, training-speaker coverage
and excluded-speaker metadata were not inspected, and the four actual
engine-requested target windows were not tested. **Targets tested: 0. Practice
scores: 0 of 3. Targets won: unknown.** The card's two-reversal premise criterion
and the user's stricter three-target continuation minimum remain unevaluated.
No aggregate gain, clean-session safety or studio preservation is claimed.

The implementation also has practical limits to measure after service recovery:
diagonal linear prediction represents only adjacent-frame dependencies; the
training corpus may cover too few phone realizations; startup prediction at a
window edge can affect a short word; and an unfamiliar genuine substitution
could receive an erroneous veto. Session observations are currently retained
in memory rather than pruned to the checker horizon. The model is a CPU JSON
side scorer; no new neural inference graph is required.

## Required continuation

After restoring the supervisor, inspect the provided help-train labels and
speaker exclusions first. Run the native SDK preparer
`src/bench/reverse-prepare-cli.ts` with the training dataset and excluded-speaker
list, then `modal/student_reverse_predictive.py` with its output, inventory and
tokens. The trainer writes `reverse-emissions.json`, `reverse-asset.json` and a
training audit report into the persistent experiment directory.

Copy the complete original primary bundle, including its existing auxiliary
teacher files, without modifying its manifest or model. Add the two generated
reverse scorer files. The Node loader verifies the descriptor's hash and length,
both predictor tensors, inventory, and acoustic clock, failing on missing or
corrupt declared files. Verify the complete bundle before selecting it through
the untracked `.autolab/model.json`; **no model selection has been written**, so
decoding the primary alone would not evaluate this experiment.

Discover comparisons through the actual SDK/checker route and report the four
registered targets using labels only for case selection. If the built veto wins
fewer than three, follow the user's rollback/write-up/killed-outcome procedure.
If it wins at least three, run the authorized practice score before deciding.
The fold gate, rather than this implementation, decides population losses and
whether to keep the change.

No scorer, lab tooling, supplied manifest, test data, evidence, registry or
experiment-card edits were made. No fold or discard was performed.
The untracked `.autolab/outcome.json` records `needs-human` for the infrastructure
blocker, without an acoustic kill or invented target-win count. This branch is
not ready for practice scoring until a trained, verified bundle is installed.

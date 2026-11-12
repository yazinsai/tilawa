# Emphasis-domain observation evidence

Status: needs-human — all authorized Modal requests failed before execution. The implementation is a prototype; no training-frozen model is active.

## Prediction

The unchanged card predicts tracker −1 and overall 0 on practice: removing `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84` may expose the documented nearby listener false warning. The earlier audit's 0.3915-edit continuation disadvantage is context from the card, not a new measurement. No additional target wins are assumed.

## Mechanism

`emphasis-channel.ts` represents pronunciation coloration as a two-state latent process. An aligned emphatic consonant (ص, ض, ط or ظ) activates the colored state; subsequent aligned phones retain it with a training-selected probability. The emission matrix is a normalized distribution over observed phone identities conditioned jointly on reference-phone identity and coloration. Both states share substitution-cost-derived phonetic priors, with regularization selected on training speakers. It retains complete phone distributions rather than independent feature marginals.

`trackerInput` averages the original posterior rows over each emitted token's inclusive frame interval, distributes omitted top-k mass across unlisted classes, and then conditions on nonblank. The resulting compact distribution travels with that token; listener identities, timestamps and confidence are preserved, and acquisition continues to receive decoded-only input.

The local search carries a path-specific posterior probability of coloration alongside each winning edit path. Ordinary diagonals and restart diagonals apply the same normalized observation likelihood relative to the pooled training distribution, using the existing three-nats-per-edit conversion. A restart inherits the source path's coloration rather than resetting it at a text location change. Insertions and reference-only deletions retain the path's state; a missing emphatic consonant cannot activate it. Ordinary word boundaries carry the state unchanged. Array replacement, recentering and rewind retain the corresponding state with the cost and origin.

Movement edges and their charges, identity and duration edits, commitment margins, event delivery, windows and checker handling are unchanged. The additional state is marginalized within each retained edit path; runtime still selects among edit paths with the existing minimum-cost recurrence. It does not forward-sum all runtime movement or edit trajectories. A new local search after global acquisition/reacquisition starts uncolored; this prototype does not adapt a session-wide pronunciation state.

The matched independent control has the same inventory, two emission rows per reference phone, regularization search, state probability parameter and training data. At every ordinary aligned phone its colored-state prior is independent of the previous phone. Emphatic alignment still activates coloration at that phone. This isolates cross-phone persistence from the extra emission capacity.

## Training protocol

`packages/sdk/src/emphasis-fit.ts` opens only the supplied listener model and `LAB_DATA/help-train`. It selects scored, none-labelled, event-free, repeat-free and non-excluded training recordings, and excludes every speaker appearing under any non-training split. It never opens practice audio or uses warning scores.

A normalized pair channel uses match/insertion/deletion probabilities 0.8/0.1/0.1, with insertion-only continuation and termination probability 0.9 after the reference is exhausted. Forward/backward marginalizes token-to-reference placement and the emphasis-domain boundaries. Emission estimation uses fractional latent-state and observed-phone counts, with at most 20 EM iterations.

Up to three deterministic speaker groups select persistence from {0.25, 0.75, 0.95} and prior mass from {8, 32}, separately for persistent and independent models. Held-out predictive log likelihood is averaged per observation, then per recording, then per speaker; no prolific speaker can dominate selection by recording count. Final fitting uses all eligible training speakers after those choices are frozen. The output report records listener, eligible-label, decoded-training-recording and partition hashes along with every selection score. The estimator writes both fitted JSON models and its report to `LAB_OUT`.

No fitted parameter table has been guessed or copied from another experiment. `EMPHASIS_MODEL` remains undefined pending successful training.

## Validation

Added unit tests cover cross-phone persistence, the identical-capacity independent control, identity/state-conditioned likelihood, brute-force agreement with forward-summed latent paths, positive normalized emissions, deterministic fitting on ambiguous placement, original posterior coordinates and residual mass, immutable tracker-input transformation, word-boundary state propagation and rewind, deleted emphatics, and neutral-model preservation of existing duration costs. The component-replay assertion allows compact local metadata while still verifying the original sounds and full checker evidence.

Node syntax checks passed for the changed engine files, estimator and new test file. Small source-level checks, using Node's built-in TypeScript stripping without installing dependencies, passed for persistence, deletion without emphasis activation, neutral identity/duration costs and normalized latent fitting. The deletion fixture includes two subsequent distinct sounds so the existing minimum-cost search resolves deletion versus substitution. These checks do not establish a TypeScript build or Vitest success. `git diff --check` passed.

All three authorized supervisor requests failed before command execution with `NotFoundError: No such file or directory`, and all three result receipts report zero spend:

- Request 1: training-data layout and the existing target trace.
- Request 2: core/SDK build followed by training-only emphasis fitting.
- Request 3: six-shard core/SDK build and tests through `bash tools/autolab_tests.sh`.

There are no shard logs or executed test results. No data, fitted model, target comparison, target warning recovery or practice score has been obtained. Repeating this startup failure further would not produce an acoustic measurement. The authorized supervisor was the only remote route attempted; no alternate Modal access or storage changes were used. `.autolab/outcome.json` records `needs-human` with no invented measurements.

## Required completion

1. Complete the six-shard build/core/SDK tests on the final source.
2. Run the training-only estimator and freeze its selected persistent model in `emphasis-model.ts`, retaining both models and the fit report for review.
3. Compare the training-frozen persistent model with its matched independent control on the target's identical local observation interval. Replay the full target through the unchanged checker and count target recoveries, including any exposed listener warning in the notes.
4. If the stated premise fails or the built change wins fewer than three targets, revert engine/test changes and commit the measured kill result. If three or more targets are won, run the authorized practice score before deciding. The supervisor's fold gate decides retention.

No practice-score allowance has been consumed. Infrastructure failure supplies neither a zero-win measurement nor evidence that the hypothesis is false.

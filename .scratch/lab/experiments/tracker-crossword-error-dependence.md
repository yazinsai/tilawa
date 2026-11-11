# Cross-word recognition-error dependence

Status: needs-human. Supervised Modal execution is unavailable; the experiment has not been trained or measured.

## Prediction

Tracker −1, overall 0 on practice. A corrected localization may expose the previously documented nearby listener warning. The card names one target: `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.

## Mechanism

The proposed model predicts a second word's recognition-error pattern from the first word's error pattern and both pronunciation structures. Its conditional-versus-independent log-likelihood ratio would contribute to competing local paths over their common observed interval. Training must use only help-train training speakers' none-labelled recordings, marginalize transcript-consistent word boundaries, share parameters through phonetic features, and freeze calibration within speaker partitions. An equal-capacity independent-word control is required to distinguish cross-word error dependence from a better single-word model. Movement edges, duration charges, commitment margins and checker handling must remain unchanged.

The card's premise test is whether the frozen conditional model gives the near continuation relative evidence beyond its independent control on the actual comparison interval. The user's execution rule additionally requires stopping and reverting if a built prototype wins fewer than three targets; the card's single named incident cannot establish a three-target gain by itself.

## Source inspection

`trackerInput` retains decoded phone observations and strips posteriors. Locked tracking advances the dynamic program in `local.ts`; `tracker.ts` consumes its position and recent cost. The search retains a cell for every reference offset in its local ayah window. Consuming transitions read the previous row, deletions close the current row, and word-start jumps retain their existing distance and movement charges. Costs are ranked before subtraction of the common row minimum. This source inspection supports locating the model at local path comparisons; it does not establish that cross-word dependence helps the target.

The existing listener channel records independent substitution, deletion and insertion counts. Its accumulator collapses held sounds before fitting. Reusing that channel for latent training alignments would require preserving the local search's separate duration charges, rather than silently changing their pricing. The reference index provides word ranges and connected pronunciation strings needed for a transcript-constrained word model.

## Validation and infrastructure

The first authorized supervisor request attempted to inspect storage and list the permitted help-train and practice data. It returned `NotFoundError: No such file or directory`, with zero reported compute cost, and no command log or data listing. Fetching the same result through `modal-result 1` reproduced that error. The error does not identify which path is missing, so storage exhaustion is not established.

A minimal `pwd` request (run 2, two-minute command timeout) was submitted at 08:32:32 UTC. The requested six-shard core/SDK build-and-test command (run 3, five-minute command timeout) was submitted at 08:36:12 UTC. At 08:38:23 UTC neither had returned a supervisor result or a command log; the command timeout does not establish a result while the supervisor request is pending. Run 2 subsequently returned the same `NotFoundError: No such file or directory`, confirming that even `pwd` could not execute. Run 3 also returned the same `NotFoundError: No such file or directory` before producing any build or test output. All three requests failed at the infrastructure boundary; no tests ran.

No training data, practice evidence or target comparison was available to this attempt. No model was fitted, no coefficients were selected, no engine code was changed, and no practice score was requested (0 of the permitted 3). The experiment card, scorer, tooling, manifests and test data remain unchanged.

## Limits and resumption

The hypothesis and kill criterion are untested. An infrastructure error is not a zero-target measurement or evidence against the mechanism, so this attempt is blocked rather than killed.

After supervised execution is restored, first inventory the authorized none-labelled training decodes and enforce the speaker split. Fit the conditional model and equal-capacity control with marginalized boundaries and training-only calibration, then inspect their relative evidence on the actual comparison interval. Only after that premise check should engine integration, behavioral unit tests, product replay and any permitted practice score proceed. No fold was requested or performed.

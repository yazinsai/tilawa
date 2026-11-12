# Replicate-shared acoustic readout

Status: blocked on the authorized supervisor execution service; hypothesis unmeasured. The draft is archived in commit `c81dbde5`; revert `597102c6` restores the active engine to its initial state.

## Prediction

The card predicts listener −1 and overall −1: one false substitution removed without a loss. Its premise is that noise-specific directions in the frozen encoder favour a rival pronunciation, while feature directions reproducible across independently degraded copies favour the written pronunciation. The four card targets are p11 at 104.32 s, p36 at 282.64 s, p55 at 107.08 s and realuser sob-049 at 36.44 s.

The user requires at least three target warnings removed before submitting a practice score. Fewer than three measured removals would kill this implementation. No target comparison or practice score was obtained, so neither the card's reversal criterion nor that requirement has been tested.

## Draft mechanism

The draft exposes effective pre-head features from the exact shipped ONNX encoder, including activation dequantization for a dynamically quantized head. An export-time check compares all original outputs and recurrent states with the unmodified graph over eight windows and checks the effective head's reconstruction of logits. These checks were implemented but could not be run.

Each eligible help-train recording receives two independent, deterministic-seeded noise/channel degradations, with no resampling or time warp. Only none-labelled, usable train-speaker takes without labelled edits or repeats are accepted; an explicit exclusion list and rejection of non-training speakers provide further guards. The actual remote labels and speaker coverage could not be inspected.

A regularized exchangeable-view CCA projection whitens pooled within-view covariance, symmetrizes cross-view covariance and retains 12 positive-correlation directions. Recordings receive equal covariance weight. Its matched control retains 12 ordinary covariance directions. Both arms use the same Gaussian phone emission capacity, four EM iterations and complete CTC boundary marginalization, with the timing prior conditioned on sequence completion. Three speaker-grouped folds refit centering and both projections without held speakers; the shared regularization is chosen from 1, 16 and 64 using training validation only. A timestamp bounds extraction and fitting to 23 hours. These choices were fixed before practice inspection.

The SDK draft discovers already-supported string-path substitutions through the existing checker windows, compares complete written and existing rival pronunciations on exactly those rows, and carries a veto only for shared-written > shared-rival and control-written < control-rival. Primary phones and posteriors are retained. Internal confirmation ownership remains unchanged; graph confirmations and insertion decisions are exempt. Session completion replays the certificate and retracts only the exact corresponding public occurrence. The optional loader requires both side assets, their hashes, primary model hash and matching frame geometry.

The extraction and exact-window probe utility is packages/sdk/src/bench/shared-acoustics-cli.ts. Training and graph extraction are modal/student_shared_acoustics.py and modal/student_shared_acoustics_export.py. No scorer, lab tooling, card, test data or supplied manifest was edited.

## Validation and execution limits

- Python syntax compilation passed for the exporter, fitter and numerical test module.
- A lightweight Node check of training degradations passed: deterministic reproducibility, distinct copies, original samples unchanged, equal length, finite samples, unchanged impulse onset and silent-input preservation.
- Numerical CCA, complete-boundary occupancy, quantized-feature extraction, exact certificate, session and loader tests were written. They have not passed an executable suite. Local numerical tests could not import the installed NumPy because its binary architecture differs from this Mac's Python.
- Supervisor run 1 requested a read-only data/model audit; run 2 requested the prescribed six-shard core/SDK build and tests; run 3 retried a smaller data/model audit. All three returned `NotFoundError: No such file or directory`, zero compute cost and no shard logs. No recording, feature or target result was returned.
- Run 4 requested only a minimal Python print through the same authorized supervisor. It also returned the identical `NotFoundError`, zero compute cost and no logs. This establishes an execution-service blocker independent of the dataset audit and draft training code.

No training observations, fitted projection/emissions, verified ONNX side asset or complete deployment bundle exists. No .autolab/model.json is declared. Practice scores used: 0 of 3. Targets won, listener change, overall change, studio safety and preservation of real substitutions are unknown; reporting zero target wins would fabricate a measurement.

## Resume

Restore the supervisor executor first. Run `git revert 597102c6` to restore implementation commit `c81dbde5`, rerun the latest build/tests plus Python numerical tests, inspect the authorized help-train schema and exclusion data, extract both views and freeze the training-selected fit. Copy the unchanged primary model and all inherited auxiliary assets into the persistent experiment bundle, verify the complete side bundle, and inspect all four warnings using the generic probe with baseline engine-generated windows and pronunciations. Labels may identify report rows only. Kill and revert if fewer than three targets are removed; otherwise run the authorized practice score and leave the gate to decide losses.

No scientific kill is claimed from infrastructure errors.

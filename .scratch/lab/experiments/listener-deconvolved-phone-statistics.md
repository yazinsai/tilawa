# Deconvolved phone statistics

Status: implementation committed as `f2a50fa1`; training and scientific validation blocked on the authorized supervisor executor. No fitted side asset or deployment model is declared. This branch is not ready for model validation or folding.

## Prediction and stopping rule

The card predicts listener −1 and overall −1: one wrong-word false warning removed without a loss. The premise is that ordinary acoustic emissions inflate phone covariance by treating noisy training observations as exact pronunciations. Separating observation noise could restore a written-versus-rival acoustic distinction.

The four requested warnings are p11 at 104.32 s, p36 at 282.64 s, p55 at 107.08 s and realuser sob-049 at 36.44 s. The user's stopping rule requires at least three target removals before a practice score. Target comparisons have not executed, so both the reversal kill criterion and that requirement remain untested. No scientific kill is claimed from an infrastructure error; targets won and score changes are unknown.

## Implemented mechanism

The extraction and exact-window certificate scaffolding was adapted from `exp/listener-replicate-shared-acoustics`; its CCA mechanism was replaced by the paired latent likelihood described here.

`modal/student_deconvolved_phone_export.py` exposes effective pre-head activations from the unchanged deployed encoder. For a dynamically quantized head, these are dequantized activation codes, with effective weights including zero points and per-channel scales. The exporter requires exact parity of all original outputs and cache states across eight streaming windows and checks reconstruction of pre-softmax logits. These checks are implemented but have not executed.

`packages/sdk/src/bench/deconvolved-phone-cli.ts` prepares paired frozen observations from usable help-train none-labelled training-speaker recordings without edits or repeats. It requires explicit speaker exclusions and also excludes every speaker appearing in a non-training split. Actual remote label schema, exclusions and speaker coverage could not be audited. Both views preserve the original samples' clock and receive independent seeded additive colored noise at a fixed 20 dB SNR; no independent filtering, cropping or time warping is used. An unaugmented feature view supplies training-only validation and initial transcript occupancies.

`modal/student_deconvolved_phone.py` learns one 12-dimensional PCA/whitening projection for both arms. Within-occurrence feature-noise covariance is the recording-balanced mean of `(view0 − view1)(view0 − view1)' / 2`. The latent model assumes `x_v = y + e_v`, `y | phone ~ N(mu, Sigma)`, and independent view errors `e_v ~ N(0, N)`. Its joint paired likelihood factors into a Gaussian for the average with covariance `Sigma + N/2` and a phone-independent difference with covariance `2N`. Six constrained EM iterations use the conditional latent mean and covariance as sufficient statistics. Positive eigenvalue constraints and pooled shrinkage maintain proper covariance matrices; the method does not subtract an estimated noise covariance from an ordinary covariance and clip the result.

The ordinary arm fits Gaussian emissions to the identical two views as exact observations. Both arms use the same projection, initialization, capacity, regularization, complete pronunciation topology and boundary marginalization. Legal outgoing timing arcs have equal probability, conditioned on completing the full pronunciation. Three speaker-grouped folds refit projection and noise estimates without held speakers. One shared shrinkage strength, from the fixed choices 1, 16 and 64, is selected by average matched phone-identity validation loss on unaugmented held training recordings. Counterfactual lexical pronunciations on genuine none takes provide diagnostic preservation controls, never selection targets. Extraction and fitting share a 23-hour deadline.

At runtime both arms receive the identical unaugmented deployed feature rows. The SDK obtains complete written and already-accused pronunciations and their exact frame window from an already-supported string-path checker decision. A certificate requires finite scores, ordinary-written < ordinary-rival, and latent-written > latent-rival. It matches the word, both pronunciations and both frame coordinates exactly. The checker retains baseline internal issue ownership and changes only the public verdict; graph confirmations remain exempt. Primary phones, posteriors, tracker input and insertion decisions are retained. Audio sessions apply the retrospective review at completion and retract only the same live event removed by an otherwise identical certificate replay.

The optional Node loader checks the primary SHA, inventory, clock, both declared side files and their hashes, projection dimensions, positive observation covariance, Gaussian normalization and emission geometry. A declared missing or corrupt bundle fails closed. The CLI's `--verify` mode loads the complete bundle before it can be declared for scoring.

## Limits

The two views share the recording's native noise. The estimator identifies additional independently injected feature variation; it cannot identify native noise already shared by both copies. Nonlinear encoder bias and temporal correlation also violate the simple independent zero-mean Gaussian approximation. The training report explicitly records the average replicate-versus-unaugmented feature bias rather than labelling it removable noise. Whether this approximate model corrects the four warnings is unknown.

The session path buffers audio and reviews at `end()`; it has not established a streaming warning-latency improvement. A written preference can still suppress a genuine substitution. The synthetic Gaussian tests and counterfactual references do not prove safety on learner mistakes, clean sessions or studio audio.

## Validation

- Python AST parsing passed for the exporter, fitter and numerical tests.
- Node's TypeScript syntax checks passed for 14 modified/new files. This is not TypeScript typechecking or a build.
- A lightweight Node check passed for independent additive copies: deterministic seeds, distinct views, unchanged source, identical length, finite samples, preserved impulse onset and silent-input preservation.
- Unit tests were added for Gaussian conditioning and constrained variance recovery, replicate-noise estimation, quantized-head geometry, complete-boundary probabilities and occupancies, strict same-feature reversals, exact-window certificates, graph preservation, session verdict handling and incomplete asset rejection. Neither the numerical suite nor the core/SDK suite executed.
- Supervisor run 1 requested a read-only help-train/model audit. It returned `NotFoundError: No such file or directory`, zero compute usage and no shard logs.
- Supervisor run 2 requested the prescribed six-shard `bash tools/autolab_tests.sh`. It returned the identical error, zero compute usage and no shard logs; build and unit tests did not execute.
- Supervisor run 3 requested the Python numerical tests. It returned the identical `NotFoundError`, zero compute usage and no shard logs; the numerical suite did not execute.
- Supervisor run 4 requested only `python3 -c "print(12345)"`, independent of data and implementation. It also returned the identical `NotFoundError`, zero compute usage and no shard logs. All four requests failed before execution, establishing an authorized execution-service blocker.

No supplied manifest, test data, evidence, experiment card, scorer or lab tooling was edited. Practice scores used: 0 of 3. No `.autolab/model.json` is present; running the primary model without the fitted scorer would not test this experiment.

## Resume after executor repair

Use only the authorized supervisor. First rerun the six-shard core/SDK suite and `python3 -m unittest discover -s modal -p test_student_deconvolved_phone.py -v`. Audit help-train labels and exclusions, then copy the complete unchanged primary directory, including inherited teacher files and external data, to a persistent experiment bundle. Export the diagnostic graph into that bundle, prepare observations with the SDK CLI's `--labels`, `--extractor` and `--excluded-speakers` arguments, and fit with the Python utility using the same exclusion list. Verify the completed bundle with the CLI's `--verify` mode before declaring `.autolab/model.json`.

Inspect all four warnings with the generic CLI's `--audio` and `--evidence` mode; labels may identify report rows only, never inference windows or pronunciations. Revert the engine implementation and record a scientific kill if fewer than three target warnings are removed. Otherwise run the authorized practice score and let the fold gate evaluate losses. No constants were selected against practice incidents.

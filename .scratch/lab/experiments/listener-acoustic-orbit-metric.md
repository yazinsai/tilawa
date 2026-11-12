# listener-acoustic-orbit-metric

Status: needs-info

## Prediction

The immutable card predicts two removed false substitutions and one lost caught substitution: listener −1, overall −1. Its mechanism is a recording-dependent, curved acoustic distortion orbit. The user's final stopping rule requires at least three of the four target warnings to be removed before a practice score is required; unavailable measurements do not establish a scientific kill.

## Implementation

The primary encoder, greedy decoding, tracker input, existing string support requirements, graph confirmations and insertion decisions are retained. A separate extractor exposes the effective input to the deployed phone projection, including dynamic activation quantization. Its export checks the primary model checksum, exact equality of all primary outputs and recurrent states over eight streaming windows, and reconstruction of logits from the exposed effective features.

Only usable `help-train` takes labelled `none`, with `split=train`, an identified speaker, no scripted events or repeats and no exclusion, supply canonical supervision. Any speaker with a non-training take is excluded as a whole, along with the supplied excluded-speaker list. Canonical pronunciations come from the existing reference index and the training take's authorized ayahs. Practice labels and recordings never enter extraction, fitting or model selection.

Each recording supplies its original features and three draws of `student_train.spec_augment`. This uses an existing training augmentation, retains native frame coordinates and allows the same acoustic occurrences to be paired across all four views. It does not invent another simulated phone channel. These masks can remove phonetic identity cues, so their suitability for actual distortion orbits remains an empirical question.

A shared twelve-dimensional PCA projection uses one scalar scale, preserving Euclidean geometry within the retained subspace. Sixteen acoustic neighborhood anchors support a smoothly interpolated positive definite metric field. The fit minimizes matched-occurrence orbit distances, penalizes collapse between distinct transcript-supported phones and regularizes each metric toward identity. The inexpensive directions depend on the query's acoustic neighborhood, rather than on the candidate pronunciation. The controls are Euclidean distance and a single globally whitened residual metric. All three share projections, phone prototypes, frame windows and timing topology. Two shared prototype refits marginalize complete pronunciation boundaries over the matched views and the average of the three geometries.

Three speaker folds rebuild projection, anchors, geometry and prototypes within their training speakers. Only these folds choose metric regularization from the fixed candidates 0.1, 1 and 10. Counterfactual lexical contrasts are training-only preservation diagnostics. Extraction and fitting enforce a 23-hour budget, retaining the original budget start across repeated exports.

The checker exposes the exact existing complete written and accused pronunciation comparison only for already supported string-path substitutions. The SDK scores all three geometries over those identical frames and marginalizes complete CTC boundary paths, including required blank separation for repeated labels. A certificate requires a strict written preference locally and strict rival preferences under **both** controls. Certificates bind the word, pronunciations and exact absolute frame bounds. Internal confirmations and ownership remain baseline decisions; only the public substitution verdict can be downgraded. Audio sessions publish matching downgrades at completion and retain certificates in recorded evidence. This is a retrospective recheck, with no claim of earlier live warning prevention.

The node loader fails closed on an incomplete declaration, corrupt file, incompatible clock or mismatched primary hash. The training recipe preserves the full primary bundle and its inherited assets, then validates the complete fitted bundle through the SDK before it could be declared for scoring. No `.autolab/model.json` is declared before that succeeds.

## Validation and execution blocker

Implemented tests cover local metric direction changes, strict agreement with both controls, exact window length, complete pronunciation paths, exhaustive boundary marginalization, primary token/posterior preservation, training speaker exclusions, invalid assets, exact-window checker vetoes, graph protection and session downgrades. A small optimizer test checks positive definite learned metrics and different-phone preservation.

Local checks completed:

- Python source compilation and shell recipe syntax pass.
- Four mathematical Python tests pass, covering exhaustive boundary paths, repeated-phone separation, neighborhood-dependent geometry and independent streaming cache/clock handling. The optimizer test is skipped because torch is unavailable locally. These small checks ran under the installed x86 Python because its NumPy extension does not load in the native arm process.
- Eleven changed TypeScript source/test files parse through Node's TypeScript stripper. This is a syntax check, not type checking.
- Actual SDK numeric functions pass isolated checks for a local preference reversal, both-control requirements, complete paths, neighborhood direction changes and exact frame length. Package imports were omitted for this isolated check; it does not validate integration.
- `git diff --check` passes.

The authorized supervisor returned `NotFoundError: No such file or directory` before executing remote requests 1, 2 and 3: the initial data/dependency audit, prescribed six-shard tests, and a second minimal audit. Requests 4 (complete bounded training recipe, 60-minute budget) and 5 (updated six-shard tests) also returned the same error before execution. All five requests failed; none produced remote command output, training artifacts or test results. The execution blocker was raised to the user through an asynchronous clarification.

No training audio has been inspected, no fit or feature parity has been measured, and no complete model bundle is available or declared. No target windows have been evaluated, and no practice-score calls have been made. Targets won, listener change and overall change are **unknown**, not zero. The premise is neither supported nor falsified. Infrastructure failure is not a scientific kill under the user's stopping rule.

## Limits and next required work

Restore the authorized supervisor execution path, run the full recipe and required core/SDK tests, inspect training fold reports and verify the complete bundle. Then compare the four target incidents through the SDK's ordinary exact-window comparison route. Kill only if the built mechanism fails the user's target requirement or its premise is wrong; if at least three targets are removed, declare the verified bundle and run the authorized practice score before judging other losses. Training masks, sparse none-labelled coverage, projection loss and boundary modelling could suppress genuine substitutions despite different-phone contrast constraints. No gain or safety claim is made here.

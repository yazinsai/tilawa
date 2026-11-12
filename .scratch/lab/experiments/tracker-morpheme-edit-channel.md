# Morphology-structured local observation channel

Status: needs-human. Prototype implemented and inactive by default; Modal training and validation were blocked before execution. No measured kill or fold claim.

## Prediction

Card `tracker-morpheme-edit-channel`, tracker, explore. Practice `lab-v1.5u`, listener `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. Predicted tracker −1, overall 0: correcting location may expose the documented nearby listener warning.

Target: `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`. The card reports the retained near continuation only 0.3915 edits behind the rhyme restart. That number is prior evidence, not a measurement reproduced here.

## Mechanism implemented

`morpheme-channel.ts` partitions the existing connected surface pronunciation into attached-prefix, stem and pronoun-suffix slices. It retains every phone, including article assimilation and duration-expanded runs. A text-derived vocabulary supports conjunction removal; unresolved analyses remain stems. This is a conservative clitic analysis, not a complete morphological lexicon.

The observation transducer carries the previous reference edit (correct identity, substitution, deletion) across those slices. The next edit has a normalized conditional distribution tied to adjacent phonetic classes, repeated-phone context, and, in the morphological model, the transition between clitic roles. Consequently, errors on either side of a unit boundary are scored jointly. Insertions consume new observations through a terminating geometric loop. Substitution distributions exclude canonical identity, and deletions retain positive probability costs. There is no freely deletable morphological unit.

Forward/backward estimation sums uncertain token placement across the full training transcript; it does not supervise boundaries from tracker visits. The estimator accepts only `none` recordings explicitly assigned to training speakers, excluding labelled events, repeats and excluded takes. It rejects mixed speaker partitions and missing evidence. Six EM iterations, three grouped speaker folds, and prior strengths {2, 16, 128} define the training-only selection recipe. Both models receive the same recordings, phone representation and fitting procedure. The phonetic control omits morphological roles.

In `local.ts`, an optional frozen channel pair scores the baseline-winning restart and best continuation with the prior run origin over the same complete heard interval. Each trajectory retains its traversed surface-phone occurrences; whole-word movement bypasses are represented by their existing movement charges. The additional cost is

`C_original − (UNIT / 3) × [log P_morphology(heard | trajectory) − log P_phonetic(heard | trajectory)]`.

Thus the matched phonetic channel is the observation control, while original movement, whole-word bypass, phone and duration edit prices stay in `C_original`. Commitment margins, event delivery and checker handling are unchanged. The optional auditor exports the two paths and their common observations without affecting decisions. Rewind snapshots include the diagnostic histories.

`morpheme-channel-fit.ts` is a training-only estimator, and `morpheme-channel-audit.ts` replays a supplied evidence file against a frozen channel pair. Neither changes scorer code, labels or evidence. No trained model was generated or enabled.

## Validation completed

Lightweight checks ran locally with Node's TypeScript stripping and built-in assertions; these are not substitutes for the requested Vitest suites.

- The complete shipped reference retained all 643,312 surface phones with one role per phone: 26,723 prefix, 581,079 stem, 35,510 suffix. These counts verify ownership, not the linguistic correctness of every analysis.
- Exhaustively summing observation strings through length nine in a two-phone fixture gave probability 0.9999999999989985, including charged empty realizations.
- The phonetic control was invariant to changed morphological roles; the joint model changed the probability of a two-sided boundary error.
- Small synthetic fits were deterministic, conditional rows summed to one, grouped selection completed, and missing roles/unknown phones were rejected.
- An identical channel pair preserved local positions through a synthetic jump and state rewind.
- An injected normalized channel reversed a synthetic restart while retaining its 20.006-unit movement charge. This verifies the integration mechanism; it is not a trained result or a target win.

Nine Vitest unit tests encode these behaviours, including surface preservation, intrinsic onsets, local controls and reversal.

## Modal blocker

The authorized supervisor returned `NotFoundError: No such file or directory` for all three requests, with zero reported compute cost:

1. Run 1: inspect training labels/evidence paths and storage.
2. Run 2: six-shard `bash tools/autolab_tests.sh` for core/SDK checks.
3. Run 3: inspect training data, run the new focused unit tests, and fit the channels.

No remote command output, fitted probabilities, target replay or test result was returned. Practice scores used: **0 of 3**. Targets won: **unmeasured**. Overall/component changes: **unmeasured**. The ordering kill criterion and the fewer-than-three-wins rule cannot be evaluated from an infrastructure error.

## Remaining work and limits

Restore the supplied supervisor runner, verify the training evidence's deployed-listener provenance and filenames, and fit both channels with the grouped selection recipe. The fit runner currently expects evidence beside each training WAV. Audit the frozen pair on the target's actual paths and ordinary-reading controls, then run the full core/SDK checks. The prototype retains complete diagnostic trajectories and also needs memory/performance validation before activation in long streams.

If the frozen morphological channel does not reverse the actual continuation-versus-restart ordering relative to the matched control, remove the engine prototype and record a measured kill. If the built replay wins fewer than three target incidents, apply the user's kill rule. A qualifying result must receive the authorized practice score before any gate decision. No constants have been selected against practice.

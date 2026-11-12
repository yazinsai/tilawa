# Smoothed location-belief distillation

Status: needs-human. A disabled, opt-in prototype is implemented. The supplied supervisor failed before executing the data probe, training job and all three test requests, including the final sources. No model has been fitted, no calibration is frozen, and no target or practice result is measured.

## Prediction

Card prediction: tracker −1, overall 0 on lab-v1.5u with listener `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. The target is `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`. A corrected route can expose the documented nearby listener warning; no checker safeguard is added to hide it.

The supplied audits say the near continuation remains available, loses by 0.3915 edits, and has previously gained support from later continuation. Those findings motivate distillation, but were not remeasured here. No training data or target evidence was returned by Modal. The card was not edited.

## Implemented mechanism

`location-belief.ts` supplies a small recurrent candidate scorer. Its 12 inputs describe relative edit cost, insertion/substitution/deletion prices, word progress and length, relative movement, predecessor displacement, and the winning edit transition kind. There are no absolute reference coordinates, verse, session, or reader identities in its input or weights.

The student has eight hidden units and 184 parameters. A candidate's hidden state follows its winning predecessor from the **previous observation**, including through a local jump. Reference-only deletion closure propagates that previous-observation predecessor instead of consuming the observation again. Local origin changes do not reset memory. Overlapping local-window recentering resolves parents by absolute cell coordinates, and saved search states restore the corresponding immutable memory arrays. A new globally acquired local search starts a new memory; this experiment does not carry uncertain localization through global reacquisition.

In `local.ts`, the original edit recurrence, jump edges and prices remain intact. Candidate selection uses raw edit cost minus a training-calibrated logit contribution. Raw costs still supply the existing local fit signal. `tracker.ts` passes an optional model to that search. Global commitment margins, event handling and checker code are unchanged. There is no inference lookahead, token rebind, coverage witness, or new warning rule.

`LOCATION_BELIEF_MODEL` is null. No untrained weights or guessed score conversion are active by default. `TrackerOptions.locationBelief` permits an explicitly supplied frozen model for validation. Reset-state and nonrecurrent controls are available to the diagnostic replay.

## Training-only implementation

`belief-build.ts` reads only `help-train/labels.jsonl` and the available student bundle. Eligibility requires explicit `split: train`, `kind: none`, no planted events, repeats, unsupported labels or exclusions. A speaker with any non-training-labelled take is excluded as a unit. Audio paths must remain within help-train. Decode caches are keyed by audio and listener hashes under this experiment's scratch directory.

`belief-smoother.ts` performs forward-backward summation over complete token-to-reference edit paths with the existing weighted phone costs. The known source pronunciation constrains supervision. The ending is constrained to the complete source; latent word boundaries and edit placements remain distributions. This is a normalized conditional edit-path posterior, not a separately trained acoustic teacher.

Each accepted recording produces an ordinary-reading control. A hash chooses an internal word for generic training-only omission and repeat splices. Three fixed quantiles sample its start and end from the smoothed boundary distributions; each spliced stream retains its known source-word sequence. Adjacent available ayah recordings from the same training speaker can form longer streams. All resulting streams are smoothed again. The export maps soft source-phone occupancy into the unchanged local search's candidate cells, without using committed tracker visits as labels. It reports excluded streams when less than 95% of mean teacher mass is reachable within that search window.

`belief-fit.py` distills the teacher targets into the recurrent student and a nonrecurrent MLP with exactly the same parameter count, features, streams and supervision. The control uses its square hidden matrix on an instantaneous hidden layer instead of prior candidate memory. The Python and TypeScript equations agree by construction, with a dedicated parity test.

All augmented streams stay in their source speaker's fold. Three speaker-separated folds select between two predeclared regularization strengths. The logit-to-edit scale is selected from a predeclared grid, including zero, using out-of-fold location cross-entropy, subject to ordinary-reading expected position error not worsening from the baseline. Every fold must contain ordinary controls. Final models are refitted on the accepted training speakers. The output records both models, selection results, folds, hashes, exclusions and source provenance. No practice input or score is accepted by the fit.

`belief-audit.ts` accepts evidence, model, time and competing word keys as diagnostic arguments. It replays baseline, recurrent, reset-state and nonrecurrent modes through the existing tracker/checker and reports candidate rankings and confirmed warnings. Those case arguments do not enter training or the engine's model features.

## Validation completed

- Strict TypeScript 5.9.3 checks passed for the changed tracker, local search and belief module and their imported engine dependencies. The pure training smoother and eligibility module were also checked separately.
- Dependency-free Node smoke checks passed against the actual source. An independent exhaustive enumeration of every edit path in a tiny graph matched the smoother's posterior numerically. Location and boundary distributions normalized.
- Smoke checks established carried memory across overlapping windows, immutable rewind, eligibility exclusion, and position parity between the original HEAD local search, audit mode and a zero-scale model.
- A separate synthetic local-search check produced an actual run-origin restart. Memory survived it; resetting memory removed that history while preserving the zero-scale path.
- Python syntax compilation passed for the fitter and its unit tests. PyTorch execution remains unavailable locally and was included in the failed Modal training request.
- Vitest tests were added for these behaviors, prefix causality, model validation, and the instantaneous control. The full core/SDK suites have not executed successfully.
- `git diff --check` passed. No scorer, lab tooling, test data, evidence, manifest, or experiment-card file was changed.

Supervisor requests:

1. Storage and help-train file probe: failed before execution with `NotFoundError: No such file or directory`.
2. Six-shard `bash tools/autolab_tests.sh`: failed before execution with the same error.
3. Updated-source six-shard tests: failed before execution with the same error.
4. Build, PyTorch unit tests, authorized decode/export, speaker-fold fit and calibration: failed before execution with the same error.
5. Final-source six-shard tests: failed before execution with the same error.

No Modal log, training schema, accepted recording count, fitted weights, calibration, target trace or suite result was obtained. No practice score was requested: 0 of the permitted 3.

## Limits

The implementation is unvalidated against the real help-train schema, recording distribution and runtime candidate distribution. Its generic splice builder uses the available none-labelled recordings rather than importing any other training corpus. Multi-ayah training coverage depends on adjacent takes actually being available. Quantile-based splice construction approximates boundary uncertainty; the teacher for each resulting stream fully marginalizes its edit alignment, but the construction does not integrate every joint pair of splice boundaries.

The recurrent state follows minimum-edit predecessors from the original candidate graph, rather than retaining all hidden histories inside a recombined cell. Training exports come from baseline trajectories seeded at transcript onset; runtime starts at acquired endpoints and can visit different states. Retained-mass exclusion can bias training away from difficult misbindings. Speaker-fold calibration constrains expected location error on ordinary training controls, not full-engine false warnings on long clean sessions. Enabled-model latency, long-session behavior, studio performance and downstream warning changes are unmeasured.

The code does not establish that future-informed teacher structure is predictable from the target prefix, that recurrence improves over the equal-size control, or that resetting memory changes the target ordering. Infrastructure failure is not evidence for any of those premises failing, and is not a measured zero-win result.

## Recovery and decision

After the supplied supervisor is repaired, rerun the final suites and training pipeline:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- \
  'bash tools/autolab_tests.sh'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && python3 -m pip install --quiet numpy && python3 -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 && python3 packages/core/train/test_belief_fit.py && BELIEF_CACHE=/scratch/tracker-smoothed-belief-distill/belief-cache pnpm --filter @tilawa/core exec tsx /app/packages/core/train/belief-build.ts && python3 packages/core/train/belief-fit.py --training /scratch/tracker-smoothed-belief-distill/belief-cache/training.json.gz --out "$LAB_OUT/belief-frozen.json"'
```

Use the returned frozen models with `belief-audit.ts` to inspect the target ordering, compare the instantaneous control and reset ablation, and check the final checker warning. If the trained recurrent model still prefers the rhyme restart, resetting memory leaves the ranking unchanged, or the built change wins fewer than three targets under the user's decision rule, revert engine changes and commit the measured premise kill. The one listed target cannot by itself establish three wins; synthetic checks are not target recoveries.

If target validation clears that rule, install the unchanged training-frozen recurrent weights as the default, run the final checks, and request a practice score before any gate-dependent narrowing. Do not adjust weights or calibration against practice. Until those steps succeed, this is a recoverable disabled prototype, not a validated experiment or a fold candidate.

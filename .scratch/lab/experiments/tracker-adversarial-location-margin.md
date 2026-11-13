# Adversarial location margin

Status: needs-human. The scorer, authorized-training exporter, adversarial fitter, matched random-corruption control and fixed-interval audit are implemented. The default model is null: no authorized training run completed. All six supplied supervisor requests failed with `NotFoundError: No such file or directory` before execution. This is an infrastructure block, not a measured premise kill.

## Prediction

Card: tracker −1, overall 0, lab-v1.5u, listener `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. Target: `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.

The supplied history says the near continuation remains available and loses by 0.3915 edits. Audits found no discarded candidate or missing cost. Earlier corrected bindings sometimes removed the tracker warning while exposing a nearby listener wrong-word warning. Those are supplied findings, not measurements made in this experiment. The card is unchanged.

## Mechanism

`trajectory.ts` defines a 24-feature linear correction to local trajectory costs: edit kind, acoustic price, shared phonetic classes, exact/soft agreement, duration-run membership, word-boundary structure and relative movement interactions. It contains no recording, reader, word identity or absolute reference-location feature. Its engine foundation reuses the existing `exp/tracker-conditional-trajectory-training` prototype; this experiment supplies a different training objective and corruption procedure.

`local.ts` adds the correction to its existing insertion, consumption, word-start restart and reference-only deletion arcs. All existing movement edges, restart sources, entry prices, window sizes and tie rules remain. A separate raw-cost array follows the selected trajectory, so recent-fit checks receive the original acoustic/movement prices rather than learned biases. Saved states include both arrays, and updates replace arrays so rewinds remain valid. Null and zero corrections reproduce the original search.

`tracker.ts` accepts a frozen `locationModel` and passes it to local comparison. Global acquisition, commitment margins, confirmation thresholds, event delivery, window construction and checker code are unchanged. `location-model.ts` remains null until an authorized training fit is available. No synthetic or guessed weights are deployed.

The exporter uses only none-labelled recordings of explicitly training-labelled help-train speakers. Any speaker with a non-training take is excluded as a unit. Unsupported, excluded, unscored and contradictory none labels are rejected. Audio paths must stay under help-train. Decode caches are scoped to this experiment and keyed by audio and listener hashes. The payload records the label, reference and listener hashes and every source recording.

Full-transcript minimum-edit alignment supplies training word boundaries, independent of tracker visits. Four kinds of known-location prefixes are built: unchanged readings, one-word omissions, internal multiword deletions and cross-ayah passage jumps. The last concatenate a source tail with a same-speaker destination two ayahs ahead, leaving an entire ayah unread while keeping the destination inside the existing local search. These are token splices of actual decoded training recordings, not new practice data. Both source hashes accompany cross-recording examples. Every training and validation partition must contain clean readings and both kinds of positive movement; fitting refuses to silently waive passage-jump supervision.

`location_margin.py` implements the same local recurrence, prices, restart source and recentering for fitting. Its structured margin compares the best true-word state with the best incorrect-location state reached through an existing restart. The complete prefix is scored on both sides. The gradient is the difference between the selected paths' accumulated features; the loss asks the true location to beat its competitor by the existing observation-dependent commitment margin.

Recognition-error support comes only from none-labelled recordings in the current training partition. A transcript alignment estimates substitution, insertion and deletion patterns. At least two observations must support a pattern, and deletions inside repeated duration symbols are excluded from this support. Insertions retain their observed reference-phone context. The median observed error rate sets a prefix's edit budget, with a predeclared two-edit cap.

The inner search evaluates eight proposals at each edit depth and selects the observation maximizing the incorrect restart's advantage. The control evaluates the same proposal budget but selects randomly. Both arms use the same recordings, examples, feature schema, optimizer, three epochs, norm bound and shared regularization configuration. Score-evaluation and update counts must match; random selection uses a separate random generator. Insertions/deletions and recentering can still produce somewhat different per-evaluation work, so these counts do not establish identical wall time or identical cell-operation counts.

Three speaker partitions freeze selection before any practice input is accepted. Corruption statistics exclude each validation fold. Validation damage is generated with the baseline model and frozen before fitting either arm, giving both models identical held-out training observations. Shared L2 selection uses robust validation loss and requires no worse original-reading gaps on clean, internal-jump and passage-jump controls in both arms, with unchanged reachability counts. If no configuration improves robustness while passing those controls, both frozen outputs have zero weights. An interrupted fit publishes no model.

To bound one supervisor fit, optimization and validation each select at most six intervals per kind, round-robin across speakers using deterministic source hashes. The final refit similarly uses at most 24 intervals. Corruption estimation uses all accepted none-labelled recordings. This is a deliberately small optimization sample; the frozen artifact reports the full and optimized counts. These limits are defined before practice inspection.

`location-audit.ts` accepts a frozen artifact, evidence path, time and competing word keys as diagnostic arguments. It captures the baseline local initialization and entire consumed prefix, then replays adversarial and random models from that same initialization on that same prefix. It rejects an ambiguous non-initial seed. It reports both competing words' finite cells and costs around the requested time, then separately replays the unchanged tracker/checker product for baseline and both models. Diagnostic keys never enter fitting or engine features, and instrumentation is a temporary output file rather than a lab-tool modification.

## Validation completed

- Eight new TypeScript unit tests passed: null/zero search and whole-tracker parity, rewind/recenter behavior, retained restart capability, relative-coordinate feature invariance, schema rejection, caller-weight snapshotting and training-label/path handling.
- Seven Python tests passed: supported corruption operations, training-only channel estimation, exclusion of duration-repetition deletions, adversarial versus random selection with equal evaluation counts, finite-difference margin gradients, bounded/timeout-aware optimizer behavior, deterministic speaker-balanced sampling and cross-language parity. The cross-language check is one test within that count; some tests check multiple behaviors.
- The Python recurrence matched every emitted prefix of eight actual TypeScript local-search cases, including nonzero corrections, duration-expanded sounds, omissions and multiple window recenterings. Candidate costs, selected positions and selected origins matched.
- The existing 12 local-tracking tests and 14 issue-drop-fit tests passed locally. These include skips, repeats, silence, streaming equivalence and long-session behavior. Full core and SDK suites did not execute on Modal.
- Strict TypeScript checks passed for the changed engine, new test, pure data helper and parity exporter. The native-dependent training exporter and diagnostic audit also type-checked with the existing SDK/core source alias and an opaque declaration for `onnxruntime-node`; this does not validate native inference.
- Python syntax checks and `git diff --check` passed. The local Python's NumPy binary has x86_64 architecture; the small numerical tests ran under `arch -x86_64`. Heavy decoding and fitting were requested only through the supplied supervisor.

## Supervisor requests and measurements

1. Storage/help-train probe: failed before execution with `NotFoundError`.
2. Six-shard core/SDK build and tests: same failure before execution.
3. Retried probe using a checked-in script: same failure before execution.
4. Build, focused tests, cross-language parity and authorized training export: same failure before execution.
5. Retried six-shard core/SDK build and tests: same failure before execution.
6. Final-source six-shard core/SDK build and tests: same failure before execution.

Practice score requests: **0 of 3**. Training recording counts, fitted weights, target ordering, target wins, component/overall changes, clean-session warnings and studio changes are **unmeasured**. No killed outcome is written because infrastructure failure does not establish the premise or a zero-win result.

## Limits and recovery

The exporter has not encountered the actual available labels or audio here. Cross-ayah examples and corruption support may be too sparse in individual speaker partitions, in which case the fitter will refuse to proceed. Transcript-alignment boundaries have uncertainty; the structured loss does not marginalize their labels. Token-spliced movement examples do not model the decoder's reaction to new audio seams. None labels are weak recognition-error supervision, not independently verified spoken phones. The optimization sample is small, and the model is linear. Candidate feature accumulation uses selected minimum paths, with nonsmooth changes when paths exchange order. Enabled-model runtime and resource use remain unmeasured.

The control checks assess training location gaps rather than actual warning behavior. They cannot establish the clean-session gate or prevent the documented exposed listener warning. The default null scorer establishes no target improvement and is left in place as an implementation awaiting authorized training, not as a completed, deployable trained experiment.

After the supplied supervisor is repaired:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- 'bash tools/autolab_tests.sh'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- \
  'bash packages/core/train/location-validate.sh && LOCATION_CACHE=/scratch/tracker-adversarial-location-margin/cache pnpm --filter @tilawa/core exec tsx /app/packages/core/train/location-build.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- \
  'OPENBLAS_NUM_THREADS=1 python3 packages/core/train/location-fit.py --training /scratch/tracker-adversarial-location-margin/cache/training.json.gz --out "$LAB_OUT/location-frozen.json" --minutes 50'
```

Use the returned frozen artifact with `location-audit.ts` on the available target evidence, passing the target time and near/restart word keys as CLI arguments. First compare the candidates on the fixed baseline interval, then compare confirmed warning streams. Do not change weights, budgets or regularization against practice. If the adversarial model still prefers the restart, adds no ordering improvement over the matched control, or the built change wins fewer than three targets under the user's rule, revert the engine changes and commit a measured premise kill with the required `.autolab/outcome.json`. If it passes that rule, install exactly the frozen adversarial model in `location-model.ts`, run final validation and request the permitted practice score before gate-dependent narrowing. The supervisor and fold gate decide whether to retain it.

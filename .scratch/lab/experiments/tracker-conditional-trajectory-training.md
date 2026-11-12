# Conditional trajectory training

Status: needs-human. The conditional trajectory prototype is implemented and disabled by default. All four supplied supervisor requests failed before execution, including the final-source Modal suites. No authorized recording, fitted model, target trace or practice result has been returned.

## Prediction

Card: tracker −1, overall 0 on lab-v1.5u, listener `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. Target: `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.

The supplied audits leave the near continuation available but 0.3915 edits behind the rhyme restart over their shared observations. No missing charge, boundary reset, candidate eviction or free prefix was established. This experiment tests whether full-route conditional supervision can learn the distinction that local generative costs did not. Correcting the binding can expose the nearby listener false warning; no checker rule is changed to hide that loss. These target facts came from the supplied history and were not remeasured here. The card is unchanged.

## Engine implementation

`trajectory.ts` defines 24 shared features: four edit-transition indicators; acoustic price; heard/written vowel, nasal and emphatic classes; exact/soft agreement; a reference duration-run indicator; word boundary and length; relative word/phone movement; ayah crossing; and movement–acoustic interactions. Absolute positions, word identities, verses, sessions and readers are absent from the model inputs. Coordinates exported for training describe graph geometry only.

`local.ts` retains its insertion, phone-consumption, word-start restart and reference-only deletion arcs, including the original restart source, distance price, surah entry price, tie handling and window recentering. A learned linear transition potential adds to each existing arc's edit-unit cost. The minimum complete trajectory supplies the selected position; alternative arcs remain available. Parameters are copied when a search starts, and rolling arrays remain immutable for rewind.

Raw acoustic costs follow that same selected trajectory separately from learned costs, so the existing recent-fit signal receives acoustic edit units rather than arbitrary learned biases. A null or zero model shares the original cost array and preserves positions and fit. The existing tracker commitment margins, confirmation thresholds, event/window rules and checker implementation are unchanged. No coverage certificate, warning veto, lookahead or replay safeguard is added.

`TrackerOptions.trajectoryModel` accepts a frozen model. `CONDITIONAL_TRAJECTORY_MODEL` remains null because fitting never ran. No guessed or synthetic-fit weights are deployed.

## Training objective and data construction

`trajectory-build.ts` reads help-train labels and the available student bundle. It admits explicitly training-labelled speakers only and excludes every speaker with a non-training take. Unsupported, excluded, unscored and lexical-substitution labels are omitted. Clean takes, scripted omissions/skipped passages and labelled word repeats supply ordered reference-word occurrences. Audio paths must remain within help-train; caches are scoped to this experiment and keyed by audio and listener hashes.

Clean recordings supply unchanged-reading controls. A hash selects an internal word for training-only omission and repeat splices. Full-transcript minimum-edit boundaries are used only to construct those splices. Multiword deletions supply genuine movement examples, and available same-speaker recordings can form adjacent clean streams or streams jumping over one ayah. Real scripted jump/repeat examples are distinguished in the export, if present. No practice audio supplies training splices. These augmentations splice decoded observations rather than rerunning an acoustic decoder on spliced waveforms.

The unchanged local search exports all four arc families for every observation, not just each winning predecessor. `trajectory_crf.py` computes:

`loss = log Z(all complete graph routes) − log Z(transcript-consistent complete routes)`.

Both partitions sum over full observation histories and latent edit placements. The numerator intersects the same graph with the ordered reference-phone occurrences in the known source transcript; repeat occurrences remain distinct, and skipped words can be traversed only through existing movement arcs. Insertions consume an observation; ordinary matches consume an observation and reference phone; deletion closure consumes reference alone. Complete transcript supervision fixes the numerator's terminal occurrence, while denominator endpoints retain prefix semantics. No per-word or per-observation conditional softmax replaces this objective.

Forward/backward deletion closure uses cumulative log-sums in linear time. Exact expected transition features provide the gradient. A separate max closure computes each partition's MAP score. A transcript route outside the unchanged graph is excluded and counted by kind, never supplied a new edge or remapped to a wrong location.

`trajectory-fit.py` fits the linear potentials with L-BFGS and selects between predeclared L2 strengths 0.1 and 1 through three speaker-separated folds. All augmentations of one speaker remain in that speaker's fold. Each fold must contain clean and movement controls. Selection requires lower held-out conditional loss and no worse mean MAP disadvantage for either control group; otherwise the frozen output has zero weights. The selected configuration is refitted on all accepted training speakers. Fitting has an explicit time budget and refuses to publish interrupted optimization. Output records folds, exclusions, accepted kinds, source provenance and label/listener/training hashes. Practice inputs and scores are not accepted by the fitter.

`trajectory-audit.ts` accepts the frozen model, evidence, time and competing word keys as diagnostic CLI arguments. It reports both candidates' accumulated cost differences over shared observations and confirmed checker events for baseline and conditional replay. Instrumentation is written to `LAB_OUT`, without editing a lab/scoring file. Diagnostic case keys do not enter the learned features.

## Validation completed

- Strict TypeScript 5.9.3 checks passed for the changed engine, pure training-data module and diagnostic audit. The exporter also type-checked with the native ONNX import represented by an unknown-value declaration; that check does not validate native inference. Training exporter/audit syntax and Python syntax checks passed.
- Eight new Vitest tests passed: zero/null whole-engine parity, local rewind/recenter parity, learned movement, feature coordinate invariance, schema validation, caller-weight snapshotting, speaker eligibility, and omission/repeat occurrences.
- Five Python tests passed. Exact log partitions, MAP scores and gradients matched exhaustive enumeration and finite differences; omission routes used existing jumps; missing arcs made supervision unreachable; repeats retained separate occurrences; recentering introduced no new source states.
- The Python MAP recurrence matched a graph emitted by the actual TypeScript search. Its transcript partition was a reachable subset of that graph.
- The original HEAD local search and the final disabled prototype returned identical positions and costs through a 700-phone stream containing an omission and repeat.
- Existing local-tracking tests (12) and issue-drop-fit tests (14) passed locally. Three hold/excursion tests initially hit Vitest's five-second timeout; rerunning those two files with a 20-second timeout passed all six tests. No assertions were relaxed or test files changed for that rerun.
- A six-stream, three-speaker synthetic fit completed cross-validation, control checks, final refit and serialization. This checks optimizer plumbing only; its numbers and weights are not an experiment result or deployment model.
- `git diff --check` passed. No scorer, lab tooling, data, evidence, manifest or card was edited.

The default local Python imported incompatible NumPy binaries; the small numerical tests passed under the installed x86_64 interpreter via `arch -x86_64`. Heavy training was requested only through the supplied Modal supervisor. Full core/SDK suites did not complete there.

## Supervisor requests and measurement

1. Storage/help-train probe: `NotFoundError: No such file or directory`, before execution.
2. Six-shard core/SDK build and tests: same failure, before execution.
3. Build, numerical tests, authorized decode/export and speaker-fold fit: same failure, before execution.
4. Final-source six-shard core/SDK suites: same failure, before execution.

Practice scores requested: **0 of 3**. Target wins, component/overall changes, clean-session warnings and studio changes are **unmeasured**. Infrastructure failure is not a premise kill or a measured zero-win result.

## Limits

Training-label compatibility, accepted recording counts, real scripted-jump coverage, candidate reachability and speaker-fold balance are unvalidated. Entire omitted-ayah takes with no remaining spoken path cannot train a local movement decision without surrounding context. Splice boundaries use a single alignment, although the training loss subsequently marginalizes edit placement. The feature model is linear and has no additional cross-word latent state; passage-level supervision comes from its globally normalized graph objective.

Training graphs freeze the baseline tracker’s causal restart sources and recentering. The enabled model can visit different sources at inference; training does not regenerate graphs from those induced states. Transcript-onset training also differs from runtime local searches seeded by global acquisition. Reachability exclusions may bias the fit toward easier recordings. MAP control gaps constrain training-route competition, not downstream warnings on fully clean sessions. Enabled-model latency, memory, long-session behavior and the large export's resource use remain unmeasured.

The prototype establishes no claim that the learned model reverses the target ordering, removes its warning, avoids the exposed listener warning or reaches the fold gate.

## Recovery

After the supplied supervisor is repaired, run final suites and export authorized training observations, then fit in a separate run so decoding does not consume the fitting timeout:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- 'bash tools/autolab_tests.sh'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && TRAJECTORY_CACHE=/scratch/tracker-conditional-trajectory-training/cache pnpm --filter @tilawa/core exec tsx /app/packages/core/train/trajectory-build.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- \
  'python3 -m pip install --quiet numpy scipy && python3 packages/core/train/test_trajectory_crf.py && python3 packages/core/train/trajectory-fit.py --training /scratch/tracker-conditional-trajectory-training/cache/training.json.gz --out "$LAB_OUT/trajectory-frozen.json" --minutes 50'
```

Use only that frozen output with `trajectory-audit.ts` on the available target evidence. Compare the near continuation and rhyme restart over their complete shared interval, and check the final warning stream. Do not adjust weights or regularization against practice. If the trained model still prefers the restart, or the measured change wins fewer than three targets under the user's rule, revert the engine changes and commit the measured premise kill. If it clears that rule, install the unchanged frozen weights, run final validation and request a practice score before any gate-dependent narrowing. The supervisor and fold gate decide the release outcome.

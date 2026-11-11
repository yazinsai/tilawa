# Recurring-word acoustic occurrence memory

Status: needs-human — prototype implemented, but supervisor recovery is needed for real-data calibration, target validation and full Modal checks. The default calibration is empty, so the product does not apply an untrained likelihood scale. This is not a measured premise kill.

## Prediction

Tracker −1, overall 0 on lab-v1.5u practice, targeting
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
The nearby continuation can expose the documented listener warning at `54:37:2`.
The supplied audit's 0.3915-edit restart advantage is the motivation, not a result reproduced here.
`lab-experiment.json` is unchanged.

## Implemented mechanism

`trackerInput` transports posterior rows in `occurrencePosteriors`, separately from the acquisition `posteriors` field. Startup and global product scoring retain their decoded-only inputs. A tracker with a calibration or explicit audit option creates a session-local `WordOccurrenceMemory`; otherwise it allocates no occurrence memory.

Admission uses actual committed word entries on one contiguous run. The word and its two neighbours must have single, positive-fit locked candidates, with no hold, relocation or unresolved parallel-span hold. Each must also acquire the existing tracker separation margin against every different reference word in the raw local DP. Separation can arrive inside the word: demanding it on its first phoneme would reject ordinarily ambiguous onsets. A complete middle-word trajectory is saved only after the right neighbour obtains this support. Interruptions clear pending bracketing, retaining already established exemplars.

The saved interval runs from the word's first observed entry phoneme to its successor's entry. It retains blank and sub-blank posterior classes and their temporal order. Memory owns snapshots and keeps two earlier complete renditions per connected pronunciation; verse location establishes adjacency but is not the memory key or a learned feature.

At an original local run-origin change, the existing best continuation competes with the existing best restart. The continuation must refer to the word whose observed entry bounds the current disputed interval; another word without its own boundary retains raw scoring. Both pronunciations need exemplars ending at or before that interval's onset. The entire query interval is retained even if its own commit is unclear, but unclear words cannot supply future exemplars.

Comparison uses squared Hellinger frame distance and subsequence DTW. The complete query is consumed, while an exemplar's prefix and suffix may remain outside the alignment. The distance is cumulative cost divided by the selected path's length, with exact cumulative-cost ties favouring shorter paths. Both candidates receive the same query frames. Their mean distances to earlier renditions yield a calibrated relative log likelihood. The existing conversion of 3 nats per plain edit expresses this contribution in local matching units.

This is a replacement ranking comparison on the current interval, not an additional token-support sum. Original edit and movement charges remain in the DP and continue to govern fit and landing confirmation. No movement edges, commitment margins, checker windows or warning rules are edited. No likelihood, missing history or missing rows retains the original selection. Phrase cancellation and a new local search clear pending occurrence state.

## Training and freeze

`packages/sdk/src/recurring-word-fit.ts` reads only `LAB_DATA/help-train/labels.jsonl` and audio canonically inside that directory. It accepts explicitly training-split, scored, none/repeat takes with no mistake events or exclusions; non-error annotations must explicitly be benign repeats/restarts. A speaker with any non-training row is excluded entirely. Audio, selected-label, reference and listener/model hashes are recorded.

The shipped listener decodes streaming chunks. The unchanged tracker runs in read-only memory audit mode, providing actual bracketed occurrences rather than gold-timed exemplars. Calibration uses other takes by the same speaker or fully earlier, non-overlapping intervals in a repeat recording. It requires earlier examples for both the correct pronunciation and a phonologically close competing pronunciation. Query frames, overlapping intervals and later intervals from the same recording cannot train their own comparison.

One nonnegative logistic likelihood scale is fitted. Four L2 prior variances are selected entirely by held-speaker training NLL, keeping every take and exemplar of a speaker in one fold. Streaming prefixes and full words share the scale. Calibration aborts if at least two training speakers do not supply suitable pairs; it supplies no placeholder fit. The selected scale is refitted on the authorized training speakers and exported with provenance. `recurring-word-model.ts` remains `null` until that output and the target premise are validated.

## Validation

- All 21 new core/SDK test bodies passed locally using Node 22's TypeScript transformation and an assertion adapter for Vitest. These are source-level assertion results, not a full Vitest suite or TypeScript build.
- Checks include temporal/sub-blank discrimination, duration stretching, complete query consumption, immutable snapshots, delayed three-word support, interval exclusion, held-speaker isolation, same-recording repeat eligibility, missing-history fallback, an actual restart reversal, and identical tracker events in read-only audit mode while collecting complete trajectories on a synthetic reading.
- TypeScript syntax transformation passed for the new runtime, estimator and audit modules. Semantic type checking remains unverified.
- `git diff --check` passed.

Supervisor requests:

| Request | Purpose | Result |
| --- | --- | --- |
| 1 | Read target session/evidence layout | Supervisor `NotFoundError: No such file or directory`; no command log or data returned |
| 2 | Read-only earlier-occurrence target audit | Same supervisor error; no target result |
| 3 | Required build/core/SDK tests on six shards | Submitted at 04:09 UTC; still no result or logs at 04:31 UTC |
| 4 | Build and training-only calibration | Submitted at 04:09 UTC; still no result or fit at 04:31 UTC |

Requests 3 and 4 remain pending; their execution has not been reported successful or failed. Read-only `modal-result` requests likewise returned no status during this work. The source changed after their submission, so the final committed branch requires the checks below even if an older snapshot eventually finishes.

No practice score was requested (0 of 3). There is no measured target-win count or overall delta. An unavailable audit is not a zero-win result, and infrastructure trouble does not establish the card's premise kill.

## Resume validation

Run the final build/tests and obtain the training-frozen artifact through the supplied supervisor:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- 'bash tools/autolab_tests.sh'

python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- 'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && pnpm --filter @tilawa/sdk exec tsx src/recurring-word-fit.ts && mkdir -p /scratch/tracker-recurring-word-acoustic-memory && cp "$LAB_OUT/recurring-word-calibration.json" /scratch/tracker-recurring-word-acoustic-memory/'
```

The single-case audit accepts the returned calibration and runs the real tracker/checker without scoring:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run -- 'AUDIT_SESSION=lab-v1.5-p46 AUDIT_AT=121.84 AUDIT_CALIBRATION=/scratch/tracker-recurring-word-acoustic-memory/recurring-word-calibration.json pnpm --filter @tilawa/core exec tsx /app/.scratch/lab/recurring-word-audit.ts'
```

Repeat with `AUDIT_APPLY=1` for the calibrated product replay. Inspect exact exemplar eligibility, identical comparison frames, raw costs, calibrated relative likelihood and the target's confirmed warning. Install only that frozen model if the premise survives. Follow the user's fewer-than-three-target-wins kill rule; if killed, revert engine changes and retain the measured write-up/outcome. If three or more targets are won, request practice scoring before deciding. No gate decision or fold is taken here.

## Limits

Real training eligibility/schema, number of repeated-word pairs, target history, discrimination, CPU latency and full build/test compatibility remain unvalidated. Subsequence DTW may find an uninformative blank-heavy match on short prefixes. Two exemplars are a bounded-memory approximation. Cross-recording training pairs may include changes in device or articulation; same-recording repeat pairs provide the closer runtime control. The local comparison uses an already observed word entry, not a separate latent boundary model, and does not rescore a continuation without that boundary. Correcting localization can expose the documented listener warning; no checker safeguard is added.

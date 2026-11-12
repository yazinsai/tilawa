# Persistent sequential evidence for local restarts

Status: prototype implemented, opt-in pending training and Modal validation. The supplied supervisor failed before executing the data probe, training fit and required tests. No calibrated default was installed and no target result was measured.

## Prediction

Card prediction: tracker −1, overall 0. Repeated looks at overlapping local evidence might select a temporary rhyme-match advantage rather than sustained movement evidence. Correcting that decision could remove `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84` while exposing the previously documented nearby listener warning.

The prediction remains untested. No practice score was requested (0 of the permitted 3). The card and existing listener were not changed.

## Inspected implementation

`local.ts` already carries minimum edit costs, origin and edit counts through immutable replacement rows, consuming one new phoneme on each update. It enumerates word-start jump edges from the pre-update best state. It does not have an independently accumulated margin statistic at each restart look. `tracker.ts` publishes a new local run when its most recent three phonemes satisfy the existing fit rule; global separation and relocation use their existing rules.

This narrows the premise: the prototype tests whether repeated selection of a minimum-cost run and its short recent fit can select a transient advantage. It does not repair duplicate token consumption or missing prefix charges, and the static inspection does not establish that the target is an optional-stopping failure. The historical 0.3915-edit advantage comes from the supplied earlier experiment findings; it was not remeasured here.

## Mechanism

`sequential-restart.ts` adds a normalized token-observation channel. Each reference state distributes probability over inserting an observed token, advancing with an observed identity, and deleting a reference phone. Reference deletion closure is summed before consuming the next observation. At the retained interval's end, an overflow state emits a normalized background distribution rather than losing probability mass. Existing held-sound and short-vowel indel discounts and surah-entry prices enter the kernel and its normalizer. The original weighted edit recurrence remains unchanged.

A comparison epoch starts from the local lock's endpoint. Its continuation filter and one filter per destination carry sufficient statistics for every observation in that epoch. Destination rows sum alternative token alignments and possible restart times. The restart-time prior is

`q(t) = 1 / (t (t + 1))`, for positive token index `t`.

It sums to one over all future times. Destinations are enumerated before observing the current token, deduplicated, and weighted by the existing movement prices under a normalized destination prior. The two-edit local jump cost supplies total movement-family prior odds separately. Each new token advances all filters once. Reading a comparison again changes no state.

In numerical terms, rows are divided by the continuation's new predictive likelihood after each token. Thus a destination row stores its accumulated likelihood ratio to continuation; its increments are differences between successive prefix log likelihoods, including the proper new restart-time mass. It does not add another copy of an overlapping prefix's fit. A destination can commit only when its log odds exceed the training-frozen calibration.

`local.ts` applies this comparison when the original DP proposes a new origin. If evidence is insufficient and an existing same-origin continuation remains finite, that continuation wins. Every original jump edge remains available. Candidates outside the fixed retained reference interval use the original movement rule. Original global acquisition, relocation, event delivery, recent-fit rules and checker behavior are unchanged. Audit mode reports comparisons while preserving the original local decisions and costs.

The epoch persists through ordinary overlapping local-window recentering. It restarts after accepted movement, or when forward progress exhausts the retained interval and exposes more reference text. At the reference's final boundary, repeated further tokens do not repeatedly reset it. Local rewind restores both the edit search and its sequential evidence snapshot.

## Training and calibration

`packages/sdk/src/restart-fit.ts` is a training-only entry point. It reads `help-train/labels.jsonl`, explicitly accepts only scored `split: train`, `kind: none` recordings with no events, repeats or exclusion, and excludes every speaker with any non-training-labelled recording. It decodes accepted recordings with the deployed `models/spliced-clean-int8` listener and caches those decodes by listener hash.

The fit estimates normalized directed token-edit counts with smoothing. Four smoothing strengths are compared using prefix predictive likelihood on deterministic speaker-separated training folds. Using the selected strength, separate out-of-fold clean local streams calibrate the evidence threshold: their maximum proposed-destination log odds, floored at zero, become the commit threshold. The final channel is fitted to all eligible training clips. The report records listener identity/hash, eligible-label hash, folds, selection and calibration maxima.

The pipeline was authored against the label contract present in existing SDK help code and earlier branch estimators. The failed schema probe returned no actual dataset information. No training recordings, speaker counts, channel parameters, calibration threshold or report were obtained. The prototype is therefore exposed through `TrackerOptions.localRestartEvidence`; no untrained channel or guessed calibration is enabled by default.

## Validation

Dependency-free Node smoke assertions passed against the actual TypeScript source using Node's type stripper and a small source-import resolver. They cover:

- Next-token probabilities sum to one, including latent deletions and interval overflow.
- An uninformative one-phone channel assigns exactly `t / (t + 1)` prior mass to all restart times observed so far; repeated looks do not create full new trials.
- Repeated comparison reads consume no observations; duplicate destinations contribute once.
- A genuine synthetic jump accumulates destination-specific evidence.
- A temporary one-token advantage remains uncommitted and is contradicted by subsequent in-order tokens.
- Rewind reproduces evidence and local positions exactly.
- Channel fitting produces normalized probabilities and distinguishes a matching reference from an incompatible one.
- Audit mode matches the original local search, while a rejecting calibration keeps an available continuation.
- Training eligibility excludes held-out speakers and deterministic folds keep each speaker together.

These are smoke assertions, not a substitute for the required TypeScript build and Vitest core/SDK suites. Dedicated Vitest tests were added for these behaviors. Node's TypeScript syntax stripping and `git diff --check` also passed; syntax stripping does not type-check.

Supervisor requests:

1. Read-only storage and help-train label-schema probe: failed before execution with `NotFoundError: No such file or directory`.
2. Core/SDK builds followed by training-only calibration: failed before execution with the same error.
3. Six-shard `bash tools/autolab_tests.sh`: failed before execution with the same error.
4. Final-source six-shard required test retry: failed before execution with the same `NotFoundError: No such file or directory`.

No Modal log, build result, test result, training output or target trace was returned by requests 1–4. No other route to Modal or data was attempted.

## Limits and recovery

The comparison is exact within its fixed local reference interval and the one-restart-per-epoch model, not over every full-session motion trajectory. Its prior sources are the original local search's accepted pre-observation positions, rather than a sum over all possible motion graphs. The retained interval's background overflow is a modeling approximation; out-of-interval restarts retain original behavior. Accepted local origin changes define epochs even when the existing event fit rule delays publication. Long-session false-movement rates and runtime overhead remain unmeasured.

Training counts use directed minimum-edit alignments, while runtime marginalizes alignments. Clean single-ayah calibration streams are seeded at their transcript's reference onset, whereas runtime starts at an acquired lock. Clean-only calibration provides no measured true-jump sensitivity; its maximum-tail threshold does not establish a full-session false-warning guarantee. Correcting localization can still expose the documented listener accusation because checker rules are unchanged.

Once the supervisor works, run:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 30 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && RESTART_CACHE=/scratch/tracker-sequential-restart-evidence/clean-decodes pnpm --filter @tilawa/sdk exec tsx src/restart-fit.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- \
  'bash tools/autolab_tests.sh'
```

Bring the returned `restart-calibration.json` into this worktree without changing it in response to practice. The diagnostic replay helper accepts the case and calibration as arguments, and runs the existing tracker/checker chunking without scoring:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run -- \
  'pnpm --filter @tilawa/core exec tsx /app/.scratch/lab/sequential-target-audit.ts --set lab-v1.5u --session lab-v1.5-p46 --calibration /app/.scratch/lab/restart-calibration.json --mode sequential'
```

Repeat with `--mode original` and `--mode audit`, then inspect the restart decision and final checker warning. If training-frozen sequential evidence still commits the offending restart before distinguishing continuation arrives, the premise fails. If the built change wins fewer than three targets, follow the user's kill rule: revert engine changes, retain the measured negative write-up and commit a killed outcome. The single listed target cannot alone supply the required three wins; synthetic examples are not additional target wins.

If the authorized target validation clears that criterion, install the returned calibration as an engine default, run the required checks, and request one practice score before gate-dependent narrowing. The code is not ready for a fold without that work. Infrastructure failure supplies no measured zero-win result and no evidence for a premise kill.

Local smoke command:

```sh
node --experimental-strip-types --import ./.scratch/lab/sequential-node-register.mjs .scratch/lab/sequential-smoke.mjs
```

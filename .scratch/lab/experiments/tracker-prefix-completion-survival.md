# Completion survival from causal tracker prefixes

Status: prototype implemented; calibration and required Modal validation blocked by an unresponsive supervisor. The trained model is not installed as a default. This branch is not a measured warning improvement or a premise kill.

## Prediction

Tracker −1, overall +0 on practice phone audio. The target is `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`. Recovering its near continuation may expose the documented nearby listener false warning. No standalone fold gain is promised.

The supplied history reports a source visit matching only 2 of 11 phones. The previous open-frontier audit establishes that unfinished terminal suffixes are already uncharged. Those are supplied findings, not measurements from this worktree. The experiment tests whether the trajectory of that partial match predicts recovery better than its endpoint alone.

## Mechanism

`local.ts`, reached through `tracker.ts`, retains a compact within-word summary for each finite candidate: reference progress and length, entry token, observed-token count, exact matches, accumulated acoustic edit cost, latest consuming cost and consecutive nonmatching observations. Replacement arrays carry those summaries through recentering and save/restore. An optional read-only observer exports causal snapshots for training. Without a fitted completion model, position decisions retain the original costs.

The completion model is a logistic cure mixture with geometric waiting times for completion and abandonment. A labelled event contributes its event density. A right-censored prefix contributes the sum of the two weighted survival functions, rather than an abandonment label. EM fits the latent eventual outcome for censored prefixes. The complete model and endpoint control each have eight features, identical data, ridge strength, training compute and likelihood. The control uses polynomial features of current progress, length and cumulative cost; the complete model replaces those extra features with trajectory statistics.

For a departure from an incomplete word, pending probability `p` marginalizes two explanations: a word whose match has already ended through recognition loss, and a pending word whose remaining suffix must be discharged. The additional cost is `−log((1−p) + p exp(−3 C_suffix / UNIT)) UNIT / 3`. `C_suffix` uses existing phone deletion costs; the conversion is the tracker's existing 3 nats/edit. Open frontiers stay free, completed words add zero, and both movement directions use the same interpretation. The movement edges, movement prices, commitment margins, event delivery and checker rules are unchanged. This introduces no pause test or inference delay.

## Training and isolation

The SDK estimator reads only help-train scored, none-labelled training takes with no events, repeats or exclusions. It excludes any speaker with a non-training annotation, including mixed-split speakers, and checks that resolved audio paths remain within help-train. It does not open practice sessions, exam recordings or warning scores.

It decodes each take through the shipped listener in 40 ms audio pushes, then runs the unchanged tracker with a prefix observer. Eligible near-best candidate prefixes are supervised from later transcript-consistent progress in that actual run: an in-order next word with the same run origin completes the hypothesis; bypass or a new local-search generation abandons it; unresolved recording endings are censored. Endpoint arrival alone cannot establish completion. Each recording has equal total weight. This supervision remains a transcript-and-tracker proxy, not independently verified acoustic word boundaries.

Three speaker-separated training folds select a shared ridge strength from 0.001, 0.01 and 0.1 by held-partition likelihood summed over both models. Both are then refitted on eligible training speakers. The estimator writes `completion-model.json` and `completion-fit-report.json` to `LAB_OUT`, recording data/model/decoder hashes, sample and speaker counts, censoring counts and all selection results. No training result has returned, so no coefficients or calibration were selected against practice or installed in the product.

## Validation obtained

Ten focused test bodies passed locally with Node 22 TypeScript stripping and a temporary assertion adapter for Vitest. This checks endpoint-control isolation, censor likelihood, fitted recovery discrimination, finite normalized probabilities, departure marginalization, completed-word behavior, unchanged uncalibrated local decisions, snapshot retention and rewind, a synthetic departure decision, training-speaker exclusion, and generation-aware completion/abandonment supervision. The temporary adapter and copied files were removed. These results are not a TypeScript build or the required Vitest suites.

Supervisor requests made through the supplied runner on 2026-11-13:

| Run | Request | Result at commit |
| --- | --- | --- |
| 1 | Read-only training-data availability inspection | No result or log returned after approximately 14 minutes |
| 2 | Six-shard `bash tools/autolab_tests.sh` | No result or log returned |
| 3 | Build, latest focused Vitest tests, isolated training fit; submitted with `--no-wait` | Queued; no result or log returned |

The supervisor enforces at most two concurrent runs; run 3 can queue behind the earlier requests. No alternative Modal access was attempted. A lack of acknowledgement does not prove a storage or credential failure, and does not falsify the completion premise.

No fitted probabilities, target ordering, target warnings, clean-session results or studio results were measured. Zero of the three allowed practice scores were requested. Target wins and overall delta are unknown, not zero. The card, scorer, lab tooling, data, manifests and evidence remain unchanged.

## Remaining work

Collect the existing requests through `python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-result 1`, `modal-result 2`, and `modal-result 3` after the supervisor resumes. If request 3 has failed before execution, rerun its command:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && pnpm --filter @tilawa/core exec vitest run test/completion-survival.test.ts && pnpm --filter @tilawa/sdk exec vitest run test/completion-training.test.ts && COMPLETION_CACHE=/scratch/tracker-prefix-completion-survival/decodes pnpm --filter @tilawa/sdk exec tsx src/completion-survival-fit.ts'
```

Freeze the returned full model and endpoint control before inspecting practice. Compare the target's continuation and restart under both models over identical observations, then replay through the unchanged checker. If the training-frozen model fails to reverse the ordering, or the built change wins fewer than three targets, revert engine changes/tests, retain the measured write-up, and commit a killed outcome under the user's rule. If at least three targets are won, request a practice score before deciding anything about the gate. A production default and a measured verdict remain outstanding; this prototype intentionally does not invent trained parameters to fill that gap.

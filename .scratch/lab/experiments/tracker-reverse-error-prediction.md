# Reverse predictive observation scoring

Status: needs-human. Prototype implemented behind an explicit tracker option. The authorized Modal supervisor failed before executing every requested command. Training, the required build/Vitest suites and the target replay remain blocked; no fitted default model has been installed.

## Prediction

Tracker −1, overall 0 on lab-v1.5u, using the card's listener `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. The named target is `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`. Correct localization may expose the documented nearby listener wrong-word warning, so the forecast includes no overall gain.

The near continuation's 0.3915-edit disadvantage is evidence supplied in the brief, not a measurement reproduced here. The card is unchanged.

## Mechanism

`packages/core/src/predictive-channel.ts` implements a normalized reference-to-observation transducer. At each reference frontier a geometric insertion process precedes either deletion of a reference phone or emission of an observation. Completion includes a termination probability. A log-domain forward/backward calculation sums all edit placements, including uncertain placement around repeated duration symbols; training uses their posterior expected counts rather than greedy alignment labels.

Emission probabilities condition on reference-phone identity, the phonetic classes of one or two neighbouring reference phones, and the classes of one or two preceding observations in the transducer's processing direction. Sparse context rows receive Dirichlet pooling toward the fitted context-independent reference-phone row. In the reverse model the complete written and observed strings are processed from right to left, so those neighbours are a reference suffix and subsequent observations. This reverses the prediction direction, not the tracker's greedy edit script. No additional audio is read at inference.

`local.ts` retains the reference traversal alongside each original weighted DP representative. Reference-consuming match/substitution/deletion transitions append their phone; an insertion appends none. A movement appends the destination phone without adding the bypassed text as acoustic deletions. The heard string includes every observation since this local search began, and survives save/restore together with its reference histories.

When the original winner changes run origin, each existing motion origin's strongest weighted trajectory is compared over that identical complete observation interval. For each candidate, both transducers independently marginalize observation edit placements against its retained reference traversal. The ranking score is the original weighted cost minus the reverse-versus-forward log-likelihood difference converted into edit units. The matched-forward control subtracts zero directional evidence while still computing both likelihoods for the audit. Movement and duration charges stay in the original cost; global commitment margins, selected-state metadata, event handling and checker rules use their existing code.

`TrackerOptions.localPredictive` accepts the frozen pair and `reverse`, `forward` or `audit` mode. Model probabilities, matched capacity and calibration are validated and copied when a local search starts. Without the option, normal engine behavior is unchanged. There is deliberately no untrained placeholder default.

## Training and control

`packages/sdk/src/predictive-channel-fit.ts` opens only `help-train/labels.jsonl`, its eligible training audio, the reference pronunciation asset and the authorized deployed model. Eligibility requires scored train/none labels, no exclusion, no planted events and no repeats. Any speaker with a non-training annotation is excluded entirely, including mixed-split speakers. The provided isolated help-train pool contains no release-held-out speakers.

The unchanged SDK streaming listener decodes recordings with its normal audio geometry. Decode caches are keyed by model, decoder-source and audio hashes. Whole contiguous labelled ayah pronunciations provide the reference sequence; latent edit placement is estimated rather than supplied by tracker visits.

Both directions use identical eligible recordings, emission schema, initialization and twelve EM iterations. Shared context length (one or two) and pooling strength (8 or 32) are selected jointly by summed held-out log likelihood over up to three deterministic, speaker-separated training partitions. The same selected configuration is refitted for each direction on the entire eligible training pool. Thus neither direction receives separate capacity or practice-driven parameter selection.

Edit-unit calibration is frozen as the mean across training folds of pooled emission mismatch log odds per existing weighted substitution cost. This calculation uses only fitted training models. Returned artifacts would include the paired model and a report of selection, calibration and source hashes. None was returned in this attempt.

## Validation

Nine focused test bodies passed locally with Node 22 TypeScript stripping and a small assertion adapter for Vitest imports. This is not a TypeScript build or a Vitest suite pass. They check:

- Exact marginal probability of an ambiguous one-reference/one-observation example.
- Normalized complete-string mass with geometric insertions and proper termination.
- An earlier mismatch receiving evidence from an already-heard subsequent observation.
- Context-independent reversal agreement and positive, normalized EM emission rows.
- Unknown-phone rejection and matched-capacity/calibration ownership.
- Matched-forward movement equivalence on a synthetic restart, retained common observation history and repeatable save/restore.
- Exclusion of held-out/mixed-split speakers, planted edits, repeats and unscored takes.

`git diff --check` passes. The synthetic observations are generated in test code. No card, scorer, lab tooling, manifests, evidence or test data was changed.

All authorized supervisor requests failed before command execution with `NotFoundError: No such file or directory`, no command log, no usable artifacts and zero billed compute:

| Run | Requested work |
| --- | --- |
| 1 | Read-only help-train schema and run-environment inspection |
| 2 | Core/SDK build, training decodes, speaker-fold selection and frozen fit |
| 3 | Required core/SDK build and Vitest tests on six shards |

Practice scores requested: **0 of 3**. Target preference reversal, actual warning removal, target wins, component/overall changes, clean-session warnings and studio effects are unmeasured. An infrastructure failure does not establish a failed premise or zero target wins; no killed outcome or fold/discard decision is recorded.

## Limits and recovery

This is a categorical predictive model with phonetic-class context sharing, not an unrestricted recurrent neural model. It sums acoustic edits within each retained reference traversal; it does not recover movement histories already discarded by the original weighted recurrence. Only the strongest weighted endpoint for each motion origin is rescored. Evidence starts after the global lock that creates a local search. Inference retains that full prefix, so long-session CPU and memory cost need measurement on Modal. Training-pool size, adequacy and real-data calibration remain unknown.

After restoring the supplied supervisor's run environment, run training without practice input and the final required tests:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && PREDICTIVE_CACHE=/scratch/tracker-reverse-error-prediction/decodes pnpm --filter @tilawa/sdk exec tsx src/predictive-channel-fit.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- \
  'bash tools/autolab_tests.sh'
```

Copy the returned frozen paired model into this worktree before opening target evidence. The read-only audit accepts session, time and model path through environment variables and replays both controls through the unchanged checker:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- \
  'AUDIT_SESSION=lab-v1.5-p46 AUDIT_AT=121.84 AUDIT_MODEL=/app/.scratch/lab/predictive-model.json pnpm --filter @tilawa/core exec tsx /app/.scratch/lab/reverse-prediction-audit.ts'
```

Inspect the same-interval likelihood differences and actual confirmed target warning, not just a changed cursor. The supplied list names one target; even a successful single warning removal is below the user's three-target kill threshold. If the measured premise fails, or the built change wins fewer than three targets, revert engine/tests/audit, retain the measured write-up and write the requested killed outcome. If three or more targets are won, install the frozen default and obtain the required practice score before any gate-dependent narrowing. This attempt has not reached that decision.

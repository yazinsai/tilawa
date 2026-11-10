# Tracker inter-emission gap channel

Status: needs-human — supervisor Modal execution unavailable; calibration and premise validation are unfinished.

## Preregistered prediction

Tracker −1 wrong call, overall 0 on lab-v1.5u practice. The target is
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
A corrected binding could expose the previously documented nearby listener warning.
`lab-experiment.json` is unchanged.

## Mechanism and prototype

The current local matcher consumes only phone identities. `tracker.ts` retains the token-run frame bounds in
`observedFrames`, but calls `local.step(ph.p)` without those intervals. The proposed channel models the number
of frames between consecutive inclusive token runs (`next.start - previous.end - 1`). It does not treat a pause
as proof of movement, change spoken-phone duration prices, or suppress a downstream warning.

`packages/core/src/interemission-gap.ts` contains the calibration and fixed-reference comparison prototype.
It predicts logarithmically binned gaps with a normalized hierarchy: boundary and erased-phone composition,
flank phone classes, exact flanking reference phones, and exact intervening reference phones. Intervening
segments longer than three phones share a long-segment backoff category. Predictions are expressed as
log-likelihood ratios against the pooled distribution. Missing timing and wholly unseen contexts contribute
zero preference; sparse contexts shrink toward their parent distributions.

The comparison marginalizes insertion, substitution and deletion placement over the same heard interval in
both numerator and denominator. The denominator uses the same edit paths without gap evidence, preventing
extra alignment routes from receiving free probability credit. Existing weighted edit costs remain intact,
including duration-expanded reference symbols and terminal reference-only deletions. Each observed gap gets
one unit of expected context count under the training edit posterior.

`packages/sdk/src/interemission-gap-fit.ts` is a training-only calibration entry point. It selects only
`help-train` labels with `split: train`, `kind: none`, `scored: true`, and no events or repeats. It decodes with
the supplied primary streaming model, then chooses hierarchical pseudocount mass from 8, 32 and 128 by
speaker-grouped predictive log likelihood, using at most five folds. It writes fitted counts and provenance
(model and selected-label hashes, frame grid, clip/speaker counts, fold results, and rounding error) to
`LAB_OUT`. Neither practice recordings nor held-out speakers participate in fitting or model selection.

This is a prototype, not an enabled tracker modification. No fitted model has been produced, and timing has
not been wired into production local comparisons. Movement transitions, commitment margins, event delivery
and checker behavior remain unchanged. This prevents an untrained placeholder from masquerading as the
specified experiment.

## Validation attempted

The supplied supervisor runner failed before command execution:

- Run 1: inspect available help-train data — `NotFoundError: No such file or directory`.
- Run 2: retry training-data inspection — the same error.
- Run 3: `bash tools/autolab_tests.sh`, six shards — the same error, with no build or test logs.
- Run 4: build core/SDK and run `interemission-gap-fit.ts` — the same error.

The supervisor result JSON for each reported zero spend and contained only that error. No evidence, labels,
audio, fitted model, test result or incident result was returned.

Five focused unit tests were added for inclusive frame gaps, normalized hierarchical backoff, zero contribution
from uninformative contexts, distinguishing a deleted nucleus from adjacent phones, and exactly one unit of
posterior count per physical gap. They have not run. Static whitespace checking passes. No practice-score
requests were made (0 of 3 used).

## Limits and remaining work

The target's saved-audit claim has not been independently reproduced in this checkout; the older timing and
syllable experiment branches named in the brief are absent. The current code inspection establishes that
timing is discarded, but cannot establish that timing favors the continuation. There is no measured target-win
count or overall delta. Infrastructure failure is not a premise kill, and no zero-win measurement is inferred
from it.

After restoring the supervisor runner, run calibration and the core/SDK tests, freeze the selected model, then
compare the target's continuation and restart using identical heard-token bounds and unchanged movement and
duration charges. Only a surviving premise warrants production integration. Apply the user's kill rule if the
built change wins fewer than three targets; if killed, remove the prototype engine/test files and retain the
measured write-up. Otherwise the supervisor and fold gate decide the score.

Calibration command:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 30 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && GAP_CACHE=/scratch/tracker-interemission-gap-channel/clean-decodes pnpm --filter @tilawa/sdk exec tsx src/interemission-gap-fit.ts'
```

Required checks:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- 'bash tools/autolab_tests.sh'
```

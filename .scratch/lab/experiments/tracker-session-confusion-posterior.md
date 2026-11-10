# Session-specific phonetic confusion posterior

Status: needs-human — the supplied supervisor runner fails before executing commands. Calibration, target validation and production integration remain unfinished.

## Prediction

Tracker −1 wrong call, overall 0 on lab-v1.5u practice. Target:
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
Correcting its location may expose the documented nearby listener false warning.
`lab-experiment.json` is unchanged.

The card's premise is that the available near continuation loses to the rhyme restart by
0.3915 edits over the same heard interval. That measurement comes from the supplied brief;
it has not been reproduced in this checkout.

## Implemented prototype

`packages/core/src/session-confusion.ts` fits a small mixture of recording-level phonetic
mappings. Each mapping predicts normalized heard-phone distributions for every reference phone.
Sparse counts pool toward a population channel and share changes through five phonetic feature
axes: sound class, manner, place, voicing, and vowel/nasal class. The mapping is shared across all
words in one recording, rather than being redrawn per phone or conditioned on reference neighbours.

Training uses fractional written/heard pair counts from forward/backward over the existing
directed edit costs. Insertions and deletions retain their original prices and do not become
identity-confusion observations. Duration-expanded symbols are retained. The mapping mixture
fits these fixed sufficient statistics; it does not jointly refit the edit alignment at every EM
iteration. This approximation is explicit and needs training-partition validation.

`committedConfusionHistory` records non-overlapping committed stretches, waits for checker
acknowledgment, excludes any stretch containing an accused word, and excludes observations
inside or after the comparison interval. An uncertain or resolved accusation still excludes
its word. `freezeBefore` owns a posterior snapshot; subsequent feedback or adaptation cannot
alter a comparison already in progress.

`freezeConfusion` supplies per-mapping substitution costs and marginalizes complete candidate
costs once over that shared frozen posterior. `confusionSpanCost` is a fixed-start, open-terminal
comparison prototype that consumes every heard token, charges leading reference deletions,
and retains duration and ordinary indel charges. It does not implement movement arcs; their
existing costs must be included when integrating it with the local search.

`packages/sdk/src/session-confusion-fit.ts` selects only help-train records with `split: train`,
`kind: none`, `scored: true`, and no labelled events or repeats. It decodes with the supplied
streaming model and compares one, two and three mappings with pooling masses 16, 64 and 256
using up to five training-speaker folds. Selection uses held-partition expected predictive
log likelihood, never practice warnings. It writes the fitted model and provenance, including
listener/model hash, selected-label hash and fold results, to `LAB_OUT`.

No fitted model has been returned. These helpers are not enabled in `tracker.ts`; the live
tracker/checker feedback plumbing and local movement DP integration remain to be implemented
after calibration. A placeholder model would not test the stated mechanism. Current movement
transitions, commit margins, events and downstream warnings therefore retain their behavior.

## Validation

Seven focused tests cover normalized emission rows, transfer to an unobserved confusion pair,
one shared mapping across phones, immutable posterior/model snapshots, prior-only and unknown
phone behavior, fractional edit counts, eligible-history isolation, and existing duration and
endpoint semantics.

The test bodies passed locally under Node 22's TypeScript stripping with a small assertion
adapter for the Vitest imports. This checks the small numerical prototype; it is not a Vitest
suite result or a TypeScript build. A first check surfaced signed zero in exact equality assertions
for floating likelihood costs; those assertions now check numerical equality to twelve decimals.

The supplied supervisor requests:

- Run 1, inspect help-train label schema: failed before execution with `NotFoundError: No such file or directory`.
- Run 2, required core/SDK build and tests on six shards: the same failure, with no test logs.
- Run 3, training-only calibration after core/SDK build: the same failure, with no fitted output.
- Run 4, retry the required six-shard checks after adding interval-isolation tests: the same pre-execution failure, with no build or test logs.

No practice-score requests were made (0 of 3). There is no measured target-win count or
overall delta. Infrastructure failure supplies no evidence for a premise kill, so the prototype
is retained and the outcome is `needs-human`, rather than recording a fictitious zero-win result.

## Remaining work

Restore the supervisor runner and obtain a real build/test result and a training-frozen model:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 30 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && CONFUSION_CACHE=/scratch/tracker-session-confusion-posterior/clean-decodes pnpm --filter @tilawa/sdk exec tsx src/session-confusion-fit.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- \
  'bash tools/autolab_tests.sh'
```

Then connect actual committed stretches and checker feedback, and use the same frozen posterior
for continuation and restart over an identical heard-token interval. Earlier adaptation speech
must remain outside that interval; feeding it into both the posterior and the candidate likelihood
would double-count it. Preserve existing movement and duration charges, commitment margins and
warning handling. Validate the target comparison and product replay before deciding whether the
premise survives. Apply the user's fewer-than-three-target-wins kill rule to a measured result;
if killed, remove the prototype code/tests, retain this write-up and record that result.

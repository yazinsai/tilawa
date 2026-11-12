# Tracker cost and edit-witness coupling

Status: killed on the premise. No engine change retained. Experiment card left
unchanged.

## Prediction

Tracker −1, overall −1 on practice, from the false `skipped_passage` warning
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
This is the card's speculative prediction, below the fold gate on its own.

## Proposed mechanism

Keep each dynamic-programming cost coupled to a persistent edit witness whose
predecessor identifies the original cell generation. Reconstruct the existing
winner through those witnesses, preserving costs, positions, margins, window
timing and lock-drop handling. The proposed defect was coordinate reuse: a
numerically correct winner would export a traceback whose predecessor metadata
had since been overwritten.

## Source audit

The experiment begins at `f97f409c`.

- Locked local matching is implemented by `createLocalSearch` in
  `packages/core/src/local.ts`, called from `tracker.ts`. It carries parallel
  `cost`, `origin` and `edits` arrays, with no traceback coordinates or predecessor
  table. Every selected transition assigns its cost, origin and edit count
  together. Consuming transitions read the previous arrays; reference-only
  deletion reads an already-finalized cell in the new row. Each step replaces
  all three arrays. Recentering copies all three from the same old cell, and
  save/restore retains the same array generation.
- `tracker.ts::localEvent` exports the selected endpoint, run origin, recent
  cost and PER. Its `quietTrace` has `path: null`. There is no local edit
  traceback to reconstruct through reusable coordinates.
- Global matching does export `SpanAlignment.path`, through
  `fuzzy.ts::alignSubstring`. It fills a complete table with
  `(reference length + 1) × (observation length + 1)` cells before traceback and
  retains that table until traceback finishes. Its coordinates are not rolling
  cell generations.
- `checker.ts` does not read tracker traceback paths. It constructs acoustic
  visits in its own alignment and uses those visits and position events to
  maintain the ayah ledger.

## Validation

Prepared temporary audit copies that retained the original numeric recurrence
and added frozen predecessor nodes for every finite cell. Each node linked
directly to its chosen predecessor object, including same-generation
reference-only deletions. An independent pass was designed to recompute
insertion, substitution, deletion, entry and jump charges from the referenced
occurrences, checking cumulative cost after row normalization, run origin and
edit count against the original arrays. It was also designed to check old
arrays for mutation and compare complete tracker/checker events with an
uninstrumented replay.

Modal run 1 requested the unchanged build and core/SDK tests on six shards:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- 'bash tools/autolab_tests.sh'
```

Run 2 requested the target audit. Both failed in the supervisor's Modal
infrastructure with `NotFoundError: No such file or directory`, returning no
test or replay logs and reporting zero compute cost. Run 3 retried the audit
with a corrected launch command:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 10 -- 'pnpm --filter @tilawa/core exec tsx ../../../.scratch/lab/cost-witness-audit/run.ts lab-v1.5-p46 spliced-clean-int8 phone'
```

Run 3 also failed with `NotFoundError: No such file or directory`, returning
no replay output and reporting zero compute cost. No runtime cost-agreement,
event-equivalence or test-pass claim is made. No practice-score calls were
made and no overall delta was measured.

## Decision

The mechanism requires a local traceback whose predecessor coordinates can
rebind to newer metadata. The inspected implementation has no such traceback:
its scalar path summaries are propagated alongside the winning cost, and the
checker never consumes tracker edit paths. Adding a persistent witness cannot
repair the proposed coordinate reuse because that representation is absent.
Kill on that premise, without changing scoring, localization or warnings.

All temporary diagnostic modules and their driver are removed. The retained
result is this write-up and the killed outcome. `targets won: 0` records that
this audit produced no engine intervention or target recovery; it is not a
measured practice replay result.

## Limits

The source audit addresses the proposed representation defect; it does not
establish why the ayah-absence likelihood prefers a skip or settle any listening
judgment. The requested target replay covered one practice session on stored
listener evidence and did not execute successfully. No constants, scorer, lab
tooling, labels, evidence or experiment card were edited.

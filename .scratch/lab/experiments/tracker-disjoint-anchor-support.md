# Disjoint acoustic support for skip anchors

Status: needs-human — the supervisor's Modal runner failed before the diagnostic command ran. The target premise remains unmeasured; this experiment is not killed.

## Prediction

Card `tracker-disjoint-anchor-support`, tracker, explore. On practice phone audio, predicted tracker −1 and overall −1, with no counted losses. The target is `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`. The card is unchanged.

## Mechanism

Retain the observed-token support interval of each accepted anchor. For a proposed skip, freeze the established source frontier and recompute destination support from observations beyond it, using the existing matching costs and commitment requirement. Export a skip-eligible witness only when that independent support suffices. The checker would withhold passage-skip candidates lacking the witness, preserving selected positions, windows, word-edit issues and lock-drop handling.

The premise must be checked first: kill if the target witnesses are already disjoint, or the destination still commits after shared observations are excluded. If a built prototype wins fewer than three targets, the builder also kills it under the supplied instructions. No result is claimed from the card's prediction alone.

## Static findings

`packages/core/src/local.ts` propagates cost, run origin and edit count through replacement arrays. A jump takes its source cost from the previous row's selected state, consumes the current observation at the destination, and resets the run origin. This separates support at the jump edge itself. It does not establish whether a subsequently published destination run overlaps a source frontier published while that alternative remained unselected.

`tracker.ts` publishes a changed local run only after the existing three-observation recent-cost check. The ledger in `checker.ts` separately retains per-ayah tracker lock spans and counting acoustic visits; it does not consume an edit traceback or an explicit independent destination witness. Consequently, source/destination token provenance and the ledger's actual neighbour spans need to be inspected together.

## Prepared validation

The diagnostic is isolated under `tracker-disjoint-anchor-support/` beside this write-up. `prepare-audit.py` creates temporary copies of the local matcher, tracker and checker; the shipped files are never edited. Added arrays follow the exact winner of each existing transition and retain the first and exclusive last observed-token identities for its run. Insertions retain the most recent reference-consuming endpoint; substitutions count as reference-consuming observations. Support is reset at jumps and retained across recentering and save/restore.

`replay.ts` reads the listener and test-set name from the unmodified experiment card. It accepts a session and time, logs accepted local support intervals and ledger neighbours around that time, and runs the unchanged tracker and checker alongside the diagnostic copies. Any difference in any published event fails the replay. It writes the comparison result and nearby token identities into `LAB_OUT`.

Run from the worktree through the authorized supervisor:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 20 -- 'python3 .scratch/lab/experiments/tracker-disjoint-anchor-support/prepare-audit.py && pnpm --filter @tilawa/core exec tsx ../../.scratch/lab/experiments/tracker-disjoint-anchor-support/replay.ts lab-v1.5-p46 121.84'
```

The first two audit requests returned exactly `NotFoundError: No such file or directory`, with zero reported compute spend and no shard command logs or result artifacts. A minimal `pwd` request returned the same error, establishing that the runner fails before executing even a trivial command. No practice score has been requested: 0 of the allowed 3 used.

## Limits and next step

No engine change or unit-test change has been made, and core/SDK tests have not run. The audit generator passed Python syntax compilation and generated all three diagnostic modules locally; the generated source was inspected. Its event-equivalence assertions have not executed. The failed infrastructure requests establish neither target support overlap nor a target win.

Restore the supervisor's Modal runner and execute the diagnostic above. If the witnesses are disjoint, record that premise kill. Otherwise recompute the destination fit after excluding the shared observations, and proceed with the engine prototype only if the existing commitment requirement fails. There is no permission request or fold decision pending.

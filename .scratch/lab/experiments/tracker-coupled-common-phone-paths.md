# Coupled shared-phone assignments at a local restart

Status: blocked on Modal infrastructure; premise and target outcome are unmeasured. No engine change is retained and the experiment has not been killed.

## Prediction

Tracker -1 wrong call, overall +0 on lab-v1.5u practice. Correcting the binding may expose the documented nearby listener false warning. Target: `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`. The card remains unchanged.

## Proposed mechanism

Represent a competing continuation and restart as a paired edit graph. Corresponding shared pronunciation subsequences must consume the same observed-token intervals. Differing subsequences keep separate edit paths. Marginalize ambiguous reference correspondences with normalized weights, symmetrically, while keeping edit prices, movement penalties, comparison horizon and commitment margin. Comparisons already using identical shared-phone assignments should stay unchanged.

The first decision is a premise audit: reconstruct both minimum-cost paths at the target restart and compare their shared-phone assignments. If those assignments are already identical, kill on the premise. Otherwise build the paired graph and check whether coupling changes the preference. No target recovery or practice delta is claimed without a successful replay.

## Source inspection and prepared audit

The local recurrence lives in `packages/core/src/local.ts`, called by `tracker.ts`. It uses an immutable previous cost row for consuming transitions, a current-row reference deletion closure, and jumps from the previous best state to word starts. It compares raw weighted costs and recentres across a two-ayah window on either side. The inspection alone does not establish the target's token assignments.

The target's near pronunciation is `بِننننُذُر`; the rhyme restart is `وَنُذُر`. Their shared `نُذُر` has four possible reference-noon correspondences because of the expanded noon run. This is a reference fact, not evidence that their observed-token assignments differ. Those alternatives must be accounted for before interpreting a binding difference.

A temporary diagnostic copy of the unchanged tracker/local recurrence was prepared. It records predecessor, operation, reference offset, observed-token index and charged price at each finite cell, carries predecessor identities through recentering and save/restore, and exports the competing paths around 121.84 seconds. The probe also compares every emitted event with a parallel run of the original tracker and fails if instrumentation changes an event. The diagnostic copies were not added to engine source. They could not execute successfully, so neither event equivalence nor the target tracebacks were validated.

## Validation infrastructure

All execution requests used the authorized supervisor command, never direct Modal access:

- Modal run 1: target traceback audit. Returned `NotFoundError: No such file or directory`, with no shard log or audit output.
- Modal run 2: `bash tools/autolab_tests.sh`, six shards, covering the core/SDK builds and tests. Returned the same supervisor error, with no test output.
- Modal run 3: minimal `pwd` launch check, to distinguish a runner launch failure from a probe failure. Result pending when this record was prepared.

The returned errors do not identify a missing path. They do not establish a code error, a premise failure, or a target win/loss. No practice scoring request was made: all three allowed practice scores remain unused.

## Limits and resumption

Implementation remains unfinished because the required target premise audit cannot run. Restore the supervisor's Modal runner, reconstruct the two paths over the same heard interval, examine all normalized shared-phone correspondences, and then apply the card's premise test. If coupling is warranted, implement it in engine code with synthetic unit tests, run the core and SDK checks, and replay the target before any score or kill decision.

No scorer, lab tooling, manifests, test data or evidence was edited. No `killed` outcome is written: an infrastructure failure is not a measured falsification of this hypothesis.

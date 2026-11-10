# Event-preserving exact recombination

Status: needs-human — Modal execution failed before the premise audit or tests could run. The experiment is unvalidated, not killed.

## Prediction

Tracker −1, overall −1 on practice phone audio, from removing the false `skipped_passage` warning at `lab-v1.5-p46/54:36:1/121.84`. No other wins were predicted. These are the card's predictions, not measured changes; `lab-experiment.json` is unchanged.

## Proposed mechanism

Keep a compact predecessor graph for exactly equal-cost histories recombined into the same local DP cell. Preserve the selected cursor, run origin, edit count, scores, margins, windows and lock-drop handling. Record reference consumption and deletion/jump transitions separately, then withhold a passage-skip candidate only when its supporting equivalent histories disagree about that interval. This uses exact numerical equality, with no near-best paths or uncertainty margin.

Source inspection confirms that `packages/core/src/local.ts` retains one `origin` and `edits` annotation per endpoint. Its consuming transition wins a tie with insertion; jump and deletion transitions replace the winner only on strictly lower cost. Consequently, tied predecessor annotations can be discarded. `tracker.ts` exports the selected local run, while `checker.ts`'s `closeGap` can generate an ayah-absence candidate from committed neighbours and the read-versus-skip fit. These facts establish where to audit; they do not establish that exact recombination causes the target warning.

## Prepared audit and validation limits

A temporary instrumented copy of the local recurrence retained every finite predecessor whose cost was exactly equal to the original cell minimum. It left the original numerical recurrence and origin/edit tie-breaking intact. Immutable predecessor nodes recorded consuming transitions, reference-only deletions and forward-jump deletions intersecting the queried ayah. Unrelated transitions shared their existing parent node; equal predecessors shared subsequent computation through a union node. Recentering and save/restore carried the graph alongside the original arrays.

The graph projection distinguished histories with no consuming transition in the queried interval from histories with at least one. It also retained possible/definite consumed-phone and deleted-phone masks and whether a forward jump crossed the entire interval. The planned target replay compared every tracker and checker event with baseline through 180 seconds, covering the restart and subsequent confirmation interval. Synthetic checks were prepared for consumption-versus-deletion forks, extension after a fork, deduplication, and recurrence/save/restore equivalence on clean, omitted, repeated and corrupted synthetic strings.

All three supervisor requests failed with the identical result `NotFoundError: No such file or directory`, without a shard log or audit output:

1. Modal run 1: target replay of stored `spliced-clean-int8` phone evidence through the unchanged checker.
2. Modal run 2: `bash tools/autolab_tests.sh`, six shards, including the required core/SDK build and tests.
3. Modal run 3: synthetic audit checks followed by the target replay.

Each returned `usd: 0.0`; no command execution was verified. The failure did not identify the missing path. There is no evidence here that the target lacks disputed exact histories, and no measured target-win count. No practice scoring was requested, leaving all three allowed practice scores unused. No exam or training recordings were accessed.

## Result

The premise remains unresolved. An infrastructure failure is not a premise kill, and no warning suppression is justified by the current evidence. Temporary diagnostic copies were removed; all engine code, existing tests, scorer, data, manifests and experiment card remain unchanged. Resume with the exact-predecessor audit when supervisor Modal execution works. If the target's supporting history has no consume-versus-skip disagreement before confirmation, kill on that observation; otherwise implement the provenance interface and checker rule, add permanent behavior tests, and validate the built change under the experiment's target-win rule.

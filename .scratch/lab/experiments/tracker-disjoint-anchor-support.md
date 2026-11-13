# Disjoint acoustic support for skip anchors

Status: implemented; target-premise measurement and full build/Core/SDK validation remain blocked by the supervisor's Modal runner. The prototype is retained for those checks. No practice score or target win is claimed.

## Prediction

Card `tracker-disjoint-anchor-support`, tracker, explore. Predicted practice phone change: tracker −1 and overall −1, with no counted losses. Target: `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`. The experiment card is unchanged.

## Mechanism implemented

`local.ts` carries a half-open observed-token interval beside each existing cost, reference origin and edit count. Consuming transitions extend the interval; a jump starts another interval. Recentring and save/restore preserve it using the same replacement-array ownership as the cost state. Token coordinates are session-local and independent of reference offsets and caller push boundaries.

`tracker.ts` exports the support interval on accepted global and ordinary local anchors. On a published local run change it freezes the previous accepted source frontier. This matters because a destination path can start while unselected, before later source progress is published. Its destination witness excludes every observation before that frozen frontier. Already-disjoint anchors retain their existing eligibility. For overlapping anchors, the tracker re-fits the destination from the remaining observations with `anchorFitCost`, retaining the independent prefix and applying the existing three-observation trailing confirmation requirement and its unchanged cost threshold. No new numerical threshold is introduced. The diagnostic can inspect both the support overlap and the recomputed fit.

A separate bounded observation buffer uses the existing acquisition history limit. Once independent support qualifies, it remains qualified for that accepted run. Later destination speech can therefore establish a witness without changing the chosen cursor or reconstructing a checker window.

`checker.ts` retains this eligibility with the destination ayah. The ayah-absence ledger withholds a candidate when its destination lacks independent support, leaving the gap available for a later qualifying lock. A later weak entry does not erase independent support already accepted in the same ledger history. The new fields are optional for existing callers supplying their own position events. The alignment's word edits, seam handling, tracker selection, movement timing, window construction and lock-drop handling retain their existing logic.

## Validation performed

- Four new unit tests cover insufficient fresh observations, a rhyme suffix that cannot independently establish its destination onset, token identities and save/restore, and run-support reset while preserving location/cost behavior.
- Two new ledger unit tests cover withholding unsupported candidates and preserving genuine skips with independent support, including support arriving later and support retained after a later weak entry.
- Because this worktree has no dependencies and Modal failed before execution, a lightweight local Node assertion harness executed the four anchor-support tests and ten existing/updated ayah-ledger tests against TypeScript-stripped engine copies. All 14 checks passed. This was a smoke check, not a Vitest or TypeScript type-check result.
- A separate local comparison replayed 115 observations, including a backward restart, through the original and modified local matchers. Position, origin, edit count and both cost fields were identical at every step; the new support fields were omitted from that comparison.
- Node's TypeScript transform parsed the 24 top-level core modules. `git diff --check` passed. No scorer, lab tooling, test data, evidence or manifest was edited.

## Modal requests and limits

The previous attempt prepared an audit but stopped without implementing engine code. This attempt implements the mechanism and updates that audit for the prototype.

Authorized run 4 requested the target audit; run 5 requested `bash tools/autolab_tests.sh` on six shards. Both returned `NotFoundError: No such file or directory`, with zero reported compute spend and no shard logs or result artifacts. The commands never reached their diagnostic or test steps. Full build and Core/SDK suites therefore remain unvalidated.

The target's source/destination overlap, independent commitment result and warning outcome remain unmeasured. The card's premise kill criterion has neither fired nor been disproved. The supplied builder criterion also requires killing a built change with fewer than three target wins; its count cannot be determined from these infrastructure failures. Practice scores requested: **0 of 3**. The forecast of one recovery is below the fold gate by itself.

The bounded support buffer cannot reconstruct an overlapping independent prefix once it has retired, unless a witness already qualified. Short genuine jumps may lack sufficient independent confirmation. Tracker-event equivalence over the complete target, collateral warning changes, clean sessions, studio audio and reader robustness still require the authorized remote validation and gate.

## Prepared target comparison

The diagnostic now compares the instrumented prototype with the uninstrumented prototype and the unchanged engine at revision `85adfbf7`. It asserts exact equality for instrumentation, and equality of every tracker event against the unchanged engine after removing only the new support fields. It records nearby anchors, support intervals, independent fits, withheld gaps, ledger neighbours and both checker event streams. It reads the listener and set from the unchanged card.

Archive the unchanged sources locally before uploading (generated copies are ignored):

```sh
python3 .scratch/lab/experiments/tracker-disjoint-anchor-support/prepare-audit.py --baseline 85adfbf7
```

Then use the authorized supervisor:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 20 -- 'python3 .scratch/lab/experiments/tracker-disjoint-anchor-support/prepare-audit.py && pnpm --filter @tilawa/core exec tsx ../../.scratch/lab/experiments/tracker-disjoint-anchor-support/replay.ts lab-v1.5-p46 121.84'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- 'bash tools/autolab_tests.sh'
```

If target support is already disjoint, independent destination support still qualifies, or the built prototype wins fewer than three targets, revert the engine/test changes and record the measured premise kill and `.autolab/outcome.json`. Otherwise leave scoring and fold disposition to the supervisor, following the supplied practice-score rule. No fold approval has been requested or performed.

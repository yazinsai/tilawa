# Ordinary word boundaries already preserve local edit state

Status: killed on the premise. Targets won: 0. No practice score used.

## Prediction

Tracker −1, overall 0: recovering the p46 `54:36:1` false skipped-passage warning could expose the documented nearby listener warning. No standalone fold was forecast.

## Proposed mechanism

Carry one acoustic edit recurrence across ordinary reference-word boundaries, treating word ownership as annotations. Preserve the phone strings, edit prices, movement edges, whole-word deletion charge, commitment margins and checker handling. The premise requires an ordinary boundary to discard acoustic state and thereby change continuation-versus-restart costs independently of the phones.

## Source audit

The premise is absent in this checkout. `tracker.ts` delegates local alignment to `createLocalSearch` in `packages/core/src/local.ts`. That search already advances a single DP row over `ReferenceIndex.phonemes`, the continuous concatenation of the reference phones.

- Insertion reads the previous row's same reference cell and carries its origin and edit count (`local.ts:113`).
- Match/substitution reads the previous row's preceding reference cell and carries its origin and edit count (`local.ts:117`).
- Reference-only deletion reads the current row's preceding cell and carries its origin and edit count (`local.ts:134`). Its closure continues through ordinary word boundaries.
- `insertionCost` and `deletionCost` inspect the continuous reference string, including sound runs across annotations. There is no per-word slice or insertion/deletion-state initialization here.
- Ordinary word starts are consulted only to admit explicit movement arcs (`local.ts:126`). These retain accumulated cost and add the existing two-edit jump charge and distance term, while resetting run origin and run edit count. Removing those arcs would change movement semantics, which this experiment must preserve.
- At ayah-window recentering, every overlapping cost, origin and edit-count cell is copied (`local.ts:84`). Global reacquisition explicitly creates a new local search; ordinary word progress only publishes a position event and does not recreate the search.

For fixed movement edges, the ordinary acoustic recurrence contains no word-boundary variable. Changing only ownership annotations cannot change its numerical costs: identical initial rows produce identical insertion and substitution candidates, then identical left-to-right deletion closure, then identical movement candidates and row normalization. This repeats for every observed token. There is no active affine-gap or other acoustic-history state being cleared at a boundary.

This applies to both the target's nearby continuation and rhyme restart: they traverse the same continuous acoustic recurrence, with the restart retaining its intentional movement charge. Implementing another continuous composition would leave their relative acoustic costs unchanged. That fires the card's premise kill criterion; no prototype engine change is warranted.

## Validation and limits

Supervisor Modal run 1 requested the supplied p46 trace at 121.84 s. Run 2 requested `bash tools/autolab_tests.sh` on six shards, which builds core/SDK and runs both test suites. All three requests returned `NotFoundError: No such file or directory` without shard logs or test results. These are validation infrastructure errors, not an engine test failure.

Run 3 requested a temporary diagnostic comparing every target DP row with a continuous Bellman recurrence, checking product-event identity against the original implementation, and checking all 32 segmentations of a six-phone synthetic reference over seven observed strings with movement destinations held fixed. This diagnostic instruments copied modules only. It did not return a result, so none of those runtime checks is claimed as passing. The temporary script was removed.

No practice score has been requested. No engine, scorer, test-data or card changes have been made, and no new unit test is needed because no engine behavior was changed. The conclusion is a source-level falsification, not a fresh measured replay of the target or a passing test-suite claim.

Only this write-up and `.autolab/outcome.json` are retained. The outcome records `killed` with zero target wins and omits an unmeasured overall score.

The source result rejects this boundary-reset mechanism; it does not establish why the rhyme restart is acoustically preferred or repair the remaining warning.

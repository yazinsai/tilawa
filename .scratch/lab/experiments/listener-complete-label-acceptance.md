# Complete pronunciation acceptance

Status: killed on source-level premise; no runtime engine change.

## Prediction and mechanism

The unchanged card predicts listener −1 and overall −1, with no losses. The
proposed mechanism is acceptance of an unfinished rival pronunciation in the
string-path CTC score. The intended correction requires every label to have
been consumed before acceptance, allowing final-label continuation and trailing
blank. Written and accused hypotheses must retain identical completion rules,
windows, pronunciation labels and confirmation bars.

The task requires stopping if the mechanism is wrong or the built change wins
fewer than three targets. This experiment stops because the existing scorer
already implements the stipulated completion semantics. No alternative veto,
duration topology or boundary-state change is substituted for the card.

## Finding

The SDK creates the core checker in `packages/sdk/src/session.ts`. String-path
support in `packages/core/src/checker.ts` calls `PosteriorFrames.nll` from
`packages/core/src/ctc.ts`; there is no separate SDK string-path forward scorer.

For a pronunciation of L labels, `nll` builds 2L + 1 states, alternating blank
and label. The accepting states are already exactly state 2L − 1, which emits
the final label, and state 2L, the trailing blank. Line 129 sums only those two
states. It never takes the best prefix or sums all reachable states.

The recurrence can stay in a state, move forward one state, or skip an
intermediate blank between unequal labels. None of these transitions skips a
pronunciation label. Consequently a path reaching either accepting state has
consumed every label. The final-label self-loop allows continuation; the final
blank self-loop allows trailing blanks. Equal consecutive labels require a
separating blank. An impossible completion returns Infinity, including a
nonempty pronunciation with no remaining frames, rather than a finite prefix
score. The string-path support calls supply no repeat arcs.

This argument holds for both written and accused hypotheses on any posterior
window. Renaming the two terminal indices as explicit accepting states cannot
remove unfinished mass because that mass is already excluded. The four
registered confirmations therefore have no exposure to this proposed defect;
the mechanism's target-win upper bound is zero. This is a structural conclusion,
not a measured practice delta or a completed target replay.

A tiny emission probability for a required label still describes a *complete*
path. Treating such a path as unfinished would introduce an acoustic threshold,
which is outside this experiment's completion-only mechanism.

## Validation and limits

Added `packages/core/test/ctc-acceptance.test.ts` with exhaustive-path comparisons
and checks for impossible prefixes, final-label continuation, trailing blanks,
empty text, and blank-separated repeated articulations. These are regression
tests of the existing contract; engine behavior is unchanged.

A lightweight local Node 22.22.2 diagnostic executed a copy of the shipped CTC
source using Node's built-in TypeScript transformation. The sole source-text
adjustment redirected the evidence import to its original TypeScript module.
Across 243 combinations of synthetic windows and complete pronunciations, its
probabilities matched exhaustive enumeration of collapsed acoustic paths within
8.326672684688674e-17. It also rejected an unfinished two-label pronunciation
despite 0.98 probability for its first label, and rejected a two-frame repeated
articulation without enough frames for the separating blank. The diagnostic and
result remain in ignored `.autolab/` files.

Supervisor Modal run 1 requested the focused Vitest tests and paired original
versus independently completed-score replays on all four card-listener target
sessions, through the free engine and true-position checker. Run 2 requested
the supplied six-shard build/test harness. Both failed before command execution
with `NotFoundError: No such file or directory`, zero reported compute spend,
and no command logs. No target replay ran. Run 3 retried the full build/test
harness and failed before execution with the same error and zero reported spend.
Thus neither the focused Vitest run nor the full core/SDK suites could run here.

The local exhaustive check does not substitute for the core/SDK Vitest suites
or target listening review. No independent audio opinion was requested because
the finding concerns the scoring graph, not an unresolved listening judgment.
No practice scores were requested, and no population gain, loss, clean-session
change or studio change is claimed.

The engine, experiment card, model manifests, evidence, test data, scorer and
lab tooling are unchanged. The remaining diff contains regression tests and
this write-up. No fold was performed.

# Listener pruning-bound certification

Status: killed on the premise. No engine change retained.

## Preregistered prediction

Card `listener-pruning-bound-certification`, listener, explore; practice set
`lab-v1.5u`, listener `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`.
Prediction: listener −1 and overall −1, with no predicted losses.

Audited branch head: `258cb4b5a097f070582bde75eb0b53955b6876bb`.
The experiment card was not edited.

## Proposed mechanism

An unfinished acoustic hypothesis might be discarded using its best CTC path
as a bound on its completed forward-summed evidence. A best-path score omits
alignment multiplicity: `max(log p(path))` can be smaller than
`log sum(p(path))`. Such a bound would not be admissible for a sum-product
objective. The card proposes correcting only those bounds, with exhaustive
fallback inside the existing seam window, while preserving pronunciation
alternatives, eligibility, exclusions, neighbour certification and teacher
confirmation.

## What the audit established

The relevant pipeline already performs complete acoustic sums without a
search frontier or numerical pruning bound:

| Stage | Implementation | Finding |
| --- | --- | --- |
| Listener discovery | `packages/sdk/src/seam-insertion.ts:39–49` | Runs the core checker on primary evidence and collects complete candidate records. It does not expand acoustic prefixes or compare unfinished scores. |
| Span discovery | `packages/core/src/checker.ts:1901–1979` | Constructs complete pause-bounded utterances, orders them longest first, and checks spelling, local-text exclusions and excised-neighbour margins. It has structural eligibility checks and first-permitted-candidate stopping, but no bound on remaining acoustic mass. |
| Lexical lookup | `packages/core/src/checker-lexical.ts:226–290` | Uses normalized consonant skeletons and the existing complete-string spelling predicate. It is not an acoustic beam or unfinished-path search. |
| Neighbour certification | `packages/core/src/seam-excision.ts:25–43` | Explicitly enumerates the Cartesian product of canonical/pausal forms for the read hypothesis and each omitted-neighbour hypothesis. Every completed form combination receives a full CTC score. |
| CTC evidence | `packages/core/src/ctc.ts:100–130` | Visits every chain state on every retained frame and log-adds self, advance, skip and repeat transitions. No state is removed because it trails the current best score. |
| Teacher dispatch | `packages/sdk/src/seam-insertion.ts:56–88` | Evaluates every collected candidate after structural checks. The cache shares crop rows, not hypothesis scores. |
| Teacher confirmation | `packages/core/src/seam-insertion.ts:84–108` | Visits every graph arc on every frame in both forward and backward recurrences. Computes the optional-word marginal from summed entrance mass divided by summed terminal mass. |

Two uses of extrema must not be confused with best-path pruning. The
neighbour helper's `Math.min` chooses the best **completed pronunciation
combination**, each already scored by forward sum. The teacher helper's
`Math.max(a, b) + log1p(exp(-abs(a-b)))` is the stable implementation of
**log-sum-exp**, not a Viterbi maximum. The final marginal clamp confines a
completed probability to [0, 1]. None supplies an unfinished-search bound.

The proposer does stop after a permitted candidate and caps some local-text
pronunciation combinations in its exclusions. Those are arbitration and
exclusion semantics, respectively. Changing them would violate this card's
instruction to preserve eligibility and exclusions; neither is the proposed
best-path bound on forward-summed acoustic evidence.

Consequently, no registered target can be discarded by the specific bound
this experiment proposes to replace: that operation does not exist in the
shared seam pipeline. This is a source-level premise result, not a claim that
every possible word span or pronunciation is eligible, or that any target's
audio was independently heard correctly.

## Validation

The existing `packages/core/test/seam-insertion.test.ts` includes independent
enumeration of all short CTC frame-label paths and compares their probability
sum with the production marginal to 12 decimal places. It also covers equal
labels requiring blank separation, repeat-arc agreement with closed CTC sums,
and duplicate pronunciation invariance. Existing seam-excision tests cover
cut-frame concatenation and canonical/pausal competitor treatment. SDK tests
cover candidate-required teacher dispatch and primary-evidence preservation.

Requested the unchanged build and complete core/SDK suites through the
authorized supervisor:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- 'bash tools/autolab_tests.sh'
```

Supervisor request: 1. The request was accepted at `2026-11-10T15:24:08Z`,
but no container output or result was available at handoff. These tests are
requested, not reported as passing; the kill rests on the source audit.

No unit tests were changed because no product behaviour was changed. No
practice scores were requested (0 of the permitted 3), and no acoustic model,
manifest, evidence, test data, scorer or lab tool was modified.

## Result and limits

Killed on the absent pruning premise; targets won: 0. No speculative engine
patch was made, so there was nothing to revert. There is no measured practice
delta and no target replay or new acoustic decode in this audit. The source
inspection and existing numerical-oracle coverage establish the narrow
absence of best-path pruning in these sum-product computations; they do not
prove that the probability model, structural candidate set or target warning
decisions are otherwise correct.

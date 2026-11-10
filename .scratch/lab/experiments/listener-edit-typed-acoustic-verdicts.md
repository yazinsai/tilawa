# Edit-typed acoustic verdicts

Status: killed on the premise by source audit. No engine change retained.

## Prediction and proposed mechanism

The card predicts listener −1 and overall −1 on lab-v1.5u, targeting
`lab-v1.5-p43/phone/warning:added_word/22:13:3/211.60` with listener
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. The experiment card is
unchanged.

The supplied incident history says that the suspicion changes from addition to
omission 200 ms later and confirms only as an omission. The proposed explanation
was that an acoustic verdict belonging to the addition survived that change.
The proposed interface would bind a verdict to its edit class, complete read and
error hypotheses, and frame interval, and recompute an unmatched comparison.
Models, evidence, margins and issue timing would be preserved.

The kill criterion is semantic: stop if the final omission already gets an
omission-specific comparison and consumes no support from the earlier addition.

## What the source establishes

Audited starting commit `feb25527de5df41ed9f9dc85d7a381eee8881ca5`.

- `packages/core/src/checker.ts:1507` builds omission gaps afresh from the latest
  alignment. The constructors at lines 1567 and 1630 contain only the omission
  class, word range and observation coordinates. The final gap record at line
  1620 does not inherit the earlier open addition's fields or acoustic decision.
- At `checker.ts:2292`, reclassification assigns `o.issue = now`. This is full
  replacement, including any optional comparison fields. The issue identifier
  and suspicion start time survive; an acoustic verdict is not stored in `Open`.
- At `checker.ts:473`, `decide()` calls `supported(o.issue)` on the current issue.
  The other support consumer, `sureOf()` at line 694, does the same. There is no
  issue-id support cache.
- `supported()` at line 486 computes its neighbors and interval from the current
  class, latest alignment and current observation bounds. For an addition the
  right neighbor is the accusation's word; for an omission it is the next word
  after the omitted range. Lines 537–545 build the two complete hypotheses and
  call `frames.nll()` separately for each. The omission hypothesis deletes the
  accused range; it does not contain the former addition's heard speech.
- `packages/core/src/ctc.ts:100` initializes a new CTC forward recurrence for
  every `nll()` call. Its cache stores individual posterior rows, not label-chain
  likelihoods or support verdicts.
- The actual SDK auxiliary interface is `SeamInsertionEvidence` in
  `packages/core/src/evidence.ts:4`. It certifies only proposed additions. The
  consumer key at `checker.ts:259` includes word, observed spelling and exact
  insertion bounds. Its sole lookup at line 1966 can produce only an
  `added_word` issue with `lexicon: true`; `supported()` never reads that map.
  The excision cache likewise produces only fresh addition records. Neither
  mechanism can certify an omission.
- The SDK's warm-crop cache at `packages/sdk/src/seam-insertion.ts:54` stores
  posterior frames. Each candidate's marginal is computed at line 85 from its
  own forms and interval. The end-of-session publisher at
  `packages/sdk/src/session.ts:202` accepts only confirmed additions; it cannot
  use an insertion certificate to publish an omission.
- There is one stored derived boolean: `frozen` at `checker.ts:303`. A slot is
  removed from `open` when frozen, after `sureOf(o)` evaluates its current issue
  (lines 707–710). Release emits that same issue snapshot (lines 698–704).
  Streaming reclassification updates only entries still in `open`, so a frozen
  addition's boolean does not transfer into a later omission.

This disproves the hypothesized semantic reuse in the current implementation.
Given the card's supplied reclassification, the omission cannot inherit an
addition verdict through these consumers. Adding an edit tag would not remove
this target. **Targets won: 0 (source-based premise result, not a scored delta).**

## Validation attempts and limits

Supervised Modal run 1 requested the supplied true-position trace tool for p43
with the card's listener. Run 2 requested an ignored diagnostic probe that logs
reclassification, complete support hypotheses, intervals, likelihoods and
certificate lookups, and compares all replay events against the uninstrumented
engine in free and true-position modes. Both requests failed before command
execution with `NotFoundError: No such file or directory`; the supervisor
reported zero compute spend and produced no trace artifacts. These failures do
not establish anything about the audio or acoustic likelihood margin.

Run 3 requested the unchanged six-shard `bash tools/autolab_tests.sh` build/test
harness. It also failed before execution with the same `NotFoundError` and zero
compute spend. No build or test result is available. All three supervisor error
responses are retained in ignored `.autolab/modal-result-{1,2,3}.json` files.

The supplied 200 ms incident history was not independently replayed here. Exact
confirmation times, omission interval, numerical likelihoods and posterior
availability at that instant remain unmeasured. If support is unavailable, the
existing policy can confirm with an indeterminate comparison; that behavior
still does not reuse the old addition verdict and is outside this card's
mechanism. This is a source audit, not an independent listening judgment.

No engine behavior was changed, so no new unit tests were added. No practice
score was requested (0 of 3 used), and no aggregate score change is claimed.
No model bundle or `.autolab/model.json` is declared. The only retained tracked
change is this result; `.autolab/outcome.json` records the killed premise.
No scorer, lab tooling, test data, manifest, evidence or card was edited. No fold
was performed.

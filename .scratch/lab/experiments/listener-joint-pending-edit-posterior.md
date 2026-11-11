# Joint pending edit posterior

Status: implemented; target premise and full validation remain unmeasured because
supervisor Modal execution fails before container startup. The engine change is
retained for supervisor validation. The experiment card is unchanged.

## Prediction and mechanism

The card predicts listener −1 and overall −1, with no predicted losses. The two
registered cases are the false wrong-word accusations in p11 at 104.32 s and p55
at 107.08 s. The proposed cause is competing unresolved lexical edits assigning
shared acoustic evidence to the wrong word. These cases remain hypotheses, not
reported recoveries.

The SDK now scores a closed union of complete CTC paths over the available
current and following ayah. Its hypotheses come from the primary checker's
existing lexical substitutions and additions, captured before independent
support filtering and the wrong-word/addition tail merge. It includes unchanged
text and compatible combinations, generates no free phones, deduplicates
hypotheses, and makes alternatives for the same event mutually exclusive.
Insertions and replacements can coexist despite overlapping greedy spans because
the CTC recurrence marginalizes their acoustic boundaries.

Every explanation consumes each original posterior frame once. Each edit pays
the existing PHRASE_SWITCH prior once outside the frame recurrence; existing
word and ayah repeat arcs remain available. The bounded union permits at most
128 explanations. Overflow, invalid input, missing rows and an impossible
independent comparison return unavailable evidence rather than a certificate
from a pruned graph.

The comparison sums all paths replacing the accused word against all paths
reading it correctly with at least one neighboring edit. A veto requires the
latter mass to exceed the replacement mass and the independent target-only
comparison to have favored replacement. Unchanged-text mass participates in
normalization but cannot by itself supply the neighboring-edit certificate.

## Streaming and recorded-evidence integration

The primary checker settles its original issue state on its existing schedule;
publication of an eligible nonlexical string-path wrong-word decision waits for
the two-ayah horizon or the end of the passage. Snapshots retain competing edit
hypotheses and their posterior rows across ordinary window commits. A temporary
far-ahead alignment cannot close the window by itself: release also requires
primary tracker progress beyond it, or an observed ending of the following ayah
when no tracker positions are supplied. Losing or changing the tracked passage
closes the retained window.

The live SDK invokes the joint scorer at that horizon. Recorded evidence also
receives a positive certificate from a read-only primary replay, carried on the
first phoneme through the existing token-preserving serialization route. The
checker accepts it only for the same word, observed sequence and original frame
span, with valid probability masses, once its available frame has arrived. EOF
cannot make evidence from a future frame available. A successful veto publishes
an uncertain copy of the original event, retaining its original timestamp.

The original internal confirmation remains intact for neighboring tail merges
and confirmation procedures. Primary tokens, posterior values and tracker
inputs are unchanged. Lexical accusations, parallel-graph confirmations and
manner warnings retain their existing paths. The rule can suppress an existing
accusation; a genuine weak substitution may therefore be lost. Pending
publication adds latency, potentially until the following ayah finishes.

## Validation and limits

Read AGENTS.md and inspected the checker, CTC forward scorer, lexical candidate
construction, SDK certificate transport and audio-review instructions. The
previous horizon-conditioned experiment branches named in the brief are absent
from this git snapshot; no earlier implementation is claimed to be reproduced.
The previous attempt's saved prototype was completed with horizon retention,
live SDK judgment and recorded-evidence consumption; the retained diff now
contains engine code.

New tests cover exhaustive tiny CTC path sums and once-per-event priors,
replacement reversal requiring a neighbor, duplicate and incompatible
hypotheses, overlapping greedy spans, graph overflow, invalid inputs and missing
rows. Checker integration tests cover original event preservation, certificate
identity and availability, rejection of invalid and future certificates,
decoded-only evidence, lexical confirmations, and late neighboring hypotheses
surviving commits without changing their confirmations. The late-neighbor test
exposed and fixed premature release from a temporary far-ahead alignment.

The added tests were exercised locally using Node 22's TypeScript transformation
and a small ignored Node assertion adapter for their Vitest assertions. All
10 added tests passed in the final run (six SDK scorer tests and four checker
integration tests):

```sh
node --experimental-transform-types --import ./.autolab/local-register.mjs \
  --test packages/sdk/test/pending-edits.test.ts packages/core/test/checker-pending-edits.test.ts
```

This is lightweight behavioral validation, not a TypeScript build or the full
Vitest suites. The synthetic fixture uses deliberately unambiguous posterior evidence
above the existing support gate to exercise transport and timing; it does not
establish the acoustic premise on either practice case.

All eight authorized supervisor requests across the initial and resumed work
returned NotFoundError: No such file or directory, with no executed-container
results. These included target traces, joint-posterior probes, and the supplied
six-shard build/core/SDK test harness. The latest requests were run 7 (the two
target probes) and run 8 (the full test harness). Infrastructure failure is not
recorded as a premise kill.

No target posterior, target recovery, full-suite pass, independent listening
judgment or aggregate wrong-call measurement was obtained. Zero of the three
practice-score allowances were used. No gains, loss counts or fold eligibility
are claimed. The generic diagnostic helper in
packages/sdk/test/pending-edits-probe.ts reports discovered hypotheses,
posterior comparisons and before/after events when Modal execution is restored;
it reads the existing scorer for replay and does not modify it.

No scorer, lab tooling, test data, manifest, evidence asset or experiment-card
edits were made. No new model assets are needed or declared. No fold was
performed and no killed outcome is declared without a measured premise result.
The supervisor must run full tests and the target probe before deciding the
premise, then apply the practice scoring and fold gate as appropriate.

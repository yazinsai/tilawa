# Formula residue discovery

Status: implemented; supervisor validation and target recovery remain unmeasured.

## Prediction and mechanism

The card predicts one recovered added word, with listener −1 and overall −1.
A permitted formula and an extra word may share an existing pause-bounded
utterance. Whole-utterance word spelling then fails even when the extra word
would spell successfully on its own.

The discovery grammar is `formula* word formula*`, with at least one
formula and exactly one complete residual lexical word. Every segment consumes a
nonempty, disjoint token interval, and the intervals must cover the utterance.
Only existing formulas and spelling costs participate. Formula segments
retain their exemption; the residual word must pass the existing neighbour,
prefix, back-run and local-substring exclusions, posterior neighbour
certification and original-audio teacher insertion gate. The prediction is below
the task's three-target threshold for requesting a practice score; that threshold
is applied to measured recoveries, not the card's prediction.

## Implementation

The SDK discovers candidates through a read-only tracker/checker replay
in `packages/sdk/src/seam-insertion.ts`. The pause-bounded utterance enumerator
is in `packages/core/src/checker.ts`. Its original path rejects an utterance containing any
formula-labelled token, then requires the entire utterance to pass
`lexicalWords(index).surplusWord`. A candidate must also pass the local-fragment
exclusion and neighbour certification after excision. The SDK subsequently
verifies its exact word spelling on original audio with the existing teacher
insertion lattice and carries the resulting marginal as auxiliary metadata.

`packages/sdk/src/formula-residue.ts` now enumerates residual word intervals using
forward and backward reachability through complete formula intervals. Intervals
are nonempty, disjoint, and cover the original utterance together. Formula-only
complete parses take precedence and remain exempt. In particular, a short word
that the existing formula tolerance can absorb does not become a new candidate.

The formula list and formula-boundary matcher live in the shared core helper
`recitation-formulas.ts`. The list is the existing phrase asset plus the existing
basmala, and the cost limit remains 0.3 per formula phone. The matcher uses the
same weighted edits and alignment direction as the existing formula predicate,
but closes both boundaries: leading and trailing observed phones must pay their
edit costs. A substring alignment cannot silently leave tokens unaccounted for.

An SDK-only checker hook supplies these residuals after all original
whole-utterance candidates. The residuals use the unchanged normalization,
whole-word spelling, neighbour/prefix/back-run/local-substring exclusions,
posterior neighbour certification, and teacher insertion marginal. No formula
labels, pause boundaries, primary phones, posterior rows, tracker inputs, or
teacher acceptance thresholds are changed.

Completed verification travels through the existing `SeamInsertionEvidence`
metadata, with no schema change. Callback-free replay requires the exact word,
observed spelling, token-frame endpoints and marginal strictly above 0.5. It also
rechecks complete formula accounting outside that interval in the original
utterance, so an otherwise valid certificate cannot authorize an arbitrary
fragment.

## Validation and limits

Ten new behavioural tests cover both word/formula orders, formulas on both
sides, repeated formulas, formula-only exemption, full boundary costs,
unaccounted speech, partial/two-word residuals, unchanged lexical exclusions,
missing outer pauses, and exact certificate replay. A synthetic blurred-onset
integration fixture reaches the SDK's ordinary teacher path using a mock teacher
with affirmative synthetic posteriors. It checks that the teacher receives a
view of the original audio and that primary evidence and audio remain unchanged.

A lightweight Node TypeScript loader and assertion harness executed all ten new
test bodies successfully, plus all seven existing core excision test bodies.
This is a source-level assertion check, not a TypeScript build or a Vitest run;
it does not replace the requested full suites. The harness is ignored under
`.autolab/` and is not part of the engine change. `git diff --check` passes.

The authorized supervisor runner has failed before producing container output:

| Run | Request | Result |
| --- | --- | --- |
| 1–7 | Data inspection, target audits, minimal `pwd`, six-shard build/tests | `NotFoundError: No such file or directory` |
| 8 | Six-shard build/tests with the implementation and new tests | `NotFoundError: No such file or directory` |
| 9 | Target audit using the built parser plus closed-alignment premise checks | `NotFoundError: No such file or directory` |

The minimal command failure establishes that the error is not evidence against
the parse mechanism. The runner provided no more specific missing path. Run 9's
result records zero compute spend and only that error, with no container output.

The target diagnostic enumerates existing native-clock pause-bounded utterances
near all six registered incidents, reports the built parser's planted-word
matches, and separately checks complete disjoint formula/word parses with the
same costs. It reads labels only to select/report incidents. It remains ignored
in `.autolab/formula-residue-audit.ts`; no target data has executed successfully.

The experiment card, scorer, lab tooling, supplied data, manifests and evidence
files remain untouched. No new model bundle is required. No practice score was
requested; all three scoring allowances remain available. No fold was performed.

There is no measured target-win count, listener delta or overall delta, and no
basis for declaring the experiment killed. The implementation is left in place
for supervisor validation. If the premise is absent or the built change recovers
fewer than three targets, revert the engine change and record that measured kill.
If it recovers three or more, request a practice score before deciding. Complete
accounting and the unchanged teacher gate limit new false warnings but do not
establish precision; clean-session, studio and other incident effects still need
the supervisor's practice evaluation.

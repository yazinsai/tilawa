# Substitution and duration-run binding

Status: killed on the observation premise. No engine change retained. The
experiment card is unchanged.

Baseline commit: `ee2ff8c0`. Practice set: lab-v1.5u, phone, with the card's
listener `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`.

## Prediction

Tracker −1, overall 0. The proposed recovery is
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
Correcting that binding may expose the previously documented nearby listener
false warning; no standalone fold was predicted.

## Proposed mechanism

Retain an explicit correspondence between a substituted reference articulation
and its observed sound. Consume subsequent pronunciation-generated duration
repetitions using available uninterrupted continuations of that same observed
sound. End the correspondence at a different articulation, a word boundary or an
observed interruption. Preserve duration charges when the continuation is absent,
and preserve movement costs, commitment margins and checker handling.

The supplied earlier audit reports one identity substitution and two duration
charges on the near continuation, which loses by 0.3915 edits. Those figures are
context from the card's history, not measurements reproduced in this experiment.

## Observation and implementation audit

The proposed consuming transition needs an additional observation from the same
uninterrupted sound run. The primary decoder does not export such continuations:

- `packages/sdk/src/listener.ts::createListener` extends `open.end` while the
  nonblank identity is unchanged. It emits that run once. When a run crosses an
  inference-window boundary, `carried` suppresses further emissions of that same
  run. Another token of the same identity requires an intervening blank or another
  sound, rather than an uninterrupted continuation token.
- The existing listener unit tests compare emitted tokens with greedy CTC runs
  across push sizes. Their all-one-token backend explicitly expects one emitted
  token even when the sound continues through every inference window.
- The card's insertion-teacher path keeps every primary token intact.
  `carryInsertionEvidence` adds auxiliary metadata; it neither splits a token nor
  supplies a new primary sound. The teacher's own decoder also collapses consecutive
  equal nonblank frames.
- `packages/core/src/bench/index.ts::chunk` preserves tokens whole. The SDK's
  evidence clock and insertion cells also carry existing tokens without expanding
  their sound runs.
- `trackerInput` preserves primary phonemes. `tracker.ts::follow` calls
  `local.step(ph.p)` once per token. `local.ts` computes ordinary consuming
  transitions from the previous cost row; reference-only deletion closure consumes
  no observation. `fuzzy.ts::deletionCost` discounts shortening of the expanded
  held run but still charges it.

Thus an uninterrupted continuation cannot be rejected solely for disagreeing with
the reference identity: it is already inside the one CTC token, or suppressed by
the decoder's carried-run handling, rather than available as another token to
consume. An intervening blank or sound ends the uninterrupted emission run. A
binding change that advances duration repetitions without an additional
observation would make those repetitions free under the existing token matching
contract.

Using a token's frame width as several reference duration units would instead
require a duration-to-occupancy rule, including how many reference units that
width supplies. The current local matcher contains no such rule to preserve after
a substitution. Introducing one would be a different duration-model experiment,
not restoration of a lost token correspondence.

## Validation and limits

A small independent check enumerated all 3,279 nonempty sequences of up to seven
frames over blank and two sound identities. Greedy CTC collapse always left a
positive intervening frame gap between consecutive tokens of the same identity.
This checks the collapse invariant, not the engine's unit suites or the target's
stored evidence.

Supervisor Modal requests:

1. The supplied trace CLI, for the target at 121.84 seconds, using the card's
   listener.
2. The unchanged build and core/SDK suites through `bash tools/autolab_tests.sh`,
   with six shards.
3. A bounded diagnostic copy of the tracker and local matcher. It retains the
   original numeric recurrence and adds predecessor records to inspect near-best
   paths and the observed frame spans around duration deletions. The copies are
   temporary diagnostics, not engine changes or scorer edits.

All three requests failed with `NotFoundError: No such file or directory`, with
no returned logs and zero reported compute spend. The temporary diagnostic copies
and their preparation script were removed after the requests finished.

No practice score was requested (0 of 3 used). There is no measured overall or
component delta, no reproduced target traceback, and no passing build or unit-test
claim. In particular, this audit does not establish whether the target's emitted
frame span would support a separately calibrated occupancy model, and a CTC blank
does not by itself prove a physical pause in the recitation.

## Decision

Kill on the observation premise of the specified token correspondence: the primary
stream has no additional uninterrupted same-sound token available for the proposed
duration-consuming transition. Do not remove duration charges or introduce a new
frame-occupancy model to force a result. The killed outcome records zero target
wins for this unchanged-engine experiment, not a measured practice delta.

No engine behavior changed, so no unit-test changes are retained. The retained
files are this write-up and the required outcome; the card, checker, scorer,
tooling, labels and evidence are unchanged.

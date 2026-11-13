# Tracker progress status transitions

Status: killed on the mechanism after a source audit. No engine change retained. No practice score.

## Prediction

Tracker −1, overall −1 on lab-v1.5u phone practice with listener
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. The sole target was
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.

## Proposed mechanism

Distinguish tentative word entry, partial acoustic matching and completed matching
in notification identity. Publish a partial-to-complete transition at the same word
and run origin, retaining its consumed token interval. Preserve location selection,
costs, timestamps of existing events, lock drops and checker confirmation rules.
The intended recovery was delivery of previously unannounced progress, without
rebinding speech or adding a coverage veto.

## What the source audit established

The location deduplication exists at `tracker.ts:1183–1207`: `localEvent` suppresses
a locked candidate when its word and run-start word equal the previous event's.
That suppresses updated confidence, cost and PER as well as changes of the internal
phone offset. However, the candidate/event contract in `events.ts` has no acoustic
match status or consumed-token interval. `LocalPosition` in `local.ts` likewise
contains an offset, origin, edit count and costs, without a completed-match
notification. Its offset can advance by reference deletion, so reaching a word
boundary alone cannot certify an acoustic completion.

More decisively, deduplication does not gate evidence delivery. In
`checker.ts:2464–2485`, every push appends its posterior rows and phonemes, including
when the position-event list is empty, and evaluates whenever an anchor exists.
`judge` constructs acoustic visits from that evidence. Its existing qualifying
visits update `readAyah` under each visited word's own ayah at
`checker.ts:1410–1421`. Complete lexical visits are also judged locally, rather
than received as tracker completion events (`checker.ts:1645–1647`).

The ordinary position consumer (`checker.ts:2372–2445`) uses the word, span origin,
lock state and explicit hold/relocation fields. Re-emitting a normal locked event
with the same word and run origin does not supply an additional acoustic interval
or open a new window. The ledger's existing progress maps are evaluated on every
evidence push as well (`evaluate`, `noteLock`, `ayahFinished`). Thus a new status
field that the existing consumer ignores cannot recover the proposed missing
delivery. Making the consumer instead certify progress from that field would
introduce a new progress rule, beyond this card's requirement to restore an
existing notification while preserving checker behavior.

The mechanism is therefore wrong at the consumer boundary. This is not a finding
that the reciter never completed a source word, or that the target's internal
offset never reached a word end.

## Validation and limits

Two diagnostic Modal requests attempted to audit terminal match/deletion kinds,
suppressed updates and checker visits on the target's card-listener evidence.
Both supervisor results failed with `NotFoundError: No such file or directory`,
reported zero spend and returned no audit artifacts. A package-relative asset-path
error in the temporary diagnostic was corrected before the second request; the
supervisor errors did not identify that script or any other missing path.

Consequently, no target replay, endpoint census, prototype result or score is
claimed. Zero target wins means none validated, not a measured practice delta.
The source-level consumer finding is sufficient to reject the specified delivery
mechanism independently of those unavailable runtime observations.

No engine or unit-test behavior was changed, so unit suites were not run. The
temporary diagnostics were removed. `lab-experiment.json`, scorer, tooling,
manifests and evidence were left unchanged. Only this result and the killed
outcome are committed; there is no change to submit to the fold gate.

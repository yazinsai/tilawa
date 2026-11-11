# Transition event exclusivity

Status: needs-human — Modal validation infrastructure did not start the audit.

## Prediction

Tracker −1 and overall −1 on practice, with substantial zero-gain risk. The
target is `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
This predicted gain is below the fold gate; it is not a claimed release gain.

## Mechanism and required evidence

Give each accepted acoustic discontinuity an immutable identity containing its
last consumed source token, first resumed token, and accepted transition
ancestry. Carry the identity through the existing position-event interface.
Only when a confirmed word edit and a ledger skip have identical source and
resume tokens, and the ledger has no independent supporting transition, may
they become alternative descriptions of one event. Retain the word edit.
Distinct boundaries must preserve independently reportable adjacent mistakes.
Selected positions, windows, costs, margins, and lock-drop handling must stay
unchanged.

The premise kill criterion requires checking those exact boundaries and the
exclusive ancestry in the target. A prior word warning catching the planted
omission does not by itself establish that the whole-ayah warning duplicates
the same event.

## Inspection and attempted validation

Read `AGENTS.md` and the audio-review instructions. Inspected `tracker.ts`,
`local.ts`, `events.ts`, `checker.ts`, and `ayah-ledger.ts` without modifying the
scorer or evidence. The inspected implementation has two separate paths:

- The local tracker chooses a DP jump to a word start and publishes a position
  when the word or run origin changes. It currently exports no acoustic
  discontinuity identity.
- The checker creates word issues from its own alignment visits. A wrong-word
  issue owns an observation interval; an omission is placed between visits.
- The absence ledger detects a gap from a settled ayah to a later locked ayah.
  Its closed read-versus-skip likelihood uses the left neighbour's frame start
  through the right neighbour's frame end. Its issue location in the token
  window comes from the right lock's first frame. The ledger currently has no
  explicit transition-support set to compare with a word issue.

These are code facts, not proof that the target boundaries match or differ.
The experiment branches listed in the brief were not present as local git
refs, so their reported findings were treated as supplied history rather than
as new measurements.

A temporary audit added read-only predecessor records alongside the local DP
states and logged their source/resume frame identities with position events.
It also logged checker issue intervals, alignment visits, and accepted ledger
fit boundaries. No prices, selection comparisons, position payloads, or
warning decisions were changed. The planned target replay used the card's
listener, `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`, and the existing
practice chunking. A second replay was to compare the complete event stream
with the unmodified engine. These invariants have not been demonstrated by a
successful execution.

Supervisor Modal requests 1 and 2 both returned
`NotFoundError: No such file or directory`, with no shard logs or outputs and
`usd: 0.0`. Request 1 also had a relative asset-path issue in its probe; that
was fixed for request 2 before submission. Neither request supplied any
evidence about the target. Request 3 submitted only `pwd` and returned the
same error, no outputs, and zero compute spend. This establishes that the
available runner fails before any engine/probe command executes; the response
does not identify the missing path.

The temporary engine instrumentation was reverted. Retry artifacts are kept
under ignored `.autolab/transition-event-audit/` and
`.autolab/transition-event-audit.patch`; they are not an engine implementation.
The experiment card is unchanged.

## Limits and next validation

No target win count or score change was measured. No practice-score call was
made, no exam was accessed, and no engine behaviour change or new unit test is
left on this branch. Core/SDK tests were not run: the proposed behaviour was
not implemented, and the available Modal runner did not execute the audit.

Restore the supervisor runner, replay the audit with the card listener, and
verify that its position/checker events equal the unmodified engine's events.
Compare the target ledger's complete transition support with the confirmed
word issue's exact acoustic boundaries. Kill only if that premise fails, or
if a subsequently built change wins fewer than three targets. Otherwise add
the narrow consumer rule and unit tests covering both same-event alternatives
and adjacent distinct-boundary mistakes, then run the required core/SDK tests
and follow the prescribed practice-score gate. Infrastructure failure alone
does not establish the premise kill criterion.

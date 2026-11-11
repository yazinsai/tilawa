# Transition event exclusivity

Status: needs-human — implemented; supervisor validation failed before execution.

## Prediction

Tracker −1 and overall −1 on practice, with substantial zero-gain risk. Target:
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
This is a sub-gate prediction, not a measured gain.

## Mechanism

The earlier attempt left only a write-up after Modal startup failed. This
revision implements the proposed contract in the engine:

- `local.ts` copies an immutable jump predecessor alongside each DP cost,
  origin and edit count. The identity contains the last consumed source
  token, first resumed token, reference offsets and parent jump. Acoustic
  identity uses phone plus exact start/end frames. Save/restore retains the
  original token frontier and predecessor snapshots.
- `tracker.ts` exports selected local ancestry through
  `PositionEvent.transitionHistory`. Global binds start a new generation;
  exact provenance across those unaligned seams is unavailable. Metadata is
  excluded from event coalescing, so it cannot cause extra position events.
- `checker.ts` snapshots the settled and resumed anchors' ancestry for each
  pending absence-ledger gap. It computes every accepted jump between them.
  Missing ancestry, a generation change or a conflicting predecessor makes
  the support unestablished. Multiple accepted jumps remain independent
  support and cannot be merged.
- A word edit can own a discontinuity only when its original acoustic span's
  preceding and following tokens exactly match that jump's source and resume.
  The checker remembers only confirmed word descriptions. A ledger candidate
  becomes an alternative description only when its complete support is one
  identical jump, including ancestry, and the word edit lies inside the
  candidate's reference span. An already-open alternative resolves rather
  than confirming an additional broader warning.

There is no time tolerance, distance preference, threshold change, rebind or
coverage witness. Candidate costs and selection comparisons, window limits,
confirmation gates and lock-drop handling retain their existing rules. The
experimental consumer rule applies only to absence-ledger descriptions.

## Validation

Read `AGENTS.md` and the audio-review instructions. Static inspection confirms
that word issues and absence proposals previously lacked an explicit shared
identity; proximity and the planted omission's existing alignment hit alone
cannot establish duplication.

Added unit tests for confirmed word alternatives, uncertain/suspected issues,
exact source/resume boundaries, transported identity snapshots, differing
ancestry, multiple supporting transitions, missing support, containment,
immutable snapshots and DP rewind. The local-search test compares all cursor,
origin, edit and cost fields with an unannotated search on the same synthetic
stream. The existing ledger integration fixture now also exercises a real
skip with explicit ancestry and no confirmed alternative, plus clean reading.

Local Node TypeScript syntax checks and `git diff --check` passed. These are
syntax checks, not a TypeScript build or successful unit-test execution.

Supervisor requests 4 and 5 retried the diagnostic and baseline p46 replays
with the card's listener, `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`.
Both returned `NotFoundError: No such file or directory`, no shard outputs and
`usd: 0.0`, before any probe executed. The earlier requests 1–3, including
`pwd` alone, had the same failure.

Request 6 submitted the prescribed six-shard core/SDK build and unit tests.
Request 7 submitted the implemented engine's p46 replay with checker boundary
logging and an unmodified baseline comparison. Both returned the same
`NotFoundError`, zero compute spend and no shard outputs. The latter checks complete
position payload equality after removing the new metadata and records word
issue boundaries, accepted ancestry and the ledger fit. Diagnostic scripts
remain in ignored `.autolab/transition-event-audit/`; no debug logging is in
the committed engine.

## Limits and next decision

The target's exact boundary equality and exclusive transition support remain
unmeasured. A nonempty wrong-word span generally has different outer
boundaries from a jump between consecutive tokens; the strict contract must
abstain in that case. Global seams with unavailable provenance also abstain.
These are deliberate limits of the identity requirement, not evidence that
this target satisfies or fails it. Even exact shared boundaries can describe
a compound mistake; the practice gate must still assess that risk.

No practice-score invocation has been made and no target wins or overall
change are claimed. The supervisor must first execute the target replay and
tests. If the target's word issue and ledger warning do not share exactly one
accepted discontinuity, or the built change wins fewer than three targets,
revert the engine changes and record a premise kill. Infrastructure failure
alone does not establish that criterion. The card, scorer, lab tooling,
manifests and evidence are unchanged.

# Order-independent restart commitment

Status: killed on the implementation premise. No engine changes.

Test set: lab-v1.5u practice, phone. Listener from the unchanged card:
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`.

## Prediction

Tracker −1, overall 0: recover the false skipped-passage warning at
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`, potentially
exposing the previously documented nearby listener warning.

## Mechanism and implementation audit

The card proposes that commitment is evaluated against a changing incumbent
while candidates are enumerated. Its repair would score an immutable set against
one pre-update state and apply the existing margin against the strongest eligible
competitor, preserving eligibility, distinct locations, edit costs and movement.

The inspected implementation does not have that local commitment mechanism:

- `local.ts`, `createLocalSearch.step`, computes replacement cost, origin and edit
  arrays before selecting a position. All consuming transitions read the previous
  arrays. Reference-only deletion closure reads the new preceding cell, as required
  by the recurrence. Jump base, distance and surah crossing all use the unchanged
  pre-update `best` throughout that computation.
- The endpoint scan runs only after the complete row exists. It selects the lowest
  cost across every cell, breaking ties by distance to plain forward progress.
  Ascending reference traversal resolves a remaining exact equidistant tie toward
  the lower offset. Costs, origins, edits and `best` are replaced only afterward.
  There is no incumbent-relative margin test in this scan.
- `tracker.ts`, `localEvent`, receives that selected position. It delays a new run
  while weak or while recent step cost exceeds the existing confirmation cap; it
  does not enumerate alternatives or apply `commitMargin` to local restarts.
- Global `assess` already sorts complete scored matches, preserves its existing
  distinct-location and fit filters, and computes the winner's lead as the minimum
  score difference against every other fitting candidate. Its margin therefore
  already uses the strongest eligible competitor. `continuationDecision` in
  `slot-spans.ts` likewise scores its set first and checks other members setwise.

Adding a runner-up margin to local endpoint selection would introduce a new
commitment rule. It would not repair an order-dependent application of an existing
margin, which is the authorized experiment.

## Validation

The source audit establishes that the local row is already a set scored against
one pre-update state. Holding the existing tie semantics fixed, minimum selection
over that row is independent of enumeration. It does not progressively apply a
margin against an incumbent. The relevant global margin comparison also already
uses the strongest eligible competitor.

Two supervisor Modal requests were attempted:

- Run 1 prepared temporary instrumented copies inside the remote container to
  compare forward, reverse and deterministically shuffled selection over frozen
  completed candidate rows, preserving the lower-offset exact tie rule. It was
  intended to replay p46 with the card's listener and compare the complete
  tracker/checker event streams against the unmodified baseline.
- Run 2 requested `bash tools/autolab_tests.sh` with six shards for the build and
  core/SDK suites.

Both returned `NotFoundError: No such file or directory` with no logs or audit
results. The supervisor result files report zero compute spend. Consequently no
candidate counts, event equality, target-warning replay, or passing tests are
claimed. The saved 0.3915-edit target gap is context supplied by the card, not a
measurement reproduced here.

No practice score was requested (0 of 3 used). No engine or unit-test changes are
retained because no behavior was changed. The outcome records 0 target wins for
this killed, unchanged-code experiment; this is not a measured practice delta.

## Decision

Kill on the premise: there is no incumbent-relative local commitment margin to
repair, and the margin-based comparison already evaluates its complete eligible
set. Introducing a new local margin would change policy rather than fix the
proposed representation defect. The engine, card, scorer, tooling and test data
remain unchanged; only this write-up and the required outcome are committed.

## Limits

This is a test of the card's proposed enumeration defect, not a claim that the
selected rhyme restart is the correct location. A cost advantage can remain wrong
without depending on enumeration. The audit preserves the existing tie semantics;
changing the treatment of exact equidistant ties would be a different experiment.
Runtime enumeration invariance could not be independently replayed because the
Modal requests failed. This kill follows from the inspected recurrence and
selection contract, not from an asserted successful runtime audit or score.

# Reference-range projection contract

Status: killed on the source-level premise. No engine change retained. Target replay was unavailable; no practice score was run.

## Prediction

Tracker −1, overall −1 on lab-v1.5u with the card's listener, `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. The sole target was `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.

The proposed defect was a cross-ayah acoustically matched interval losing its source-ayah intersection when converted to per-ayah progress. The correction would give reference intervals an explicit start, exclusive end and transition kind, project only acoustic matches, and preserve deletion, jump, partial-word, scoring, timing and lock-drop behavior.

## Source audit

The proposed interval-to-ayah conversion is absent from the current tracker:

- `Match` carries `offset`, `length` and an alignment path. `firstWord` maps the start; `lastWord` maps `offset + max(0, length - 1)`. `toCandidate` exports both word endpoints, so a cross-ayah candidate retains its source word in `span.start`.
- `LocalPosition.offset` is explicitly an exclusive reference endpoint. `localEvent` maps `offset - 1` to the current word and maps `origin` separately to `span.start`. Neither operation projects an entire matched interval into the endpoint's ayah.
- The checker does not derive per-ayah acoustic progress from the enclosing candidate span. `track` obtains the current word cursor and uses the span start for window anchoring. `noteLock` records that cursor word under `ayahOf[cursor]`; it is a point observation, not an interval converted through one endpoint.
- Acoustic read progress is collected in `judge` from individual word visits. Every qualifying visit calls `growSpan(readAyah, ayahOf[v.word], v.word, ...)`. Its word and ayah come from the same reference word. Source-side visits therefore have their own source-ayah entry, regardless of where a later visit ends.
- Those visits retain the existing completeness requirements: positive consumption, `MIN_COVERAGE`, consonant support where required, sufficient consumed phones, and a visit that is not judged a wrong word. A deletion-only visit or insufficient fragment cannot establish read progress. `growSpan` extends an entry rather than overwriting its earlier extent.

The ledger's relevant distinction is completion. `ayahFinished` requires either the locked or read entry's `wordTo` to reach the ayah's last word. `closeGap` can consequently propose a skip for an unfinished ayah even though its prefix already has per-ayah progress. That is different from losing a source-side intersection during conversion.

The existing excursion-rebind write-up already reports p46 walking from 54:33 into 54:36 within one window, without an excursion. The supplied span-from-visits result also reports visits to 54:36:1–3 inside 110.96–121.84 s while the ledger still favored the skip hypothesis. These saved findings are contextual evidence, not a fresh replay with this card's listener. The current reference has five words in 54:36; retained prefix progress alone does not meet the ledger's completion condition.

Introducing matched-path provenance and making partial progress prevent a whole-ayah ledger proposal would change the warning contract. This audit found no existing endpoint-based range projection to repair while preserving the card's other behavior, so no such change was made.

## Validation and limits

Two diagnostic runs were requested through the authorized supervisor Modal runner. The diagnostic was designed to retain selected local transition histories, inspect read and locked progress, and compare instrumented events byte-for-byte against a shipped replay. Temporary instrumented source copies were to run only inside Modal, with no scoring changes.

The first request failed before command execution with `NotFoundError: No such file or directory`. The second request returned no output or result within its requested 10-minute duration; its execution status was unavailable when this probe was closed. Neither request supplied a usable target trace. The temporary local diagnostic was removed. No unit tests or practice scores were run, and no target recovery is claimed. `targets won: 0` in the outcome records zero demonstrated recoveries, not a measured full-practice delta.

The kill rests on the inspected source contract, not an acoustic judgment or a newly reproduced target trace. It rules out the stated endpoint-based conversion mechanism in this implementation; it does not establish that the existing ledger's treatment of partial progress is correct. All engine code, tests, lab tooling, scoring data and `lab-experiment.json` remain unchanged.

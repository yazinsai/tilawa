# Marginalize the origin of the local comparison window

Status: killed on the premise. No engine change retained, no demonstrated target wins, no practice score requested.

## Prediction

Tracker −1, overall 0: recover the reported false skipped-passage warning at `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`, allowing one nearby listener false warning to become exposed. The card specifies listener `exp-listener-teacher-insert-marginal-gate-ad6eb8c0` and practice set `lab-v1.5u`.

## Proposed mechanism

Treat the beginning of the existing local acoustic comparison window as uncertain. Evaluate continuation and restart over identical nested intervals within retained history, apply identical normalized start probabilities estimated from transcript-consistent boundaries in help-train training speakers, and marginalize each complete explanation separately. Preserve movement edges, edit prices, commitment margins and checker handling; exclude release-held-out speakers from estimation.

The premise is that one arbitrary acoustic-history cutoff truncates distinguishing source context and supplies the reported 0.3915-edit restart advantage.

## Source audit

The inspected implementation does not use such a cutoff for local candidate ranking.

- `tracker.ts` creates `createLocalSearch` when it establishes a lock. During ordinary following it advances the same search with `local.step(ph.p)` for every observed phone. Trimming `observed` does not recreate or reset that search.
- `local.ts` carries a DP row of accumulated costs, run origins and edit counts. Insertions and substitutions extend the previous row; a jump starts from that row's best accumulated cost plus the existing movement cost. Reference-only deletions extend the current row. Both continuation and restart therefore inherit the scored prefix.
- After selecting a row minimum, the implementation subtracts that same minimum from every candidate. This preserves their cost differences; it does not erase earlier relative evidence.
- `RECENT = 16` truncates only the list of recent cost increments. Candidate selection never reads that list. `position()` uses its average to report recent fit, and tracker handling uses recent fit to detect weakness and decide whether to publish a new run. It is not the interval over which the candidate row is scored.
- The tracker's six-phone lookback and 48-phone weak-search window bound global reacquisition evidence. The 64-phone `heard` buffer and saved states serve phrase detection and rewind. They do not define an acoustic start for the local DP. Its two-ayah bounds are reference-position bounds, with retained cells carrying their prior accumulated scores.

Thus restoring any retained checkpoint with its complete DP prefix scores would retain the preceding context, rather than introduce a different acoustic cutoff. Erasing those scores and rescoring an independently initialized suffix would introduce a new context-discarding mechanism; it would not repair the hypothesized cutoff in the existing comparison. No boundary prior was estimated because the premise was absent.

## Validation attempted

Temporary read-only instrumentation was prepared to record local states and buffer trims during a single-case product replay. The diagnostic would replay nested suffixes from 64 retained checkpoints, compare their final candidate rows, and repeat with the diagnostic `recent` list cleared. This was a proposed verification, not a completed measurement.

Supervisor Modal request 1 attempted that replay with the card's listener. Request 2 attempted the supplied six-shard core/SDK build and test command, `bash tools/autolab_tests.sh`. Both supervisor results reported exactly `NotFoundError: No such file or directory`. Neither provided command output, a replay result or test results; the failure location is unknown.

All temporary instrumentation and its driver were removed. The engine files match the starting branch byte for byte, and `lab-experiment.json` was not edited. No new unit test is retained because no behavior change is retained. No practice-score allowance was used.

## Decision and limits

Kill because the proposed fixed acoustic-window cutoff is absent from local ranking, independently of the Modal failures. This is the user's “mechanism is wrong” stop condition, not an empirical claim that the restart beats continuation at every independently initialized window origin.

The reported target warning and 0.3915-edit advantage were not independently reproduced here. There is no measured practice delta, no completed boundary-family experiment, and no claim that a newly introduced suffix model could never change the ranking. The result is narrower: the existing local comparison already carries accumulated prefix evidence, so the proposed explanation of lost context at one recent-history cutoff does not match this implementation. Engine changes were reverted; the supervisor receives a killed outcome with zero demonstrated target wins.

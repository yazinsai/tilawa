# Session motif tied decoding

Status: blocked by supervisor execution; no acoustic result. The experiment
card is unchanged. The unvalidated prototype was removed from engine code.

## Prediction and mechanism

Predict listener −1 and overall −1: jointly assigning one latent multi-phone
sequence to acoustically matched recurring motifs may remove an inconsistent
wrong-word warning without changing the primary decode.

Select nonoverlapping recurring subword intervals from frozen acoustic features,
without using Quran text or requiring greedy phone-label agreement. Keep the
graph bounded and use only occurrences completed before the disputed interval.
Freeze matching and abstention on speaker-separated partitions of help-train
training speakers' none-labelled recordings, excluding release-held-out
speakers. The experiment is limited to one day of cached-feature work.

At an existing supported string-path wrong-word accusation, compare the complete
written and already-accused pronunciations. Matched occurrences share a latent
multi-phone sequence, but each occurrence retains its own CTC likelihood and
timing paths. Marginalize the motif's placement in each complete pronunciation.
Issue a veto only if tied scoring prefers the written pronunciation while the
otherwise identical untied arm still prefers the rival. Primary phonemes,
posteriors, tracker events, graph confirmations and insertion decisions must
remain unchanged.

The card's prerequisite is at least two disputed motifs with independently
matched earlier occurrences that reverse preference. The task's additional
continuation rule requires at least three target warnings won before proceeding
to practice scoring. Neither prerequisite was measured here.

## Work prepared

A provisional SDK module constructed energy-normalized log-mel motif features,
bounded windows of multiple acoustically emitted phones, causal nearest-neighbour
edges verified by dynamic time warping, and nonoverlapping earlier matches.
It chose one edge by acoustic distance before comparing pronunciations, avoiding
reference-driven selection among whichever matches happened to produce a veto.

The prototype sequence scorer summed separate complete CTC path likelihoods for
the shared sequence in every occurrence, rather than averaging aligned frames
or voting on token identities. It marginalized the shared sequence's placement
inside the complete query pronunciation and supplied a matched untied comparison.
A provisional checker callback exposed only already-supported string-path
confirmations and retained the baseline internal ownership ledger.

A generic SDK diagnostic CLI was prepared to cache none-labelled training takes
and inspect engine-generated comparison windows. Unit tests were written for
exhaustive small CTC path sums, repeated identities, different occurrence
durations, complete-pronunciation coverage, causal/nonoverlap restrictions,
acoustic selection independent of phone spelling, and the callback's scope.
These are prepared code and test cases, not validated behavior. Calibration,
certificate transport, normal SDK integration and a deployed bundle were not
completed.

## Validation and execution blocker

All four requests through the supplied supervisor route failed with the same
result, before returning command output or a shard artifact:

`NotFoundError: No such file or directory`

| Supervisor run | Requested work | Result |
| --- | --- | --- |
| 1 | Inventory `/lab/data`, available models and help-train labels | Execution error; no inventory |
| 2 | Minimal `pwd && ls /lab/data` retry | Same execution error |
| 3 | Supplied six-shard `bash tools/autolab_tests.sh` command | Same execution error; tests did not run |
| 4 | Explicit training image, list `LAB_DATA` with Python | Same execution error |

The result files are `.autolab/modal-result-1.json` through
`.autolab/modal-result-4.json`. Each contains the error above and no command
exit code or shard results. No logs, feature caches or audio observations came
back. The failure therefore cannot be attributed to a label schema, a missing
particular recording, a model's pronunciation preference, or the implementation.
The supervisor's Modal execution path needs repair before those can be checked.

No training-speaker calibration was performed. No target was inspected or
measured, no test suite completed, and no practice score was requested (all three
practice-score allowances remain). Targets won and listener/overall changes are
unknown, not zero. No conclusion about the acoustic premise or fold gate follows.

Local `git diff --check` passed before removing the prototype. Its complete patch
is retained in ignored `.autolab/session-motifs-prototype.patch`; after rollback,
`git apply --check` verified that it can be reapplied to this branch. It is a
resumption aid, not an evaluated or shipped implementation.

No engine changes, model declaration, auxiliary asset bundle, scorer/tooling
changes, supplied manifest edits, test-data changes or evidence edits remain.
No killed outcome is declared: the task permits killing on the acoustic premise
or measured target gains, and neither was observable. Resume by restoring the
supervisor route, calibrating on the authorized training speakers, and testing
the prerequisite before implementing certificate transport or requesting a
practice score.

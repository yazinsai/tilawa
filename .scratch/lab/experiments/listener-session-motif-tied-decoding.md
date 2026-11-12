# Session motif tied decoding

Status: not implemented; blocked by supervisor execution, with no acoustic
result. The experiment card is unchanged. The unvalidated prototype was removed
from engine code.

The previous review requested engine code or an explanation of why the change
cannot be completed. The prototype was restored and revised, and three fresh
requests through the supplied supervisor route were attempted. All three failed
with the same execution exception as the original four. This prevents obtaining
the required training-speaker feature cache, freezing acoustic matching rules,
checking the target preference reversals, and running the required tests. The
mechanism is implementable, but these required experimental steps cannot be
completed through the currently available route. An uncalibrated, unvalidated
prototype is not a completed implementation of this card.

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
the shared sequence in every earlier occurrence, rather than averaging aligned
frames or voting on token identities. It marginalized the shared sequence's
placement inside the complete query pronunciation and supplied a matched untied
comparison. The revised query calculation keeps one CTC chain across the motif
cuts, enforcing the blank needed between repeated identities on opposite sides
of a cut. Its placement prior is normalized before observing likelihoods.
A provisional checker callback exposed only already-supported string-path
confirmations and retained the baseline internal ownership ledger.

A generic SDK diagnostic CLI was prepared to cache none-labelled training takes
and inspect engine-generated comparison windows. The revised training-only
calibration utility partitions authorized training speakers deterministically,
fits an acoustic distance boundary on one partition, and requires it to pass a
disjoint training-speaker partition. Transcript alignments evaluate the matching
rule during calibration; they do not select the acoustic graph's edges. Only
train-split, none-labelled takes without exclusions, events, repeats or unsupported
labels are admitted. No recording was actually loaded and no rule was frozen.

Decoded identities were removed entirely from the window-selection predicate;
the prepared invariance check also replaces every emitted identity with the same
symbol. Unit tests were written for
exhaustive small CTC path sums, repeated identities, different occurrence
durations, complete-pronunciation coverage, causal/nonoverlap restrictions,
acoustic selection independent of phone spelling, and the callback's scope.
These are prepared code and test cases, not validated behavior. Calibration,
certificate transport, normal SDK integration and a deployed bundle were not
completed.

## Validation and execution blocker

All seven requests through the supplied supervisor route failed with the same
result, before returning command output or a shard artifact:

`NotFoundError: No such file or directory`

| Supervisor run | Requested work | Result |
| --- | --- | --- |
| 1 | Inventory `/lab/data`, available models and help-train labels | Execution error; no inventory |
| 2 | Minimal `pwd && ls /lab/data` retry | Same execution error |
| 3 | Supplied six-shard `bash tools/autolab_tests.sh` command | Same execution error; tests did not run |
| 4 | Explicit training image, list `LAB_DATA` with Python | Same execution error |
| 5 | Fresh inventory of practice data, models and help-train labels | Same execution error; no inventory |
| 6 | Supplied six-shard `bash tools/autolab_tests.sh` command after restoring the prototype | Same execution error; tests did not run |
| 7 | Core/SDK builds, focused unit tests, none-labelled feature caching, speaker-separated rule freezing | Same execution error; no command output or cache |

The result files are `.autolab/modal-result-1.json` through
`.autolab/modal-result-7.json`. Each contains the error above and no command
exit code or shard results. No logs, feature caches or audio observations came
back. The failure therefore cannot be attributed to a label schema, a missing
particular recording, a model's pronunciation preference, or the implementation.
The supervisor's Modal execution path needs repair before those can be checked.

No training-speaker calibration was performed. No target was inspected or
measured, no test suite completed, and no practice score was requested (all three
practice-score allowances remain). Targets won and listener/overall changes are
unknown, not zero. No conclusion about the acoustic premise or fold gate follows.

Local `git diff --check` passed before removing the prototype. The original patch
is retained in ignored `.autolab/session-motifs-prototype.patch`; the revised
prototype, calibration utility and tests are retained in ignored
`.autolab/session-motifs-prototype-v2.patch`. After rollback, `git apply --check`
verified that the revised patch can be reapplied to this branch. These are
resumption aids, not evaluated or shipped implementations. Run 7's exact
build/test/cache/calibration command is retained in `.autolab/modal-request-7.json`.

No engine changes, model declaration, auxiliary asset bundle, scorer/tooling
changes, supplied manifest edits, test-data changes or evidence edits remain.
No killed outcome is declared and no target count is fabricated: the task permits killing on the acoustic premise
or measured target gains, and neither was observable. Resume by restoring the
supervisor route, calibrating on the authorized training speakers, and testing
the prerequisite before implementing certificate transport or requesting a
practice score.

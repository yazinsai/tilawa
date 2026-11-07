# Verify before seam span suppression

Status: implemented; target premise audit and test execution blocked by the
supervisor queue. The experiment card is unchanged.

## Prediction and mechanism

The card predicts one recovered addition with no losses: listener -1 and overall
-1 on lab-v1.5u. The proposed defect is arbitration before acoustic verification:
a retained seam spelling fails the existing teacher insertion verdict, while an
overlapping lower-ranked spelling that would pass never receives that verdict.
The proposed change would retain existing distinct word-and-interval candidates,
verify them independently, and suppress overlaps among passing candidates using
the existing ranking. Candidate generation, acoustic evidence, pronunciation
graphs, confirmation constants, primary decoding and tracker input would stay
unchanged.

The premise criterion requires at least two registered targets to contain such
a discarded, independently passing candidate. The task also requires rollback
when the built change wins fewer than three targets. No aggregate practice score
is needed if the premise is absent.

## Code inspection

Baseline implementation commit: `14a5a11f`.
`packages/sdk/src/seam-insertion.ts` discovers candidates by replaying the free
tracker and checker on native frame cells. Its `seamInsertion` callback records
distinct complete candidate objects and returns `true`. The callback therefore
accepts a candidate provisionally before the asynchronous teacher pass runs.

The relevant arbitration is in `packages/core/src/checker.ts`, rather than a
separate SDK overlap function. Seam utterances are ranked longest first. An
accepted candidate sets `found`, stops the remaining span/visit search, and can
occupy a lexical issue slot that later searches skip. A declined callback permits
the search to continue. The SDK subsequently evaluates the recorded candidates
with the existing `seamInsertionMarginal` and transports their marginals as
metadata; the consuming checker requires a marginal strictly above 0.5.

Within one scan, `seamUtterances` partitions completed pause-bounded utterances
into disjoint token intervals; it does not enumerate overlapping subspans.
`lexicalWords(index).surplusWord(w1, said)` returns a single spelling for the
utterance. The SDK's `seen` key is the complete candidate object, and its
verification loop contains no overlap comparison. Thus first-accepted-candidate
arbitration is present, but a ranked list of competing overlapping spellings is
not present within one scan. Different visits or later scans may still give an
utterance different whole-seam scoring bounds. The target census retains full
candidate objects and tests whether leaving all proposals unaccepted exposes any
relevant alternative; the local partition alone is not a target kill result.

## Validation and limits

The prepared SDK change returns `false` from discovery after retaining each
distinct proposal. Every retained proposal still receives the unchanged teacher
verdict. On replay, the consuming checker applies its original ranking and
first-accepted-candidate behavior only after the certificate's marginal passes
0.5. No primary phones, posterior rows, tracker input, lexical generator,
pronunciation graph or confirmation constants are changed.

The SDK regression uses real forward-backward scoring on synthetic teacher rows:
the first proposed spelling is unsupported, while a later spelling over the same
interval is supported. It requires both verdicts, deduplicates repeated proposals,
shares crop inference, and checks that primary evidence remains unchanged. The
mock discovery stops on an accepted callback, so returning `true` reproduces the
lost-alternative failure.

The diagnostic compares the original accepting discovery callback with a
callback that declines every proposal. Both arms use the card's declared
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0` phone evidence, unchanged
candidate generation and costs, under both free tracking and true-position
events. Full candidate objects are retained, including different whole-seam
bounds. Labels only select and report the four registered incidents. This is a
candidate-delivery audit, not a practice score or an independent audio opinion.

Supervisor requests and their available results:

| Run | Request | Result |
| --- | --- | --- |
| 1 | Inspect available model/evidence paths | Copied log and status show exit 0; the declared p07 listener evidence is available. |
| 2 | Initial four-shard premise census | Copied logs show exit 1 because the ignored `.autolab` probe was omitted from the source snapshot. This is a diagnostic setup failure. |
| 3 | Corrected four-shard premise census; archived source also embedded in the pending request | Queued; no output or result file available. |
| 4 | Supplied six-shard `bash tools/autolab_tests.sh` build and test harness | Queued; no output or result file available. |

After copying runs 1 and 2's outputs, the supervisor did not publish their
`modal-result` files or start runs 3 and 4 for over twenty minutes. The executor
has no Modal credentials, so the pending runtime checks could not be completed
through the authorized route. The temporary diagnostic is archived in ignored
`.autolab/suppression-premise.ts` and is excluded from the committed change.
Completed logs remain under `.autolab/modal-out-{1,2}/`.

There are no measured target recovery counts or independently evaluated target
teacher marginals. The premise criterion has neither passed nor fired; this
external execution blockage is not a scientific kill result. The implementation
is left in place for verification, with no killed outcome asserted. No practice
score was requested (all three permitted scores remain). No new models or
side-ear assets are introduced: the existing insertion teacher and registry are
used unchanged. Aggregate gains, clean-session behavior, studio regressions and
the kill criterion remain unvalidated.

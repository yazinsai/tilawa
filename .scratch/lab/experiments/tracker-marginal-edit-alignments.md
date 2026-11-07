# Normalized marginal token edit alignments

Status: blocked on missing Modal supervisor results. No experiment verdict.
Prototype: `751f6abc`. The decoder and comparison/replay helpers are retained
so the pending validation can resume. The experiment card is unchanged.

## Prediction

The unchanged card predicts tracker −1 and overall 0 on lab-v1.5u practice
phone, with listener `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`.
Target: `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
Correcting the restart can expose the previously documented listener wrong-word
warning nearby; that possibility is explicitly counted in the forecast.

The premise is that the rhyme restart wins because one edit alignment is cheap,
while the nearby continuation has stronger probability summed over its alternative
token alignments. The frozen normalized marginal must reverse that ordering over
the identical complete heard-token interval. Otherwise the card's premise kill
criterion fires.

## Mechanism and premise probe

The minimum-path local DP lives in `packages/core/src/local.ts`, called by
`tracker.ts`. `edit-alignment.ts` prototypes the acoustic edit marginal for that
comparison. At each reference cursor, a geometric insertion loop competes with
reference advancement. On advancing, a reference-conditioned deletion competes
with an emission drawn from a normalized phone-confusion row. Every outgoing edit
choice sums to one. The terminal cursor also has an insertion loop and explicit
termination. Extra alignment routes divide existing probability mass.

A log-space forward algorithm sums insertion, deletion and emission paths. A
maximum-path diagnostic uses exactly the same frozen model and routes; comparing
it with the forward sum separates acoustic-price changes from the effect of
alignment ambiguity. Likelihood uses the existing three nats per plain edit.

The optional `TrackerOptions.editAlignmentModel` enables the local DP prototype.
Every reference endpoint has its own summed acoustic score. Movement alternatives
retain the existing minimum comparison and existing jump/distance/surah-entry
costs; jump alternatives are not summed as extra votes. The selected acoustic
route carries a plain-edit witness for the unchanged confidence, confirmation
and misfit rules. Initial reference deletions close the channel before its first
heard token. Save/restore and recentring preserve both score and witness state.
The ordinary model-free path remains available as the replay control.

Before replacing the tracker comparison, a temporary instrumented source copy
reconstructs the existing near and restart reference traversals and their movement
costs. Both traversals must carry the entire identical heard interval, and their
reconstructed ordinary costs must agree with the real local DP. The probe scores
all token edit alignments of each traversal, keeping the destination locations
and existing jump, distance and surah-entry prices separate. A second marginal
conditions on retained jump token boundaries, checking sensitivity to alternative
token assignments across those boundaries. These are premise diagnostics, not
a claim to sum every possible movement history in the Quran.

No commit margin, event delivery, checker rule or scorer changes for this probe.
Every instrumented tracker push and end event is checked against the unchanged
tracker, and the ordinary checker receives its original evidence and events.
The product replay separately enables the frozen model and feeds its events to
an unchanged checker using the original lab chunks and `trackerInput` handoff.

## Training

Modal run 1 inspected the supplied help-train volume: 489 label rows, all
`split == train`. The estimator refuses any other split. It selects audited,
usable phone recordings with clean, skipped-word, skipped-ayah or repeat scripts.
Actual scripted omissions are removed and repeats expanded before aligning
decodes to speech. Substitution, vowel, tajweed and unsupported scripts are
excluded because their altered phones cannot be reconstructed safely from these
labels. The final word uses the reference's isolated form to avoid supervising
ordinary waqf as an acoustic error.

The unchanged SDK streaming listener decodes 20 ms pushes with
`models/spliced-clean-int8`. A deterministic full phoneme alignment estimates
insertion, deletion and phone-confusion counts. A fixed Jeffreys pseudo-count of
0.5 smooths the normalized distributions. The fit is frozen and hashed before
practice evidence is loaded. No practice session, reader, verse or held-out
speaker enters the estimator.

## Validation and limits

Seven focused tests cover exact enumeration of the one-token edit routes, total
output probability including insertions and empty speech, a minimum-to-marginal
ordering reversal on synthetic ambiguity, deterministic fitting and exact speech,
stable long-stream likelihood and prefix accounting, and scripted-edit/held-out
exclusion, and exact local progress, a genuine skip and save/restore with the
model enabled.

Training labels supply word order and scripts, not independent phoneme truth.
Unlabelled pronunciation, unrepresented prefaces and reference-form differences
can still enter alignment-derived acoustic counts. Training learners' real phone
recordings also differ from artificially degraded practice studio audio.

No full practice score has been requested (0 of the allowed 3 used). No overall,
component, studio, clean-session or leave-one-reader delta is claimed.

## Pending supervisor work

At 2026-11-07 15:35 UTC, the run-1 inspection log and its shard status were
downloaded, but even run 1's supervisor completion record had not appeared.
Runs 2–4 had neither completion records nor downloaded outputs. Consequently
there is no usable fitted model, test result, target ordering or recovery count
to report. This infrastructure delay does not establish a premise kill.

The following requests are saved under `.autolab/modal-request-<n>.json`:

| Run | Work | Expected output |
| --- | --- | --- |
| 1 | Inspect training schema and model bundle | `.autolab/modal-out-1/0/log.txt` (downloaded, shard code 0) |
| 2 | Decode eligible training phone recordings, six shards | Own experiment's `/scratch/tracker-marginal-edit-alignments/decodes/` |
| 3 | Full core and SDK builds/tests, six shards | `.autolab/modal-out-3/<shard>/log.txt` |
| 4 | Focused tests, frozen fit, all-endpoint ordering and ordinary product replay | `.autolab/modal-out-4/0/{fit,ordering,baseline,prototype,target}.json` |
| 5 | Fresh full core and SDK builds/tests for the committed prototype | `.autolab/modal-out-5/<shard>/log.txt` |

Run 4 waits for every expected decode before fitting. It hashes the model before
reading practice evidence, checks every reconstructed path's complete heard
interval and ordinary cost, compares every baseline tracker push and end, and
replays the fitted model through the ordinary checker. Run 5 was queued after
the tracker integration and final unit-test assertions to validate that version.

The supervisor enforces its two-run concurrency limit. No direct Modal command,
alternative credential route or full practice score was used.

The prototype is optional and requires the fitted model: it is **not enabled by
default** and is not ready for a practice-score or fold decision. Next, collect
the pending results with the authorized `modal-result` command. If the frozen
ordering still prefers the rhyme restart, or the built prototype wins fewer
than three targets, remove the engine prototype and temporary helpers, commit
the measured write-up and a killed `.autolab/outcome.json`. Otherwise install
the training-frozen model in engine code, validate it, and request practice-score
before deciding. No killed outcome file is written without those measurements.

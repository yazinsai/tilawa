# Simultaneous close-mic and phone reference pairs

Status: killed before training; the required independently transcribed simultaneous
recordings are unavailable to this execution. The experiment card is unchanged.

## Prediction and mechanism

The card predicts two recovered missed additions with one collateral wrong call:
listener -1 and overall -1 on lab-v1.5u. The proposed supervision is a newly
commissioned corpus of evaluation-excluded learners practising freely, recorded
simultaneously on a close microphone and their own phone. Independent annotation
of the close-mic speech must establish what was actually performed, including
natural deviations, repetitions, incomplete attempts and pauses. Scripts,
teacher guesses and canonical verse text cannot supply that truth.

A monotone recording-clock map would associate the two captures of each spoken
phone occurrence. Phone-input training would use performed-transcript CTC and
close-view occurrence distributions with emission timing marginalized within
corresponding occurrences. Teacher targets would remain only where they agree
with independent transcription. The original authorized training mix and an
ordinary-reading output anchor would limit drift. An identical-data phone-only
control is essential to distinguish paired supervision from merely adding data.
The resulting primary listener would retain the SDK vocabulary and output clock,
with tracker and checker rules unchanged.

## Premise audit

Starting commit: `ee018aad40b1624ee2830fd6d8ebbd6feb452f04`.

Read `AGENTS.md` and `docs/agents/audio-review.md` first. Repository searches found
no supplied simultaneous recording corpus, close-mic performed transcripts,
capture-pair provenance or monotone recording-clock maps. The locally present
audio is a small SDK fixture, not the newly commissioned learner corpus.

The existing sources establish why the available labels cannot substitute:

- `modal/student_prep.py` packages a single FLAC waveform with canonical `ref`,
  teacher `hyp` and teacher top-five frame posteriors. `tools/student_manifest.py`
  constructs `ref` from the tlog label's canonical reference. These are neither
  an independent close-mic transcript nor two recordings of the same performance.
- `modal/student_train.py` loads that single waveform and its reference/teacher
  labels. Its distillation joins teacher frame indices to the student's output;
  it has no simultaneous-capture occurrence or recording-clock supervision.
- `.scratch/v1-roadmap/issues/12-help-clips-import.md` documents 489 training
  recordings from 105 training speakers, with phone/laptop device categories,
  scripted-mistake events, repeats as non-errors and audit verdicts. Those are
  documented source counts, not counts independently verified in this run.
  Device categories and readings of the same ayah do not establish simultaneous
  captures of one performance. Audit approval of a planted label does not provide
  a full performed transcript or close-mic phone-occurrence annotation.

No callable recording-commissioning or independent human transcription service
is available in this session. The supplied supervisor runs code; it cannot create
the new human performances or their independent annotations. Synthetic phone
degradation, matching separate takes, forced alignment of scripts or teacher
pseudolabels would change the experiment's supervision mechanism.

## Remote validation and limits

Two read-only inventory requests used the supplied supervisor command. Run 1
attempted to list help-train audio counts and metadata field names under
`LAB_DATA/help-train`. Run 2 used `--image train` without a GPU to inspect Parquet
schema names and counts in the authorized `/student/s1`, `/student/aug` and
`/student/eval` mounts, plus top-level supplied-data entries and this experiment's
scratch entries. Both reserved five-minute timeouts.

Both requests exited with status 1 and
`NotFoundError: No such file or directory`, returning no inventory output.
Requests and failure responses remain ignored under `.autolab/modal-request-1.json`,
`.autolab/modal-request-2.json`, `.autolab/modal-result-1.json` and
`.autolab/modal-result-2.json`. No mounted-data count or schema inspection is
claimed. The remote failure is an execution limitation, separate from the
missing commissioned corpus in the supplied inputs and local workspace.

Verified eligible simultaneous pairs: 0. Targets acoustically evaluated: 0.
Targets won: 0, meaning no target recovery was demonstrated; this is not a
measured model comparison. Neither the card's two-stream consonant-restoration
criterion nor the task's three-target continuation requirement can be tested
without the paired corpus and its identical-data phone-only control. This result
does not disprove the acoustic hypothesis.

## Disposition

Stop on the missing-supervision premise. No engine edits were made, so no engine
rollback is needed and no new unit behavior exists to test. Core/SDK suites were
not run for this write-up-only disposition. No training, model export, model
selection or practice score was requested; all three practice-score attempts
remain unused and no GPU budget was spent.

Leave the primary listener, tracker, checker, experiment card, supplied manifests,
test data, evidence, scorer and lab tooling unchanged. Only this result write-up
is committed; `.autolab/outcome.json` records the killed outcome and remains
untracked. A future execution needs actual simultaneous captures with independent
performed transcripts, exclusion provenance and recording-clock correspondence
before the training mechanism can be implemented and evaluated.

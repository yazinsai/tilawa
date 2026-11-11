# Paired-waveform teacher targets

Status: blocked by the authorized execution service, not killed on the premise.
No engine or model changes remain in the active diff. The experiment card is
unchanged. Draft implementation is retained in commit `91b1247b` for resumption;
it is not a completed or evaluated model intervention.

## Prediction and mechanism

The card predicts listener -1 and overall -1, recovering a faint planted
consonant by eliminating contradictory teacher supervision. Materialize each
augmented waveform once, use that exact observation and valid extent for both
teacher and student, and identify cached targets by the transformed waveform
and teacher/processor identity. Correct only demonstrated waveform-transform
mismatches. SpecAugment remains student-only feature masking.

The proposed measurement is a bounded CPU phone-head refit on frozen encoder
features against an identical-data current-target control. Authorized Quran
recordings and help-train training speakers' none-labelled takes supply the
examples; release-held-out speakers are excluded. Both arms must share features,
transcripts, initialization and update order. Checkpoint selection must use
speaker-disjoint partitions within the admitted training speakers, before any
target replay. Vocabulary, frame timing, tracker, checker and verification
rules stay unchanged.

## Source finding

The current source does contain the hypothesized observation mismatch:

- `modal/tlog_pass.py::TeacherPass` decodes original mono/resampled waveforms.
  Teacher inference applies no RIR, noise or random gain. Top-five targets are
  computed with the bfloat16 teacher and persisted as float16 probabilities.
- `modal/student_prep.py` joins that teacher row to the recording and packs the
  FLAC waveform with its teacher targets. Recording identity is preserved.
- `modal/student_train.py::ClipDataset.__getitem__` reads that waveform, applies
  stochastic RIR, noise and gain, and computes student fbank from the transformed
  waveform. It still returns the original cached teacher-target slice.
- `losses` aligns those original teacher rows with the student output using the
  existing delay/clamp rule and valid student lengths. It does not rerun the
  teacher on the augmented observation.

Thus waveform transformations can change what the student hears while the
targets continue describing the unaugmented recording. This is distinct from
the item-provenance audit, which established recording identity, and from the
padding audit, which found zero direct supervision outside valid student frames.
This source finding does not measure the size of the acoustic discrepancy on
historical cache contents, or prove that it caused any of the target misses.

## Draft and review

Commit `91b1247b` contains a draft change to the training path and new
`modal/student_paired_kd.py` / `modal/test_student_paired_kd.py` files. It can be
restored with `git cherry-pick 91b1247b` after the service is available.

The draft owns immutable float32 waveform bytes and valid sample extent,
passes the same observation to student fbank and the teacher processor, and
uses a bounded cache keyed by waveform, extent, sample rate, teacher parameters
and processor configuration. Existing targets are retained when the actual
waveform has not changed. Teacher and target-storage precision match the
original teacher pass; correcting augmentation must not also introduce a
precision change. Original targets remain available for the current-target
control. Student validity masking and the existing frame alignment are retained.

The helper also captures frozen features at the existing phone head, rejects
scripted help-train mistake labels, and defines paired/current bounded CPU
refits with a common update order and training-only checkpoint selector. It
rejects overlapping fit/selection speakers and release-held-out speakers.

These helpers are not a complete experiment driver. Actual training-data
admission, historical target inspection, encoder-feature preparation, teacher
inference, head refitting and standard SDK bundle export remain unfinished.
The real Muaalem adapter, tensor integration and exported-model parity are
unvalidated. No claim of a ready-to-use model is made.

## Validation and limits

The lightweight command
`arch -x86_64 python3 -m unittest discover -s modal -p test_student_paired_kd.py -v`
passed five tests: distinct transformed waveforms get distinct targets, sample
extent/rate/teacher identity enter the key, source mutations cannot alter stored
observations or targets, eviction recomputes the correct observation, and
scripted help-train takes are rejected. These small local checks do not train
or run an acoustic model. Python compilation and `git diff --check` also passed.

The tensor integration class was skipped locally because torch and soundfile
were unavailable. Its intended checks use the actual extracted data/loss
functions to test waveform sharing, unchanged-waveform preservation, valid
padding gradients, control-target preservation and equal-target refit identity.
Those checks are prepared, not reported as passing.

All four authorized supervisor requests returned
`NotFoundError: No such file or directory` before command execution:

| Request | Purpose | Result |
|---|---|---|
| 1 | Standard execution environment probe | Failed before execution |
| 2 | Training-image dependency/data probe | Failed before execution |
| 3 | Six shards of `bash tools/autolab_tests.sh` | Failed before execution |
| 4 | Tensor integration tests on the training image | Failed before execution |

Requests and error results remain in ignored `.autolab/modal-request-*.json`
and `.autolab/modal-result-*.json`. Core/SDK suites did not run; their pass
status is unknown. No alternative Modal access, training on the Mac, scorer
changes, test-data changes or benchmark-specific engine rules were used.

Targets evaluated: 0. Target recovery, listener change and overall change are
unknown. Practice-score calls: 0. No refit, checkpoint selection, target replay,
studio validation or model export ran. No `.autolab/model.json` is declared.

## Disposition and resumption

The mismatch exists in source, so the card's false-premise stop condition has
not fired. Infrastructure failure also cannot establish the replay kill
criterion. The experiment is incomplete and needs the supervisor's execution
service restored. Ignored `.autolab/outcome.json` records a blocked status,
without inventing a zero-gain or killed result.

The draft was committed and withdrawn from the active engine diff because no
complete model bundle exists. A default decode with the shipped model would
not evaluate this intervention. Resume by restoring the draft, validating its
tensor and teacher interfaces, admitting training-only data, completing the
identical-data refits and export, and replaying the card targets. Then follow
the user's target-win and practice-score rules. No fold or discard was run.

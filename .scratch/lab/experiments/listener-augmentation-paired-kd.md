# Paired-waveform teacher targets

Status: training correction implemented; model experiment blocked by the
authorized execution service. Target recovery and practice changes are unknown.
The experiment card is unchanged.

## Prediction and mechanism

The card predicts listener -1 and overall -1. Its hypothesis is that teacher
targets describe a different acoustic observation from the augmented student
input, suppressing faint consonants. Materialize each waveform once, share its
valid extent between the two front ends, and cache targets by transformed input
and teacher identity. SpecAugment remains student-only feature masking.

The proposed measurement is a bounded CPU phone-head refit on frozen encoder
features against an identical-data current-target control. Authorized Quran
recordings and help-train training speakers' none-labelled takes supply the
examples; release-held-out speakers are excluded. Both arms must share features,
transcripts, initialization and update order. Selection must use disjoint
speakers within the admitted training population, before target replay.
Vocabulary, frame timing, tracker, checker and verification rules stay unchanged.

## Demonstrated source mismatch

- `modal/tlog_pass.py::TeacherPass` computes targets on original mono/resampled
  waveforms, without RIR, noise or random gain. It uses bfloat16 inference and
  persists float16 top-five probabilities.
- `modal/student_prep.py` joins these targets with the recording and packs the
  FLAC waveform. Recording identity is preserved.
- `modal/student_train.py::ClipDataset.__getitem__` then applies stochastic RIR,
  noise and gain to the student waveform. Before this change it returned the
  original target slice despite that acoustic transformation.
- The existing KD delay/clamp and output-length masking were already correct;
  no change to their frame alignment is proposed.

This establishes an augmentation mismatch in the training recipe, beyond the
earlier recording-provenance audit. It does not establish its magnitude in
historical caches, or show that it caused a particular practice miss. Prep also
quantizes audio to PCM16 FLAC; this experiment does not separately correct that
conversion or change historical unchanged-waveform targets.

## Implemented change

`modal/student_paired_kd.py::AcousticInput` owns immutable float32 waveform
bytes, sample rate and valid sample extent. `ClipDataset` materializes the
augmentation once; student fbank and the teacher processor receive that same
trimmed observation. Artificial batch padding remains outside the valid extent.
The existing valid student output mask still excludes padding from KD.

When KD is enabled, changed waveforms get fresh teacher targets. A bounded
process-local cache identifies them by waveform bytes, extent/mask, sample rate,
actual teacher parameters and processor configuration. The teacher and stored
target precisions match the original pass. Unchanged waveforms keep their
existing targets. `paired_kd=False`, also exposed through the training entry
point, retains original targets for the control without mutating the batch.

The helper captures CPU frozen features at `ctc_output`, trims invalid output
frames, and fits only the existing linear phone head. Paired/current arms share
the feature snapshot, transcripts, original weights and update order. Refit is
bounded to at most 512 updates and selects on transcript loss plus original
output preservation using training-only validation speakers. It rejects
overlapping fit/selection speakers, unadmitted or release-held-out speakers,
empty speaker identities and scripted help-train mistake labels.

These are callable refit primitives, not a complete experiment driver. Actual
data admission, historical-target comparison, feature preparation, teacher
inference, refitting, selection and SDK bundle export remain unfinished. No
ready-to-use model is claimed. The implemented training change remains on this
branch, correcting the previous attempt's withdrawal of all engine code.

## Validation and limits

`arch -x86_64 python3 -m unittest discover -s modal -p test_student_paired_kd.py -v`
passes six lightweight checks: transform-specific cache identity, valid extent
and teacher identity, immutable input/targets, correct eviction, rejection of
scripted help-train labels, and invalid observation/speaker rejection. Python
compilation and `git diff --check` pass.

The torch integration class is skipped locally because torch and soundfile are
unavailable. Its prepared checks cover actual data/loss functions, gain and
combined RIR/noise/gain waveform sharing, unchanged-target preservation, zero
padding gradients, control-batch preservation, frozen feature capture, equal
targets producing identical refits, and speaker exclusion. These tests and the
real Muaalem adapter have not been validated in the training environment.

The authorized supervisor repeatedly fails before command execution:

| Request | Purpose | Result |
|---|---|---|
| 1 | Standard execution environment probe | `NotFoundError` |
| 2 | Training-image dependency/data probe | `NotFoundError` |
| 3 | Six shards of core/SDK validation | `NotFoundError` |
| 4 | Training-image tensor integration tests | `NotFoundError` |
| 5 | Resumed execution probe (`pwd`) | `NotFoundError` |
| 6 | Resumed six-shard core/SDK validation | `NotFoundError` |
| 7 | Resumed training dependency/checkpoint/help-train probe | `NotFoundError` |
| 8 | Resumed tensor integration tests | `NotFoundError` |

Requests/results are retained in ignored `.autolab/modal-*.json`. The errors
are `NotFoundError: No such file or directory`, with zero reported compute
cost. They do not establish a training or test failure inside the container.
Core/SDK pass status is unknown. No alternative Modal access, local acoustic
training, scorer changes, test-data changes or benchmark-specific rules were
used.

Targets evaluated: 0. Target wins, listener change and overall change are
unknown. Practice-score calls: 0. No refit, selected checkpoint, target replay,
studio validation or model export ran. No `.autolab/model.json` is declared;
scoring the shipped model would not evaluate this intervention.

## Disposition

The source premise holds; neither a false premise nor the replay kill criterion
has been demonstrated. Execution failure cannot establish a killed result.
The experiment is incomplete, and ignored `.autolab/outcome.json` records the
execution blocker without inventing a zero gain. No fold or discard was run.

Resume after the supervisor service is restored: validate tensor and teacher
interfaces, admit training-only data, perform the identical-data refits, select
and verify the complete standard SDK export, then replay targets. If fewer than
three targets recover relative to the control, follow the user's kill/revert
rule; otherwise run practice scoring before deciding the disposition.

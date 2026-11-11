# Training crop supervision censoring

Status: killed on the training-construction premise. No engine change retained.

## Prediction and mechanism

The card predicts listener −1 and overall −1 on practice. The mechanism would
preserve original recording coordinates when constructing training crops and
marginalize the partial occupancy of phones intersected by those crops. A small
frozen-feature pronunciation scorer would be fitted with this supervision and an
identical-data complete-label control. Only a written-over-rival preference
reversal relative to the control could veto an existing string-path wrong word;
primary decoding, tracker events, graph confirmations and warning bars would
remain unchanged. Model selection would use training-speaker partitions only.

The initial prerequisite is an existing crop construction that cuts supporting
speech while retaining labels from the longer recording. The task requires
stopping when the mechanism's premise is absent, and requires at least three
target recoveries before a practice score.

## Finding

The shipped `spliced-clean` training recipe has no such crop construction.
`docs/benchmarks/student-v2.json` records `max_clip_s=30`, `target=spliced` and
`buckets=clean-match`; its reproduction command uses `modal/student_train.py`.

- `load_clips` (`modal/student_train.py:151`) filters by recording length at
  line 166. It excludes a recording over the cap instead of taking its first
  30 seconds. The subsequent slices select complete FLAC byte buffers and their
  corresponding complete teacher posterior buffers, not acoustic subintervals.
- `ClipDataset.__getitem__` (line 206) decodes that complete FLAC buffer and
  computes fbank from the whole waveform. Noise, gain and peak-aligned reverb
  preserve its sample count. SpecAugment masks values in a clone of the full
  feature matrix; it never shortens that matrix or selects a temporal crop.
- `collate` (line 266) pads complete feature matrices and records their actual
  lengths. `losses` (line 306) uses those lengths; attention chunk masks change
  available context without creating separately labelled crop examples.
- `student_prep.py` resamples and encodes whole input recordings. The teacher
  pass (`modal/tlog_pass.py`) also decodes whole recordings and rejects overlong
  ones, deriving its greedy edge labels from the same recording. The student
  target (`tools/student.py:88`, `tools/per_breakdown.py:115`) uses reference text
  between exact anchors and teacher labels outside them. It does not restore a
  missing canonical prefix or suffix at a recording boundary.

Thus there are zero **constructed training crops inheriting pre-crop complete
labels** in this recipe, for any phone class. Creating a new random-crop dataset
with deliberately incorrect complete labels and then correcting it would test a
newly introduced defect; it would not establish the card's explanation for the
shipped listener's warnings. No scorer was fitted and no veto was installed.

## Validation and limits

A read-only Modal diagnostic was prepared to execute the actual `load_clips`, `ClipDataset`,
augmentation and target functions, extracting their definitions without loading
the credential-dependent Modal app. Parquet IO uses fixtures; FLAC decoding,
tensors and augmentation are real. A fbank spy captures the complete waveform
presented to feature extraction. Fixtures exercise the cap and byte-buffer
offsets at 16,000, 479,999, 480,000 and 480,001 samples, with augmentation both
absent and forced on. Generic edge-label checks cover every inventory symbol.
These are software-construction checks, not synthetic training data or acoustic
performance measurements. Diagnostic source and outputs are retained only under
ignored `.autolab/` artifacts after the premise audit.

Both supervisor requests failed before producing container logs or executing
the checks:

- Modal run 1: `python3 modal/student_crop_audit.py` —
  `NotFoundError: No such file or directory`, cost $0.
- Modal run 2: the supplied six-shard `bash tools/autolab_tests.sh` — the same
  `NotFoundError`, cost $0. No core or SDK test pass is claimed.

A lightweight local loader-only attempt also stopped before executing the audit:
the installed NumPy extension is x86_64 while Python runs as arm64. No packages
or environment settings were changed. The diagnostic parsed successfully with
`python3 -m py_compile`; this establishes syntax only.

The killed result rests on the inspected source construction and its recorded
shipped recipe, **not** on successful fixture execution, a trained comparison or
a measured acoustic recovery. The infrastructure errors are validation limits,
not the reason for killing the hypothesis. The ignored audit can be rerun when
the supervisor's normal execution environment is available.

This audit does not prove that every original recording begins or ends at a
complete articulation, that its teacher labels are correct, or that fbank and
encoder frame geometry are ideal. Original recording-edge label uncertainty is
distinct from inheriting longer-source labels after a training crop, and was not
measured here. No release-held-out audio, practice audio, labels or evidence were
used to fit or select anything. No target recovery is claimed: targets won 0,
with no runtime veto evaluation. No practice score was requested; overall change
is unmeasured. The experiment card, scorer, lab tooling, manifests and supplied
evidence are unchanged. No fold or discard command was run.

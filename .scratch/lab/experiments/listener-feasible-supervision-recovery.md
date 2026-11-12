# Feasible full-recording CTC supervision recovery

Status: killed on the recovery premise. No engine changes or model refit.

## Prediction and mechanism

The card predicts one recovered added word, no losses, listener −1 and overall
−1 on practice. Its five registered targets belong to listener/added_word/missed.
The proposed mechanism makes CTC feasibility explicit, including the blank
separator needed between adjacent equal labels, then recovers invalid examples
from intact recordings or corrected valid lengths. A duration-changing training
transformation would fall back to its feasible original. Only after establishing
such recoverable supervision would the experiment freeze the encoder and compare
a bounded phone-head refit against an identical-recording, identical-update-count
control, using authorized transcripts and training speakers' none-labelled takes.

The required premise is an example that currently has an infeasible target but
becomes feasible through one of those recovery routes, with the same transcript
and deployed acoustic clock. The checked-in training recipe has no such route.

## Audit result

`modal/student_train.py:306` does use `zero_infinity=True` for CTC. An impossible
alignment can therefore contribute zero sequence loss. That observation alone
does not establish recoverable supervision; the shipped recipe also applies
teacher KD independently, so zero CTC does not mean zero total supervision.

The waveform and length audit found:

- `modal/student_prep.py:48` decodes the complete source recording, averages
  channels, resamples to 16 kHz and writes the whole waveform to FLAC. There is
  no temporal crop. Amplitude clipping does not remove samples.
- `modal/student_train.py:151` admits recordings under the duration cap and
  rejects longer recordings; it does not truncate admitted recordings.
- `ClipDataset` reads the entire FLAC. Reverb trims its convolution to the
  original sample count, noise mixing matches that count, and gain preserves it.
  SpecAugment masks values without removing feature frames. None of these
  transformations changes the target.
- `collate` obtains each valid feature length from the actual feature tensor,
  before batch padding. `Student.forward` propagates the lengths returned by the
  subsampler and encoder. `losses` passes those output lengths directly to CTC;
  the teacher's stored frame count is used for KD, not as CTC's input length.

The trainer pins icefall to
`3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`. Inspection of that pin's
`subsampling.py` and `zipformer.py` confirms that the training output length is
`((feature_frames - 7) // 2 + 1) // 2`: three time-axis convolutions followed by
the encoder's final downsample, rounding odd lengths up. Those are the lengths
already returned to CTC. The streaming embed's separate right-context handling
is not used in this training forward path.

For a target `y`, CTC needs
`len(y) + count(y[i] == y[i - 1])` frames. Independently enumerating all 3,280
blank/two-label paths of lengths zero through seven verified this minimum for
all 31 two-label targets of lengths zero through four, including repeated labels.
Applying the actual convolution geometries and final downsample independently
matched the reported formula for all 5,990 feature lengths from 11 through
6,000. These lightweight checks passed; their script and result remain in ignored
`.autolab/feasibility-audit/` artifacts.

`docs/benchmarks/student-v2.md` identifies `spliced-clean` as the shipped recipe,
with `--target spliced` and KD weight 2. `tools/student.py:88` and
`tools/per_breakdown.py:42` collapse adjacent equal labels in that target. Thus
the additional repeated-label separator requirement is zero for this recipe,
although the general feasibility calculation must include it for other targets.

For every admitted example in this path, restoring the intact original leaves
both its frame count and target unchanged. Correctly computing valid lengths
also leaves the lengths unchanged. If its target is infeasible, it remains
infeasible; an augmentation-only structural failure cannot occur. Training a
head anyway would test ordinary refitting rather than recovered supervision.

## Validation limits and disposition

This is a source and structural audit, not a census of historical training
Parquet rows or an acoustic evaluation of the five targets. It does not claim
that historical training contained no zeroed CTC examples, nor that a feasible
target necessarily received good acoustic supervision. It establishes that the
proposed same-transcript, same-clock recovery cannot rescue an invalid example
through the checked-in preparation and training path. No evidence was obtained
of historical cached recordings deviating from that path.

Supervisor Modal run 1 requested an inventory of the supplied training data and
experiment scratch directory. Run 2 requested the supplied six-shard
`bash tools/autolab_tests.sh` harness. Both clients exited with
`no result for modal run ... in 90 minutes; the supervisor may be paused`.
No remote data audit or passing core/SDK suite is claimed. There is no changed
engine behavior requiring new unit tests.

Stop on the premise, with zero targets recovered and no practice score. No
head was trained, selected or exported; no model bundle is declared. The
experiment card, engine, scorer, tooling, supplied data, manifests and evidence
were not changed. This report is the only tracked change. The untracked
`.autolab/outcome.json` records the killed outcome. No fold was performed.

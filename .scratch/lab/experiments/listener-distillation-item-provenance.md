# Listener distillation item provenance

Status: killed on the pairing premise. The experiment card is unchanged; no
engine or model changes remain.

## Prediction and mechanism

The card predicts listener −1 and overall −1: teacher supervision may be paired
with another utterance after sorting or packing, and correcting that pairing
may recover a planted addition. The proposed implementation carries audio,
transcript, teacher posteriors and valid lengths in one immutable item record.
The bounded phone-head refit and identical-data continuation control depend on
a demonstrated pairing correction; training-speaker partitions would select
checkpoints before target diagnostics or practice scoring.

## Shipped source audit

Audited the branch's unchanged shipped training path, based on `81640f28`,
including the documented `spliced-clean` recipe (`docs/benchmarks/student-v2.md`:
spliced CTC targets and top-five KD weight 2).

- `modal/tlog_pass.py`: decoded entries are `(source-row record, waveform)`
  pairs. Duration sorting moves those pairs together. Each flush sends the
  waveforms in pending order and attaches the returned teacher posteriors to
  the matching pending row. `TeacherPass.infer` restores neither axis by a
  separate permutation; its valid-length mask and outputs share batch order.
- `modal/student_prep.py`: teacher rows are indexed by source row within the
  shard, then joined to audio at that same source row. Manifest iteration order
  and the teacher parquet's duration order cannot change this join.
- `modal/student_train.py::load_clips`: filtering and random subsampling select
  row indices once. All fields are read from the same table selection. FLAC and
  teacher buffers are concatenated in that order; top-ID and probability
  payloads use the same five-values-per-frame offsets.
- `ClipDataset` uses one item index for waveform, transcript and teacher slices.
  The spliced transcript is derived from the reference and teacher transcript
  at that same index. `batches` sorts indices, and `collate` uses the resulting
  item order for every field and valid length, including concatenated CTC targets.
- `losses` gathers teacher frames along the time axis separately for each batch
  item, retaining the batch axis. The student forward path has no item sorting.
  `evaluate` assigns each sorted-batch hypothesis back through `b["i"]`.
  Export converts checkpoint weights and does not recombine training items.

No independent item permutation was found at these boundaries. An immutable
record could prevent future mistakes, but this audit has identified no changed
supervision for that refactor to correct.

## Validation and limits

The lightweight premise probe passed, running the extracted shipped functions
with an independent immutable source-item oracle. It covered three synthetic
shards with unsorted source order, distinct equal-duration items, differing
teacher frame lengths, four subsampling seeds, hours/count limits, four batch
seeds, shuffled and duration-sorted batches, and both teacher and spliced CTC
targets.

| Check | Passed |
|---|---:|
| Loads / total selected item instances | 12 / 178 |
| FLAC and teacher-buffer offset checks | 178 |
| Batch feature/transcript/teacher/length identity checks | 2,848 |
| KD gathered-target identity and valid-length checks | 2,848 |
| Restored evaluation-item checks | 356 |
| Deliberately swapped equal-length teacher batches detected | 12 / 12 |
| Uninjected identity mismatches | 0 |

No engine behaviour changed, so no permanent unit tests were added. The
untracked diagnostic and result remain in `.autolab/provenance_source_probe.py`
and `.autolab/provenance-source-result.json`; its command was
`arch -x86_64 python3 .autolab/provenance_source_probe.py`. This small CPU-only
source probe ran locally; no training, acoustic inference or heavy compute ran
on the Mac.

Parquet reads, tensor operations, fbank and neural model outputs are
stand-ins backed by NumPy and real FLAC decoding. Numerical CTC/KD losses are
patched; the shipped KD time-gather and output-restoration functions execute
unchanged. Real transcript helpers are used. This validates data identity, not
acoustic inference or loss numerics. A same-duration teacher-target swap is
the negative control.

Historical training-cache inspection was unavailable: both supervisor requests
using the training image returned `NotFoundError: No such file or directory`.
The GPU request was rejected because this explore card has no GPU allocation.
The standard CPU environment request (run 5) and all six requested test shards
(run 4, `bash tools/autolab_tests.sh`) also returned the same `NotFoundError`
before execution. Core and SDK test results are therefore unavailable, not
reported as passing.

These are compute limitations, not evidence about cache contents. Source and
synthetic checks cannot prove that an old cache was generated from these exact
source files, or rule out manual corruption. They also do not explain the
teacher/student consonant discrepancy.

## Verdict

Stop at the card's first kill criterion: the audited shipped path and synthetic
identity checks found no teacher-target item mismatch to correct. The immutable
item refactor would preserve the same supervision, rather than repair an
observed defect. Historical cache contents remain unverified as described
above; this is not a claim that every historical payload is authentic.

The temporary engine-side probe was removed. No head refit, continuation
control, checkpoint selection, target acoustic recovery test or model export
was performed, since all depend on a demonstrated pairing correction. There
were zero practice-score calls and no measured overall delta. The outcome's
`targets won: 0` records no realized recovery from this killed experiment; it
is not a five-target acoustic comparison. `.autolab/outcome.json` records the
killed status, and no `.autolab/model.json` or new model bundle is declared.

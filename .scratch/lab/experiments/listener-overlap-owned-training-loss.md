# listener-overlap-owned-training-loss

Status: killed — the current training path does not create overlapping training
windows with duplicate valid-frame supervision.

## Prediction and mechanism

The card predicted one recovered added-word mistake, zero losses, listener −1
and overall −1. The proposed explanation was repeated KD or auxiliary loss on
the same source-recording observations when training windows overlap. The
intervention would retain the overlap as acoustic context, partition framewise
supervision by source coordinates, leave transcript CTC unchanged, and compare
an encoder-frozen phone-head refit against an identical-window,
identical-update-count control. Checkpoint selection would use authorized
training recordings and help-train training speakers only.

The premise check fails before that refit: there is no training-window ownership
to correct in this implementation.

## Source audit

- `tools/student_manifest.py` identifies each example by source shard and row;
  it supplies no temporal crop coordinates. `modal/student_prep.py:61` indexes
  requested examples by row, then reads and stores the entire source waveform
  and its complete teacher arrays. `modal/tlog_pass.py` also sends complete
  waveforms to the teacher; it batches by duration and rejects overlong clips.
- `modal/student_train.py:151` selects prepared rows without replacement.
  The duration cap rejects a clip rather than splitting it.
  `ClipDataset.__getitem__` at line 206 reads the full item, applies
  length-preserving waveform augmentation, and computes its fbank once.
  SpecAugment masks features within that item. The flat-buffer offsets locate
  packed recordings and teacher arrays; they are not overlapping windows.
- `batches` at line 282 permutes item indices, sorts within duration pools, and
  places each index in one batch. The epoch loop at line 451 loads this plan and
  performs one loss call and optimizer update per finite batch. It does not
  accumulate losses across sliding windows.
- `Student.forward` at line 134 calls the subsampler and chunk-masked encoder
  once on the full padded batch. Chunk size and left context constrain
  attention; they do not construct additional training examples or loss rows.
- `losses` at line 306 computes full-sequence CTC and, when enabled, one KD
  reduction. Its validity mask selects each student output row once.
  `tools/student.py:77` sums each valid row's top-k cross-entropy once and divides
  by the number of valid rows. Padding contributes zero. There is no auxiliary
  framewise objective in this training path.
- The actual overlapping windows are in
  `modal/student_export.py:92`: ONNX inference consumes `2 * chunk + 13` fbank
  frames and advances by `2 * chunk`. This path concatenates successive CTC
  output blocks and carries inference state; it computes no training loss.

The documented `spliced-clean` recipe in `docs/benchmarks/student-v2.md` invokes
this trainer with spliced **text targets**, not overlapping audio crops.

## Validation

A lightweight local audit executed the production `batches` function extracted
by AST, without importing Modal or constructing an encoder. It checked 360
plans: item counts 0, 1, 3,999, 4,000, 4,001 and 8,005; padded-duration budgets
1, 120 and 600 seconds; seeds 0–9; and both shuffle modes. Across 1,200,360 item
visits there were **zero duplicate visits and zero omitted visits**. This covers
the duration-pool boundary and batches with clips exceeding the batch budget.
These are synthetic item counts, not measurements on training recordings or
practice incidents.

The full KD gradient fixture and existing Python tests were also requested from
the supervisor, along with the six-shard core/SDK build and test command. The
first two requests and both retries returned
`NotFoundError: No such file or directory`, with no test logs. Neither the KD
gradient fixture nor `tools/test_student.py` nor the core/SDK suite is reported
as passing. The initial helper was in the ignored `.autolab` directory; the
audit retry embedded its code in the command and used the training image.
The local full fixture could not run:
native Python's NumPy has an incompatible binary architecture, and the x86
interpreter has no PyTorch. No gradient-test pass is inferred from the sampler
result or the source audit.

No behavior was changed, so no new persistent unit tests were added.

## Limits and decision

This establishes absence of duplicate supervision from generated overlapping
training windows in the checked-in preparation and training implementation.
It does not audit unidentified upstream copies of recordings or quantify every
other possible source of duration bias. Teacher-index clamping may reuse an
endpoint target for different student observations; that is a separate frame
alignment issue and does not establish the proposed window duplication.

No encoder/head refit, checkpoint selection or model export was performed.
The six named target streams were not re-decoded, so no acoustic recovery claim
is made. Targets won is recorded as 0 for this unbuilt intervention; listener
and overall score changes are unmeasured. No practice-score calls were used.

The experiment stops on the absent premise, as required. Engine code, the card,
model declarations, scoring code and test data are unchanged.

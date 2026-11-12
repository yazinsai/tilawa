# Listener subprecision update retention

Status: killed on the premise. The current trainer already retains float32 phone-head parameters and optimizer updates. Experiment card unchanged.

## Prediction and mechanism

The card predicts one recovered added-word miss, listener −1 and overall −1.
Its speculative mechanism is that reduced-precision phone-head parameter writes
round away small, transcript-supported corrective updates. The proposed remedy
is float32 master parameters and optimizer updates, with reduced precision used
only for the forward computation, followed by a bounded encoder-frozen refit
against an identical-data, identical-update-count current-procedure control.

The premise must hold before refitting: at least one missing target consonant
class must have an update that exists before the current parameter write and
vanishes through its precision conversion. If the trainer already uses float32
parameters and writes, a second master copy introduces no such contrast.

## Source audit

The current trainer already implements the proposed precision separation:

- `modal/student_train.py:132` constructs the phone head with ordinary
  `nn.Dropout(p=0.1)` and `nn.Linear(max(dims), VOCAB)`, with no reduced dtype.
- `modal/student_train.py:425` constructs the model and calls `.to(device)`
  without a dtype conversion. ScaledAdam receives those model parameters.
- `modal/student_train.py:311–315` confines bfloat16 autocast to the forward
  computation, then converts log probabilities to float32 for the acoustic loss.
- `modal/student_train.py:469–470` calls backward and the optimizer outside the
  autocast context.
- Resume and `load_student` use ordinary `load_state_dict`, which copies into
  the existing destination parameter dtype. Saving uses the model and optimizer
  state dictionaries without conversion. No reduced-precision parameter cast or
  changed default dtype occurs in `modal/student_*.py` or `tools/student.py`.

The optimizer is pinned to icefall revision
`3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`. Its [Zipformer optimizer source](https://github.com/k2-fsa/icefall/blob/3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1/egs/librispeech/ASR/zipformer/optim.py) has SHA-256
`7282cd6db1d234e086110feb1f39f076a76095a6b026b939cebc81f2ed0f5d45`.
`BatchedOptimizer.batched_params` groups by parameter dtype, uses `torch.stack`
without conversion, and writes updated tensors back with `p.copy_`.
`basic_step`, `scaling_step`, and `momentum_step` allocate the squared-gradient,
scale, and accumulated-update state in `torch.float`. `ScaledAdam.step` adds the
update to the stacked parameter before that direct copy. A float32 head therefore
has float32 parameters, optimizer arithmetic and writeback; there is no
bfloat16 parameter-storage boundary at which the card's updates could vanish.

`modal/student_export.py` reuses the same head for streaming export. Inference
quantization is a separate boundary and does not reduce the parameter storage
used during training. Export quantization and reduced-precision forward or
backward computations are outside this experiment's write-retention mechanism.

## Runtime validation and limits

Two supplied-supervisor requests failed with
`NotFoundError: No such file or directory`: run 1 requested retrieval of the
pinned optimizer and runtime availability; run 2 requested the small CTC
writeback audit. Neither returned usable runtime output, so no runtime assertion
is counted as passed. The pinned source was retrieved directly for the static
audit; no alternate Modal access was attempted.

The prepared audit would read no recordings, labels or practice evidence. It
would extract the exact lite phone-head expression from the trainer and execute
the pinned optimizer's unmodified definitions. Small synthetic cached features
and a CTC transcript are intended to generate updates below a bfloat16 unit of
precision. The fixture would compare the optimizer's tensor immediately before
writeback with the actual parameters afterward, and separately apply a
hypothetical bfloat16 conversion. It would also check gradient, parameter and
optimizer-state dtypes, repeated update accumulation, and checkpoint round trips.

Had it completed, this fixture would have tested parameter-write semantics,
not whether any target audio has a useful acoustic gradient. No training-speaker
refit, target recovery, practice score or release benchmark is claimed.

The required six-shard core/SDK build and test harness was requested as supervisor
run 3 and failed with the same `NotFoundError: No such file or directory`.
No core/SDK test pass or failure was obtained. No engine behavior changed, so no
behavior-specific unit test was added. Local validation was limited to syntax-compiling the audit
script and checking the source and final diff.

## Outcome

The premise is absent in the current training procedure: there is no
reduced-precision phone-head parameter write. Adding float32 master parameters
would duplicate the existing storage precision, so a matched refit would not
isolate the proposed mechanism. This source finding applies to all phone-head
classes without using target identities, recordings or labels. It does not
establish the precision of every historical training run or rule out errors in
forward computation, gradients, or inference quantization.

Stop before refitting. No engine implementation was edited, no replacement
model was trained or declared, and no practice score was consumed (0 of 3).
The five registered targets have no recoveries from this branch: `targets won`
is 0, recording that no changed listener was delivered, not an acoustic replay
measurement. No overall or listener score delta is claimed.

Only this write-up is committed. The ignored `.autolab/outcome.json` records
the killed result. No scorer, lab tooling, test data, manifest, evidence or
experiment-card changes were made, and no fold was performed.

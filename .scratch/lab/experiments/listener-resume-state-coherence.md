# Listener resume state coherence

Status: killed on the premise. The checked-in resume path already restores all
four state components. The experiment card is unchanged.

## Prediction and mechanism

Predict listener −1 and overall −1, recovering one added-word miss without
counted losses. The proposed defect is resumed optimizer moments combined with
fresh update counters or a reset learning-rate scheduler position. A corrected,
versioned checkpoint transaction would restore the phone head, moments,
per-parameter update counters and scheduler position together. Only a demonstrated
defect would justify the bounded frozen-encoder head refit against an
identical-data, identical-update-count control using the actual old resume path.

## Source audit

`modal/student_train.py:427–436` constructs ScaledAdam and Eden, then loads the
checkpoint's `model`, `opt`, `sched` and `state` before any training update.
`model.load_state_dict` includes the phone head. The optimizer state is loaded
whole; the trainer does not extract moments or construct replacement counters.
Missing required top-level sections raise `KeyError`; there is no exception
handler that continues training with a partial restore.

The save at `modal/student_train.py:442–446` writes those same four sections plus
the configuration into one temporary file, replaces `ckpt.pt` with
`os.replace`, and commits the volume. The trainer therefore does not assemble
its model and optimization state from independently written checkpoints.

The dependency is pinned to icefall
`3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`. Its
[Zipformer optimizer source](https://github.com/k2-fsa/icefall/blob/3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1/egs/librispeech/ASR/zipformer/optim.py)
stores same-shape/dtype parameters in batches. Each batch's state is attached to
its first real parameter and serialized by PyTorch's `Optimizer.state_dict`.
The same state mapping holds `step`, `delta`, `exp_avg_sq`, `param_rms`,
`scale_grads`, and `scale_exp_avg_sq`, with clipping history in the group's first
batch. ScaledAdam inherits PyTorch's state loading; counters are not omitted.
The trainer pins PyTorch 2.8.0. Its
[optimizer serialization implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L743)
serializes entire per-parameter state mappings, and `load_state_dict` recursively
loads every key. ScaledAdam's Python integer `step` is returned unchanged by
that recursive loader. The same checkpoint therefore supplies the update counter
and its matching moments.

Eden's scheduler state contains both `epoch` and `batch`. The trainer restores
them and applies `step_epoch(state["epoch"])` before its next batch. After an
update it calls `step_batch(state["step"])` before incrementing the successful
update count. The scheduler's batch index being one behind `state["step"]` is
therefore the normal update order, shared by uninterrupted and resumed runs.
At an epoch-boundary save, the progress epoch has advanced while the scheduler
still holds the completed epoch; the next loop iteration advances the scheduler
in both executions.

`init_from` is an explicit encoder-only initialization with fresh optimization,
used only when no run checkpoint exists. `load_student` restores the model for
inference/export. Neither is the training resume path.

## Validation and limits

A temporary synthetic CPU audit was prepared to extract and execute the actual
trainer's save function and resume statement using the pinned ScaledAdam and
Eden. It would compare complete model, optimizer, scheduler, learning rates and
progress state against uninterrupted training after checkpoints inside epochs,
at epoch boundaries, and across optimizer scale/clipping update cycles.
Repeated parameter shapes and two parameter groups exercise packed optimizer
state. Missing top-level model, optimizer, scheduler and progress sections would
be checked for rejection.

Supervisor runs 2 and 3 requested that audit with the training image. Runs 1 and
4 requested the supplied six-shard core/SDK build and test harness. The first
pair returned `OSError: unexpected end of data`; the retries returned
`NotFoundError: No such file or directory`. No command logs, audit result or
test results were returned. Runtime continuation equality and passing tests
are therefore unverified. The kill decision rests on the checked-in trainer
and the exact pinned dependency source, not on those infrastructure failures.
The temporary probe was removed from engine sources and retained only with
ignored diagnostic artifacts under `.autolab/resume-audit/`.

This audit concerns coherent state saved by the current trainer with unchanged
model parameter groups and scheduler configuration. It does not establish
bitwise acoustic-training reproducibility across random augmentation, dropout,
worker random state or changed resume settings; those are outside the proposed
moments/counters/scheduler discontinuity. It also does not claim schema
validation for arbitrary nested checkpoint corruption. No real training
checkpoint provenance, target acoustics, or warning gains have been measured.

No acoustic refit, training-data selection, model export or practice score has
been performed. No engine behavior has been changed, so no new behavior tests
are retained. All three allowed practice scores remain unused. The seven target
spans were not decoded or measured; `targets won: 0` in the killed outcome
records that no intervention was produced, not a scored zero-gain acoustic arm.

The card's first kill condition fires: checkpoint restoration already preserves
the complete specified update state. Adding checkpoint versioning or refitting
the head without a demonstrated old-moments/new-counters discontinuity would
not test this mechanism. The report is the only committed experiment change;
no scorer, lab tooling, test data, supplied manifest, evidence, tracker, checker,
SDK decoder or experiment-card edits were made. `.autolab/outcome.json` records
the killed result. No fold was performed.

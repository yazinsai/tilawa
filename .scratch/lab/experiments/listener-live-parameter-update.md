# Listener live parameter update

Status: killed on the source-level absence of the proposed ownership defect.
Runtime update parity remains unverified. Experiment card unchanged.

## Prediction and mechanism

The card predicts one recovered addition, listener −1 and overall −1, with
substantial zero-gain risk. The proposed defect is an optimizer retaining
superseded parameters while backward computes gradients on replacement live
parameters. Repair would bind groups to the final model, require exactly one
owner per intended trainable parameter, and transfer state only with verified
identity and shape before a matched, frozen-encoder phone-head refit.

The discriminator is live ownership and update equivalence to an independently
constructed reference optimizer. A refit without an ownership defect would not
isolate the card's mechanism. The planned training and checkpoint selection
would use authorized transcripts and help-train training speakers' none takes,
with release-held-out speakers excluded; none of that training was started.

## Source audit

The checked-in training path already retains live parameters:

- `modal/student_train.py:132` instantiates the ordinary dropout/linear phone
  head during model construction. `train` completes `build_model(...).to(device)`
  at line 425 before constructing ScaledAdam at line 427.
- `get_parameter_groups_with_lrs` iterates `model.named_parameters()` once and
  appends each parameter to one learning-rate group. The trainer passes no
  module exclusions. ScaledAdam extracts these same objects into `params`; it
  does not clone them into persistent optimizer-owned replacements.
- Resume loads the model, optimizer, scheduler and progress at lines 433–436.
  Model loading uses default `load_state_dict`, without `assign=True`.
  Encoder initialization at line 380 and inference loading at line 502 use
  the same copying semantics. No training-time module replacement, parameter
  wrapping, parameter assignment or dtype conversion occurs after optimizer
  construction in `modal/student_*.py` or `tools/student.py`.
- PyTorch 2.8.0's default model loader calls `param.copy_(input_param)`.
  Its optimizer loader maps saved state onto the current group's parameters
  and explicitly retains the group's existing `params` list. Loading checkpoint
  values or moments therefore does not substitute old checkpoint tensors for
  the live parameters.
- ScaledAdam's `batched_params` reads the group's parameters and their current
  gradients on each step, temporarily stacks same-shape/dtype parameters,
  then writes the updated rows back with `p.copy_(stacked_params[i])`.
  These temporary tensors are an update implementation, not stale owners.
- `modal/student_export.py:54` assigns existing modules into the streaming
  wrapper. It shares the phone head; it does not create a replacement head
  that is subsequently trained by the old optimizer.

The dependencies are pinned to icefall
`3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1` and PyTorch 2.8.0.
The retrieved [optimizer source](https://github.com/k2-fsa/icefall/blob/3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1/egs/librispeech/ASR/zipformer/optim.py)
has SHA-256 `7282cd6db1d234e086110feb1f39f076a76095a6b026b939cebc81f2ed0f5d45`;
the [group-construction source](https://github.com/k2-fsa/icefall/blob/3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1/icefall/utils.py)
has SHA-256 `35beb5077b458872621094d7b176adf58fb6a7d4de81a60f033184052b6555d6`.
The [model loader](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/module.py)
and [optimizer loader](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py)
were checked at those pinned versions, rather than inferred from a different
optimizer or an artificial replacement example.

Two preserved corrective-head implementations were also inspected as context.
`exp/listener-contrast-space-head-training` constructs every final head before
its ScaledAdam optimizer, and loads selected states only after updates finish.
`exp/listener-unconditional-count-training` applies its gradient directly to
each live `head.parameters()` tensor and has no separate optimizer-owned copy.
These checks are not an exhaustive audit of historical external training runs.

## Runtime validation and limits

A temporary engine diagnostic was prepared to extract and execute the current
trainer's model/optimizer/resume setup statements, using the actual lite model
and pinned ScaledAdam. It would check all live identities and missing, duplicate
and stale owners. Synthetic negative controls would check rejection of each
ownership defect; they are not evidence that a defect exists in this trainer.

The prepared comparison would construct a separate final model and new
optimizer, verify matching parameter names and shapes, restore equivalent state,
and copy the gradients from the actual trainer's acoustic loss into that
reference. It would compare every parameter and full optimizer/scheduler state
after fresh and resumed updates. A separate 104-update cached-feature CTC head
fixture would span the optimizer's scale and clipping-history periods.
Model/head state loading and export wrapping would be checked for identity
preservation. The probe uses synthetic features and checkpoints; it reads no
recordings, labels, practice evidence or prior model artifacts.

Supervisor run 1 (training-image availability), run 5 (the CPU ownership audit),
and runs 2 and 4 (the supplied six-shard core/SDK build/test harness) returned
`NotFoundError: No such file or directory`, without command logs. Run 3 was
rejected because GPUs are reserved for bet cards; this card is explore. No
runtime ownership count, numerical update-parity result or test pass was
obtained. Local validation syntax-compiled the probe, checked the exact pinned
source and checked the final diff. Infrastructure failures are not counted as
passed assertions or implementation test failures.

The source analysis applies to the current trainer with an unchanged model and
parameter-group configuration. It cannot certify unseen historical refits,
arbitrary checkpoint rewrites, or recognition quality. Identity-correct updates
also do not establish that the acoustic loss teaches the missing consonants.

## Outcome

The source-level premise is absent in the current procedure: groups enumerate
the live model once, and neither checkpoint loading nor export creates the
superseded tensors the proposed repair requires. Moving optimizer construction
after checkpoint copying would leave its parameter values, gradients, shapes,
groups and restored state unchanged. ScaledAdam's update reads those inputs
from the same live owners. There is no demonstrated stale or omitted group to
repair and no distinct repaired-versus-current refit to run.

This is a source-based rejection of the ownership mechanism, not a claim that
the card's numerical reference-update check passed. That runtime part of the
kill criterion remains unverified because the supplied supervisor could not
return the audit. No alternative Modal access was attempted.

Stop before refitting. The temporary diagnostic was removed from engine sources
and retained with ignored source-audit artifacts under `.autolab/parameter-audit/`.
No engine behavior changed, so no behavior-specific unit test is retained.
No training recordings were selected, no replacement model was trained or
declared, and no target replay or practice score was performed (0 of 3).
`targets won: 0` records that this branch delivers no changed listener, not a
measured acoustic zero-gain arm. No overall or listener score delta is claimed.

Only this report is committed; the ignored `.autolab/outcome.json` records the
killed result. No experiment-card, scorer, lab-tooling, supplied test-data,
manifest, evidence, tracker, checker or SDK changes were made. No fold occurred.

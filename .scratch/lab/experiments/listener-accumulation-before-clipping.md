# Accumulate the complete training gradient before clipping

Status: killed on the source premise; no engine change retained.

## Preregistered prediction

Experiment `listener-accumulation-before-clipping`, listener, explore. Predicted
one recovered added word with no losses: listener −1 wrong call, overall −1.
The card's listener is `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`
and its practice set is `lab-v1.5u`. `lab-experiment.json` was left unchanged.

## Mechanism

The proposed defect was clipping individual microbatch gradients before the
complete effective batch had contributed. The proposed correction would first
normalize all recording losses with the effective batch's weights, accumulate
their gradients, unscale once if scaling is used, clip once, and update once.
A frozen-encoder phone-head refit would compare that procedure with an
identical-data, identical-effective-batch current-update control, selecting
checkpoints only on authorized training-speaker partitions.

That comparison is conditional on finding the defect. The card explicitly
stops when the existing update already clips after the complete weighted
gradient has been formed.

## Source audit

Audited worktree base: `263f58c287e5c7b2047bf774d29165c0b2cce0ed`.

- `modal/student_train.py:276–305` collates duration-bucketed recordings into
  one batch. It does not split an effective update into microbatches.
- `modal/student_train.py:306–331` computes the complete batch's scalar loss.
  CTC uses a summed recording loss divided by total valid output frames.
  Optional KD is reduced over valid frames and combined using the existing
  CTC/KD weights. `tools/student.py::soft_target_loss` excludes invalid frames
  before the reduction.
- `modal/student_train.py:460–470` computes that loss, clears gradients,
  checks finiteness, calls `out["loss"].backward()` once, and then calls
  `opt.step()` once. The backward pass therefore combines all weighted
  recording contributions before the optimizer receives the gradient.
- `modal/student_train.py:427–428` uses icefall `ScaledAdam` with
  `clipping_scale=2.0`. There is no recording-local or microbatch-local
  clipping call in the trainer.
- The model forward uses bfloat16 autocast, with float32 log probabilities
  for the acoustic losses. The trainer has no GradScaler, scaled backward,
  or unscale operation to move.

A Python AST census of every `modal/student_*.py` found exactly one backward
call, the call above, and no `clip_grad*`, `GradScaler`, or `unscale*` calls.
Thus the purported sequence of separately clipped microbatch contributions
does not exist in the supplied training implementation. Changing batch size
or grouping multiple existing optimizer updates would change the effective
batch and update count, rather than correct this proposed ordering defect.

## Validation and result

The existing source was inspected without loading training or evaluation
audio. Supervisor request 1 asked to inspect the optimizer at the trainer's
pinned icefall revision `3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`.
Request 2 asked for the existing core/SDK build and tests through
`bash tools/autolab_tests.sh` on six shards. Both were submitted at
`2026-11-11T20:47:00Z`; neither had a supervisor result by
`2026-11-11T21:32:19Z`, more than 45 minutes later. The remote optimizer
implementation and runtime tests therefore remain unverified. No test pass
or remote clipping result is claimed. The premise conclusion rests on the
supplied trainer's complete-batch loss and single backward/update sequence.

No head refit, candidate model export, target decode, or practice score was
performed. Practice-score requests used: **0 of 3**. Targets won: **0**
(no intervention was built; this is not a measured paired acoustic result).
The overall and component score changes were not measured.

## Limits and disposition

This audit establishes the update order in the supplied student trainer. It
does not test whether changing clipping policy, batch size, optimization
conditioning, or supervision could improve weak consonants; those would be
different mechanisms. No claim is made about recovering a planted consonant,
preserving clean-session warnings, or satisfying the fold gate.

The existing complete-batch backward already precedes the optimizer update,
so the premise kill criterion fires. No engine or test behavior was changed,
and no replacement model is declared. The killed outcome is written to
untracked `.autolab/outcome.json`; only this experiment write-up is committed.

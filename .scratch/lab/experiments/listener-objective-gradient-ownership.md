# Listener objective gradient ownership

Status: killed on the source premise; no engine changes or model refit.

## Prediction and mechanism

The card predicted listener −1 and overall −1 on `lab-v1.5u`, from recovering a
missed added word. The proposed defect was separate backward operations
overwriting or detaching transcript gradients before teacher imitation reached
the phone head. A repair was conditional on demonstrating a difference between
the current update gradient and the sum of independently computed weighted
component gradients.

If that discrepancy existed, the experiment would combine the existing losses
without changing their weights, clipping or optimizer, then compare a bounded
frozen-encoder phone-head refit with the actual current procedure. Recordings,
updates and training-speaker checkpoint selection would be matched, with
release-held-out speakers excluded. The user's stopping rule requires at least
three target wins before practice scoring.

## Source audit

The inspected baseline is `fa2c6ab5` on
`exp/listener-objective-gradient-ownership`. The training recipe documented in
`docs/benchmarks/student-v2.md` uses `modal/student_train.py`, with CTC weight 1
and KD weight 2 for the shipped `spliced-clean` acoustic model.

The current trainer already has the proposed ownership structure:

- `losses()` computes CTC and initializes `loss = ctc_weight * ctc`.
- With KD enabled, it adds `kd_weight * kd` to that differentiable tensor.
- The training loop calls `opt.zero_grad(set_to_none=True)` once, then
  `out["loss"].backward()` once, then `opt.step()` once.
- There is no intermediate backward, gradient clear, `.grad` assignment or
  detached student loss in `modal/student_*.py`.
- `tools/student.py:soft_target_loss` gathers from the live student log
  probabilities. Converting those probabilities to float32 retains their graph.
  Cached teacher probabilities are targets, not student computations whose
  gradients should be retained.

There is no separately weighted preservation loss in this baseline trainer.
Adding one would change the objective rather than recover an existing gradient,
so it is outside this card's conditional repair.

These operations are at `modal/student_train.py:318`, `:329`, `:461`, `:469`
and `:470`; the KD helper is at `tools/student.py:77`. A local AST inspection
confirmed exactly one `zero_grad`, `backward` and optimizer `step`, in that
order, and no gradient assignment. `losses()` contains no `detach`, `detach_`
or `item` calls. The training loop's `item()` reads input lengths for throughput
logging; loss scalars are converted for logging after the optimizer update.

Audited file SHA-256 values:

- Trainer: `e5b2dc0d99880b686f562a80d9b28714ba2dd086609afc579cac02b67a5cb448`.
- Helper: `ec1d9e9acc4c09c3ec553c4c6cb16a76dbc9f7ffb86b9fc8501a541cde601ce9`.

## Numerical audit

The numerical audit could not execute. Its prepared harness would execute the
unmodified `losses()` function and the actual phone-head expression extracted
from the trainer's AST. For each component it would make a fresh forward pass
and obtain its weighted gradient independently, restoring random state before
each forward so the head's dropout mask is identical, then summing those
component gradients.

The candidate side would execute the trainer's actual gradient-clear, backward
and optimizer-step statements. A capture optimizer would record the gradient
arriving before clipping or optimization; deliberately preloaded stale
gradients must be cleared. This isolates objective ownership from optimizer
transformations.

The planned 72 synthetic-feature cases cover three random seeds, head dropout
0 and the current 0.1, equal and unequal recording lengths, shorter and longer
teacher targets, KD disabled and the shipped weight 2, and frame delays 0, 2
and −1. Every enabled component must have a finite nonzero head gradient.

## Validation and limits

Supervisor runs 1 and 2 requested the gradient audit plus existing Python
student tests, and the prescribed six-shard core/SDK build and tests. Both
returned `NotFoundError: No such file or directory` without shard logs or
results. Runs 3 and 4 retried the six-shard tests with explicit `--image node`
and the audit/tests with explicit CPU `--image train`; both returned the same
error without shard results. The retry embedded the audit in its command
because ignored `.autolab` scripts are not included in the supervisor's source
bundle. The numerical comparison and all requested runtime tests remain
**unverified**; there are no gradient-error measurements or test-pass counts.

Local AST checks and Python syntax compilation of the audit harness passed.
No engine behavior was changed, so no behavior-change unit tests were added.

The prepared numerical audit uses CPU float32 frozen-feature inputs. It would
not test CUDA bf16 numerical rounding, the encoder's custom backward functions
or the mathematics of ScaledAdam clipping. Those are separate questions from
clearing or detaching a phone-head objective gradient. Synthetic features
suffice to test composition; they establish no acoustic recovery on a target
recording.

No training recordings, help-train labels or practice audio were used; no
checkpoint was selected, no model was exported, no full-session target replay
was run and no practice scores were requested (0 of 3 used). Target wins recorded
as zero mean that no candidate was produced, not a measured refit's failure to
recover those spans. Overall and listener score changes are unmeasured.

## Decision

The proposed repair is already present in the current procedure. There are no
separate loss backward passes or intermediate clears to remove, and no detached
student loss to reconnect. No difference from the component-gradient sum was
demonstrated. The runtime infrastructure failure leaves numerical parity open;
it does not supply evidence for the card's proposed defect.

Stop on this absent source premise, as requested, rather than introduce another
objective or refit an unchanged control. The engine, optimizer, clipping,
coefficients, alphabet, timing, tracker, checker and insertion verification
remain unchanged. `.autolab/outcome.json` records `killed` with zero candidate
target wins and no claimed practice-score delta. The experiment card is intact.

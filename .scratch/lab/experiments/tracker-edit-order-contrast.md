# Contrast ordered edit scripts with count-matched shuffled scripts

Status: needs-human. Prototype implemented; blocked before training calibration and target validation.
All four supplied supervisor requests failed before executing commands.
No fitted model or measured practice result is available.

## Prediction

Tracker −1, overall 0 on lab-v1.5u practice, with listener
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. The target is
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
Correcting the local binding may expose the documented nearby listener wrong-word
false warning. This experiment leaves checker handling unchanged.

The supplied 0.3915-edit disadvantage of the near continuation is prior evidence from
the experiment brief, not a measurement reproduced here. `lab-experiment.json` is unchanged.

## Mechanism

The inspected local matcher retains weighted costs, origins and total edits, but no
ordered edit representation. `packages/core/src/edit-order.ts` adds immutable ordered
scripts of match, substitution, insertion, deletion and word-boundary operations. Its
small recurrent model has five categorical states: the preceding operation is its
hidden state, and 25 learned transition parameters score successive operations.
Transition counts are centered at their exact expectation under uniform permutations
with the same operation counts. The model can distinguish deletion-then-matches from
matches-then-deletion, including their position relative to word boundaries.

`local.ts` carries up to 16 distinct ambiguous minimum-cost scripts beside each
existing reference cell. Their persistent prefixes and recurrent sufficient statistics
survive jumps, recentring and save/restore. A jump retains the earlier observations
and emits the destination boundary; it does not misrepresent its bypassed phones as
independent deletions. Existing movement, duration and surah-entry charges remain in
the original weighted cost.

When the original winner would change run origin, each finite retained candidate gets
the same normalized marginal of script evidence. The correction is subtracted from
its weighted cost at the existing three nats per plain edit. Each candidate's original
acoustic script weights normalize before summation, so extra script routes do not
receive free counting credit. A selected candidate retains its original origin/edit
representative and follows the existing tracker events and commitment rules.

The option `TrackerOptions.localEditOrder` supplies a frozen model and selects
`audit`, `ordered` or `shuffled`. Audit computes the contrast while retaining the
original decisions. No placeholder default has been installed: until actual training
produces a calibration, the normal engine follows its original behavior.

## Training and control

`packages/sdk/src/edit-order-examples.ts` opens only help-train and the authorized
`models/spliced-clean-int8` listener. Eligibility requires train/none/scored labels,
no exclusion, no events and no repeats. Any speaker appearing in a non-train label
is excluded from all training. Model-hash-keyed decodes are cached in this experiment's
Modal scratch directory.

Complete transcript-constrained k-shortest alignments provide latent word-window
boundaries. One-to-three-word windows are contrasted with different Quran
pronunciations of exactly the same reference-phone length and boundary count.
Positive and competing scripts are then grouped by exactly equal match/substitution/
insertion/deletion/boundary totals. Neither edit totals, reference length nor location
distance can solve these paired contrasts; no reader or verse identity reaches the model.
Ambiguous scripts within each matched group receive normalized acoustic weights.

`edit-order-fit.py` fits the 25-parameter recurrent score using marginal pairwise
logistic loss. Its equal-capacity control receives a shared count-preserving shuffle
within each matched pair, so ordering is removed without introducing random
label-correlated differences between the pair's inputs. Both arms use the same
examples, optimization and regularization. An order-blind model cannot distinguish
these exactly matched pairs; its regularized optimum is zero evidence.

Three regularization strengths and five likelihood-ratio scales, including zero,
are compared using speaker-balanced out-of-fold loss over up to five training-speaker
partitions. Selection precedes a single final refit on authorized training speakers.
The output contains the frozen model and a provenance/selection report. No training
examples, fitted parameters or report have been returned by Modal in this attempt.

## Validation

Nine focused test bodies passed with Node 22 TypeScript stripping and an assertion
adapter for Vitest imports. This establishes numerical and synthetic behavior; it is
not a Vitest suite pass or a TypeScript build. The tests cover equal-count ordering
contrasts, shuffled invariance, exact permutation centering, normalized ambiguous
scripts, directed consumption, boundaries, malformed calibration, audit/zero-model
equivalence, common observation counts, model ownership, rewind and speaker exclusion.
The Python estimator passes syntax parsing. `git diff --check` passes.

Supervisor requests:

| Run | Work requested | Result |
| --- | --- | --- |
| 1 | Authorized help-train schema/model inspection | Pre-execution `NotFoundError: No such file or directory` |
| 2 | Full core/SDK tests on six shards | Same pre-execution failure |
| 3 | Core/SDK build, training examples and frozen fit | Same pre-execution failure |
| 4 | Final full core/SDK tests on six shards | Same pre-execution failure |

No practice score was requested (0 of 3). Target preference reversal, target warnings,
target wins, component/overall changes, studio behavior and clean-session warnings
remain unmeasured. Runner failure is not evidence for a premise kill.

## Limits and recovery

This is a first-order categorical recurrent model, not an unrestricted neural sequence
model. Inference marginalizes at most 16 minimum-cost scripts at each existing cell;
training uses at most 16 k-shortest scripts and transcript-constrained latent crops.
More expensive histories discarded by the original recurrence are not recovered.
The original cost-optimal origin/edit representative is retained within each cell.
Evidence begins when local search starts, after initial global locking. The adequacy
of the matched training pool and runtime/memory on real long sessions are unmeasured.

Restore the supplied runner, then run the training workflow without opening practice:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 30 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && EDIT_ORDER_CACHE=/scratch/tracker-edit-order-contrast/clean-decodes pnpm --filter @tilawa/sdk exec tsx src/edit-order-examples.ts && python3 packages/sdk/src/edit-order-fit.py --examples "$LAB_OUT/edit-order-examples.json" --out "$LAB_OUT"'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- \
  'bash tools/autolab_tests.sh'
```

Copy the returned frozen model into this worktree. The read-only replay aid accepts
the case and comparison locations as arguments, runs audit/ordered/shuffled with the
original checker, and writes all nearby candidate contrasts and confirmed events:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run -- \
  'pnpm --filter @tilawa/core exec tsx /app/.scratch/lab/edit-order-target-audit.ts --set lab-v1.5u --session lab-v1.5-p46 --at 121.84 --near 54:36:5 --restart 54:37:9 --model /app/.scratch/lab/edit-order-model.json'
```

Inspect the common observation count and the near-minus-restart differences under
both frozen arms before claiming an order-specific reversal. Then inspect the actual
target warning and the explicitly accepted nearby listener risk. Apply the user's
kill rule only to an observed failed premise or fewer than three measured target
wins; if enough targets are won, install the frozen default and request a practice
score before gate-dependent narrowing. This branch has not reached that decision.

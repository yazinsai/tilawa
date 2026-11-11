# Padding-invariant phone-head loss

Status: killed on the premise; padding already contributes zero direct
phone-head training gradient. Experiment card unchanged.

## Prediction and mechanism

Predict one recovered addition with no counted losses: listener -1 and overall
-1. The proposed cause is artificial blank supervision from batch-padding
positions, disproportionately training silence in batches with short recordings.
The planned intervention freezes the encoder and refits its existing phone head
on cached features, excluding artificial padding from distillation and auxiliary
frame losses while retaining the existing weights of real frames and recordings.
An identical-data, identical-update-count original-loss refit is the control;
checkpoint selection would use training-speaker partitions only.

The card stops if padding already contributes zero training gradient. That
premise is checked before collecting data, refitting a head or exporting a model.

## Source audit

The current training path already implements the proposed validity contract:

- `modal/student_train.py:266` collates feature lengths separately from padded
  feature tensors, and teacher lengths separately from padded teacher tensors.
- The student's forward pass propagates those feature lengths through acoustic
  subsampling and passes a padding mask to the encoder. It returns per-item
  `out_lens` alongside the phone log probabilities.
- `modal/student_train.py:315` divides summed transcript CTC by
  `out_lens.sum()`, and passes `out_lens` as CTC input lengths. Batch-padding
  positions therefore participate in neither CTC paths nor the divisor.
- `modal/student_train.py:326` builds `valid[b,t] = t < out_lens[b]`.
  `tools/student.py:77` multiplies teacher cross-entropy by this mask before
  summing and divides by `valid.sum()`, never the padded tensor extent.
- Teacher frame gathering clamps to each recording's last teacher frame.
  Although this can repeat a teacher row over trailing positions, padded
  student rows are masked before reduction. Repetition within the valid acoustic
  extent is retained and is a different supervision question.
- The supplied training path has only CTC and this optional distillation loss;
  no separate auxiliary frame loss adds padding supervision.

For finite outputs, the distillation expression is
`sum(valid * cross_entropy) / sum(valid)`. Every padded position has zero
derivative with respect to its phone logits. CTC likewise has no derivative
past its input length. A frozen-encoder head refit applying the proposed mask
thus has the same objective and gradients as the original computation.

## Validation and limits

The existing unit test at `tools/test_student.py:75` already checks that teacher
cross-entropy is normalized over valid frames and excludes a padded row.

A disposable diagnostic was prepared to evaluate the actual `collate` and `losses` functions
extracted from the unchanged training source, with the actual `soft_target_loss`
helper. Its frozen-feature linear head uses two recordings with valid output
lengths 8 and 13 and teacher lengths 6 and 16. It compares output widths 13 and
29, changes only artificial padded feature values, and separately differentiates
CTC, KD and their weighted sum at teacher delays 0, 2 and -1. It checks padding
logit gradients, padding feature gradients, valid-frame gradients, head-gradient
and loss invariance, and independently verifies the frame/recording reduction
against individually trimmed recordings. A real blank-supervised frame remains
inside the valid interval so real silence is retained.

Eight local source/AST contract checks passed: preservation of feature and
teacher lengths, CTC input lengths and valid-frame divisor, KD output mask and
its use at the helper call, masking before reduction, and the absence of any
additional frame-loss term. This is source validation and an algebraic proof,
not an observed autograd measurement. The audited source SHA-256 values are:

- `modal/student_train.py`:
  `e5b2dc0d99880b686f562a80d9b28714ba2dd086609afc579cac02b67a5cb448`
- `tools/student.py`:
  `ec1d9e9acc4c09c3ec553c4c6cb16a76dbc9f7ffb86b9fc8501a541cde601ce9`

Both authorized remote validation requests failed at the execution-service
boundary with `NotFoundError: No such file or directory`:

- Modal run 1 requested the numerical audit on the training image; the command
  did not execute and produced no gradient measurements.
- Modal run 2 requested
  `modal-run --shards 6 --timeout 30 -- 'bash tools/autolab_tests.sh'`;
  neither the build nor core/SDK test suites ran, so their pass status is unknown.

The requests, error results and prepared diagnostic are retained under ignored
`.autolab/` files. No new unit test is committed because no training or engine
behavior is changed; the existing valid-frame test is left intact.

This audit isolates direct phone-head supervision on fixed encoder features.
It does not claim general encoder batch invariance, recover any planted
consonant, or measure a practice score. No target audio is used for training or
selection. The source proof confirms the card's zero-padding-gradient stop
condition, independently of the unavailable numerical audit. No refit,
checkpoint selection, model export or practice score was performed. Targets
won: 0; targets evaluated: 0. This does not claim measured zero practice gain.
No engine,
scorer, lab tool, test-data, manifest or evidence change has been made.

## Decision

Stop on the false supervision premise. Adding another validity representation
or refitting the head would not remove any artificial padding gradient, because
the current loss already removes it. There is no engine patch to revert and no
model bundle to declare. The write-up is the only committed change; the killed
outcome is recorded in ignored `.autolab/outcome.json`. No fold was performed.

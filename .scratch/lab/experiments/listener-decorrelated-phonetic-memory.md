# Decorrelated phonetic memory

Status: implementation and local mathematical tests complete; remote training,
model packaging and the acoustic premise remain unvalidated. The experiment
card is unchanged.

## Prediction and mechanism

The card predicts two recovered additions and one collateral wrong call:
listener −1 and overall −1 on lab-v1.5u. Its proposed failure is redundancy in
upper acoustic features, which might preserve vowels or nasals while losing
weak consonant distinctions. This is a hypothesis, not an established finding.

The student exposes its existing pre-CTC states through `forward_features`,
without changing the ordinary forward result, parameter names, alphabet or
streaming export. The experiment freezes the acoustic stem and first three
encoder stacks and jointly trains the remaining stacks and existing phone head.

Detached CTC forward/backward computes transcript-conditioned occupancy for
each target occurrence. Distinct occurrences have distinct states even when
their phone identities are equal. The canonical index's word offsets separate
duration-expanded runs within a word from equal phones across separate words.
Ambiguous canonical occurrence layouts are excluded instead of guessed. Blank
states and padded frames never enter occurrence pooling.

Each represented phone class has equal total weight in the feature statistics.
Within-class occurrence sampling is capped and seeded. Per-dimension scales are
calibrated from fit-speaker recordings only. A variance floor and off-diagonal
covariance loss act on the pooled pre-CTC embeddings; their alignment weights
are detached so the model must change its representation rather than its pooling.

The dedicated trainer retains the checkpoint's transcript CTC and cached KD
configuration, the shipped augmentation chain, and a frozen shipped-output KL
preservation term. The help none-labelled recordings have no cached teacher
posteriors; both arms use supervised CTC and shipped-output preservation there.
No mistake-labelled help takes enter training.

The continuation control receives exactly the same recordings, augmented
batches, dropout draws, occurrence samples, variance floor, ordinary objective,
trainable parameters and update count. It also computes covariance, with a zero
loss multiplier. Positive covariance weights 0.01 and 0.1 are compared only on
new development groups drawn from training speakers. The selected treatment is
paired with the control checkpoint at the same update count. A separate
training-speaker confirmation group is evaluated once after selection and cannot
change the selected model.

The loader requires tlog voice-cluster provenance and help speaker provenance.
It excludes whole clusters containing a gold-excluded recording or a
non-training row, rather than excluding only one recording. Help records must
explicitly belong to the training split. The existing tlog clusters are
approximate speaker groups; they cannot prove perfect identity separation.
The actual mounted data and checkpoint provenance have not yet been audited.

The export path retains the base bundle's alphabet, 40 ms output clock, feature
configuration, window, shift, context and state tensors. It compares actual
PyTorch streaming output with fp32 ONNX on training-only confirmation clips and
records int8 agreement. It verifies and copies the unchanged auxiliary insertion
teacher and external weights before writing a completion manifest. No tracker,
checker or insertion-verification rule is changed.

## Validation and limits

Twelve local unit tests passed using workspace-local NumPy and PyTorch. They
check occurrence forward/backward against exhaustive path enumeration, including
consecutive equal labels; impossible transcripts; blank exclusion; deterministic
class sampling; explicit duration merging; preservation of distinct
articulations; real-frame slicing; class-balanced statistics; feature gradients;
variance-floor behavior; and the control/treatment covariance-loss difference.
Python compilation and `git diff --check` also passed.

The canonical parser was additionally exercised on three ordinary ayahs from
the existing index, with matching span and passage results. The auxiliary asset
declarations resolve to `teacher.onnx` and `teacher.onnx.data`. These source and
mathematical checks do not validate model training, ONNX export, speaker
provenance in the mount, consonant recovery or warning precision.

Supervisor run 1 (read-only inventory) and run 3 (the required six-shard
core/SDK test harness) returned `NotFoundError: No such file or directory`
before returning command output. Run 2 requests a training-image inventory;
run 4 requests the new unit tests and strict training-data audit. Their results
were pending when this initial record was written. These infrastructure errors
do not establish a scientific kill or a zero-gain result.

No training-selected checkpoint or complete replacement bundle exists yet.
No `.autolab/model.json` is declared, and no practice score has been requested.
The main-model fallback would not evaluate this experiment. Core and SDK tests
are not reported as passing: their remote request failed before execution.

## Required acoustic check

After training-only selection and complete export, compare the treatment with
its matched control on the six registered phone additions: p07, p33, p48, p56,
p57 and p79. Count correctly ordered restorations of previously missing planted
consonants separately from actual confirmed warning recoveries. Target labels
may select diagnostics but must never select regularization or checkpoints.

The card's acoustic criterion is at least two restorations relative to control.
The task's final rule also requires at least three recovered target warnings to
continue to a practice score. Neither count is available. If either required
premise fails after building and measuring the model, revert the engine changes
and record the scientific kill. If three or more warnings are recovered, request
one practice score and leave loss accounting to the fold gate. Ordinary
collateral losses alone do not justify stopping before that required score.

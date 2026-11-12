# Teacher tail as censored supervision

Status: implementation and synthetic validation complete; real training and
export blocked by the supervisor service. The experiment card is unchanged. No model has been declared for
scoring and no practice score has been consumed.

## Prediction and mechanism

The card predicts listener −1 and overall −1: one recovered addition with no
losses counted. This is below the fold gate's minimum improvement. The final
implementation instructions require at least three target wins before practice
scoring; that requirement takes precedence over the card's two-stream pilot
threshold. Collateral losses are judged by the fold gate, not an acoustic kill.

The current `tools/student.py:soft_target_loss` divides the recorded top-five
probabilities by their retained sum and applies cross-entropy over that set.
`modal/student_train.py:losses` calls it after zero-delay, clamped 25 Hz mapping.
The cached generator stores fp16 softmax top-five entries, rather than a full
teacher distribution. Absolute retained sums exist in that source representation;
this experiment deliberately uses only the conditional probabilities consumed
by the existing objective. It does not claim that those raw sums are unavailable.
No full teacher targets are regenerated and no missing phone probabilities are
claimed to be recoverable.

Let S be the K retained classes, q their normalized probabilities, V the
inventory size, m the smallest q, and alpha the unknown retained mass. Admissible
teacher distributions obey:

- t_i = alpha q_i for i in S;
- 0 ≤ t_j ≤ alpha m for j outside S, by the recorded top-k ordering;
- sum(t) = 1, hence 1 / (1 + (V−K)m) ≤ alpha ≤ 1.

`censored_soft_target_loss` uses the minimum KL(t || student) over this admissible
set. This is a minimum-relative-entropy, set-valued loss: it profiles the unknown
retained mass and pools unrestricted omitted classes through their student mass.
It is not a Bayesian posterior over missing teacher probabilities. Within the
allowed tail, excluded classes retain their student ratios. When retained ratios
and ordering already agree with the observation, both loss and gradient are
zero even with nonzero omitted probability. Ordering violations remain penalized.
Full-support observations have the ordinary KD gradient. A recorded zero
smallest entry forces the allowed tail to zero; fp16 rounding therefore limits
what the objective can recover.

An analytic active-set projection sorts the omitted student probabilities,
caps those that would outrank the smallest retained class, and solves the
remaining normalization with a weighted geometric mean in log space. The
completion is detached and the loss differentiates through the student only,
using the envelope theorem. This avoids training toward an arbitrarily chosen
uniform tail or pushing every excluded phone to zero.

The original trainer exposes `--kd-target censored`, with renormalized supervision
remaining its default. `modal/student_censored_train.py` supplies the bounded
paired experiment. It reuses frozen streaming feature capture, source enrollment
and classifier export from `exp/listener-noise-aware-teacher-loss`, without that
experiment's channel estimation or corrected objective.

## Training and export contract

Both arms start from the same shipped head and receive identical cached encoder
features, recordings, augmentation views, batches, clipping and update counts.
Only the KD loss differs. Transcript CTC, shipped-output KL preservation and
conservative real-silence blank supervision remain common. No decoder, tracker,
checker, inventory, timestamp, frontend or insertion-verification change is made.

The fixed recipe admits up to 512 authorized original clean-match recordings
and help-train training speakers' none-labelled takes, at most 30 seconds each.
Original gold/excluded and nontraining speaker clusters are removed in full.
Help nontraining/excluded speakers are also removed in full; mistakes, repeats,
unsupported and unscored content do not receive canonical transcripts. Exact
waveform duplicates across different speakers abort preparation.

Speakers are reserved separately within each source by stable hash before
acoustic comparisons. Each recording has an ordinary view and two views from
the shipped RIR, noise, gain and SpecAugment functions and ranges. Original views
inherit their cached conditional top-five targets with the existing clamped
frame mapping. Help takes contribute CTC and output preservation only, because
there are no existing sparse teacher targets for them. Silence labels require
a real 120 ms neighborhood below one 16-bit quantization step; teacher blanks,
CTC alignment and padded frames do not define silence. Other pauses retain
ordinary-output preservation.

Twelve paired epochs, batch size eight, KD weight two (the shipped recipe),
and the fixed common optimizer settings
are established before target queries. The censored checkpoint is selected by
speaker-averaged ordinary CTC plus pause and shipped-output preservation loss
on reserved training speakers. Selection includes exported weight rounding and
the initial checkpoint. The control is retained at the same epoch/update count.
Input, cache, partition, initial-head and selection hashes are recorded.

Export replaces only existing classifier constants, preserving graph nodes,
encoder tensors and streaming states. Feature capture uses the rounded
activations consumed by the shipped integer head and trims final drain padding.
Export requires every registered inherited teacher asset and external weight
file to exist and match its checksum; both new bundles preserve those files.
Generated manifests receive the new model checksum and byte count. No fallback
model declaration is created for an incomplete bundle.

## Validation and limits

Python compilation and whitespace checks pass. An independent NumPy-only
active-set calculation passed 50 random fixtures against a dense feasible-mass
search (maximum KL gap 1.77e−9), with finite-difference gradient error at most
7.07e−10. This independently checks the derivation, not real audio.
Added tests cover feasible tail mass with no suppression gradient, retained ratios, ordering caps, independent
constrained-optimizer agreement, finite-difference envelope gradients,
full-support equivalence, zero retained entries, invariance to absolute retained
mass, padding, invalid observations and extreme log probabilities. Synthetic
training tests cover label filtering, sparse frame mapping, real-silence masks,
frozen cache hashes, speaker separation, matched update counts, training-only
checkpoint selection and original trainer loss routing. Reused ONNX tests cover
float/int8 feature reconstruction, per-channel quantization, head-only graph
changes, unchanged state, final padding and complete auxiliary bundles.

All 34 tests passed locally, with no skips, in 207.05 seconds:

```sh
.autolab/censored-test-env/bin/python -m pytest -q \
  --basetemp=.autolab/pytest-tmp tools/test_student.py \
  tools/test_student_censored.py modal/test_student_censored_train.py \
  modal/test_student_head_refit.py
```

The isolated environment used Python 3.13.7, Torch 2.8.0, NumPy 2.5.4,
SciPy 1.18.1, ONNX 1.19.0 and ONNX Runtime 1.22.1. One harmless test-fixture
NumPy-to-tensor performance warning was emitted. The head-fit test uses synthetic
12-frame arrays; no real recitation inference or training ran on the Mac.

Four authorized supervisor requests failed before container execution with
`NotFoundError: No such file or directory`, each reporting USD 0.00:

1. Read-only data/dependency inventory, normal image, 10-minute limit.
2. Supplied six-shard core/SDK build and test harness, 30-minute limit.
3. Numerical suite followed by prepare, paired refit and complete export,
   training image, 60-minute limit.
4. Latest numerical suite and read-only asset/dependency inventory retry,
   training image, 10-minute limit.

No remote command logs or input inventories were returned. Core and SDK tests
therefore have not run successfully. These failures do not establish missing
data or assets and are not an acoustic premise failure. Further real execution
requires the supervisor service to be restored.

No real feature cache, selected trained head, verified bundle, full-session target
comparison or practice result is available. Target wins and listener/overall
changes remain unknown, not zero. Ordinary-reading preservation is a control,
not a guarantee of unchanged warnings. Real source schemas, speaker coverage,
streaming reconstruction tolerance and preparation runtime remain unverified.

## Continuation

Run the numerical suite and then the following fail-fast pipeline through the
authorized supervisor's training image:

```sh
python3 -m pytest -q tools/test_student.py tools/test_student_censored.py \
  modal/test_student_censored_train.py modal/test_student_head_refit.py
python3 modal/student_censored_train.py prepare \
  --model /lab/data/models/spliced-clean-int8 \
  --out /scratch/listener-teacher-tail-censored-loss/refit-v1
python3 modal/student_censored_train.py fit \
  --out /scratch/listener-teacher-tail-censored-loss/refit-v1
python3 modal/student_censored_train.py export \
  --model /lab/data/models/spliced-clean-int8 \
  --out /scratch/listener-teacher-tail-censored-loss/refit-v1
```

After both bundles pass verification, compare their identical full-session SDK
streams on p07, p33, p48, p56 and p57. Target labels only select/report cases;
they never select training examples, checkpoints or inference candidates. Check
whether a previously missing planted consonant returns in order relative to the
matched control and whether the unchanged engine confirms the addition. Fewer
than three target wins calls for engine rollback and a measured kill note; at
least three calls for an authorized practice score before deciding or narrowing.

Implementation commit: `5bb5b6e8`. All three practice scores remain available.

Only after complete bundle verification should ignored `.autolab/model.json`
select `refit-v1/censored-bundle`. There is no killed outcome for infrastructure
failure, and a main-model decode would not evaluate this refit. No scorer, lab
tooling, supplied data, evidence, tracked manifest or experiment-card edit was
made. No fold was performed.

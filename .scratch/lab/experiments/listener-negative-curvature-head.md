# Negative-curvature frozen phone-head refit

Status: implemented; training and acoustic validation blocked and unmeasured.
No replacement model has been declared. The experiment card is unchanged.

## Prediction and mechanism

Predict listener −1 and overall −1: one recovered addition with no predicted
losses. This is within the lab noise band and below the fold floor. The premise
is speculative: a complete transcript-conditioned CTC saddle may suppress weak
consonants even after ordinary first-order head fitting becomes stationary.

`modal/student_negative_curvature.py` computes exact Hessian-vector products of
the fully marginalized CTC objective. Its log-space forward-backward recurrence
propagates directional derivatives through both path messages, the transcript
partition function and frame-class occupancies. The head Hessian therefore
includes the transcript-conditioned path covariance. It neither freezes path
assignments nor substitutes a framewise Fisher matrix. Unreachable states have
exactly zero mass; adjacent identical phones require a blank; impossible full
transcripts fail rather than disappearing from training.

The encoder is frozen. All arms start from the existing affine head with its
quantized weights dequantized, then share one complete-corpus L-BFGS warm start
of at most 128 objective evaluations. Removing dynamic activation rounding in
the head is common to every arm and cannot establish a curvature gain. No new
head capacity, pronunciation inventory or decoder constraint is added.

The complete objective is equally weighted by recording, then fixed view:
CTC divided by transcript length, full teacher cross-entropy averaged over real
teacher frames, shipped-output KL averaged over real student frames, and L2
to the initial head with coefficient 0.01. Real speech, pauses and breaths
remain in both preservation terms. These terms limit drift; they do not prove
that a refitted head stays quiet on clean recitation.

Each of six rounds runs a 24-probe fully reorthogonalized Lanczos/Ritz search
and one measured Rayleigh-quotient probe on the complete fitting objective.
Only a measured negative quotient authorizes a curvature direction. Its
eigenvector residual is reported; convergence to the most negative eigenvector
is not assumed. The curvature arm otherwise uses its ordinary gradient.
The first-order control uses its gradient; the random control uses a seeded
unit direction. The controls run the same Hessian probes for diagnostics and
discard their directions, so value/gradient and HVP call counts match exactly.
Breakdowns are padded with complete zero-direction HVP calls.

Every arm evaluates both signs at the same four candidate norms: the common
radius, half, quarter and eighth of it. Radius starts at one percent of the
initial head norm and halves every two rounds. Only actual complete training
objective decrease accepts a candidate. Accepted norms can differ between
arms; attempted norms and evaluation budgets are matched. The first-order arm
uses the warm solver's 1e−5 gradient stationarity test, rather than amplifying
roundoff into a full-radius step. All eight candidates are evaluated even if
an early one improves. No practice incident chooses a direction, radius or
checkpoint.

The initial head, shared warm start and six shared rounds are candidates. One
shared phase minimizes the mean of all three selection objectives, with equal
weight per selection speaker. Selection speakers belong to the authorized
training pool and are disjoint from fitting speakers. The fitting deadline is
at most one day; budget exhaustion is an error, never claimed convergence.

## Data and export

`modal/student_head_data.py` and `modal/student_head_export.py` reuse the native
feature extraction, enrollment and bundle-export contracts from
`exp/listener-alternating-path-head-fit`. The alternating optimizer is not
imported. Preparation requires authorized, verified performed-transcript
JSONL recordings, help-train training-speaker `none` takes, an explicit release
speaker exclusion list and the shipped noise/RIR augmentation banks.

None takes need empty events, non-errors and unsupported lists. Mistake scripts
and repeats are not treated as performed transcripts. Speakers found in any
non-training split in either corpus are excluded globally before enrollment.
The explicit release exclusions apply to both corpora. Audio hashes prevent
duplicates and partition leakage. Cache hashes bind frozen features, teacher
targets and performed transcripts; model and auxiliary hashes bind the source
bundle. Whole speakers are assigned deterministically to fit or selection,
and both partitions must contain both source types.

Each recording has three shared fixed views: original and two with the shipped
waveform augmentation and SpecAugment recipe. Before caching, exposing the
native pre-head features must leave every source output and state byte-equal
across carried-state synthetic chunks. Preparation does not read practice
audio, labels or incident evidence.

The exporter replaces only the existing projection, removes its unused nodes,
and preserves encoder/state operations, ordered alphabet, 40 ms clock,
frontend and streaming geometry. SDK decoding, tracker, checker and insertion
verification are unchanged. All source auxiliary files, including the SDK
teacher and external weights, are copied and checksum-verified. Every selected
arm must reproduce its fitted head on every selection view within 2e−4
probability tolerance. The training report is written only after all three
complete bundles pass. Existing source manifests are never rewritten.

The supervisor entry point is `modal/student_negative_curvature_train.py`:
`prepare --model --help-labels --transcribed --exclusions --augmentation --out`, then
`fit --prepared --out --seconds`. Persistent outputs belong under
`/scratch/listener-negative-curvature-head`. Missing training enrollment or
teacher assets fail explicitly, with no fallback to the practice set.

## Validation and limits

All 20 generated-fixture unit tests passed, with no skips:
`.scratch/negative-curvature-venv/bin/python -m unittest discover -s modal -p
test_student_negative_curvature.py -v`. They ran only small synthetic examples
on the Mac, with NumPy 2.5.4, SciPy 1.18.1, ONNX 1.23.2 and ONNX Runtime 1.31.1.
Python compilation and whitespace checks pass.

Tests compare transcript mass, occupancies and their derivatives with exhaustive
CTC paths, then compare the complete head gradient and HVP with finite
differences and Hessian symmetry. They cover repeated labels, exact blanks,
peaky long lattices, equal recording/view weights, unchanged caches, deadlines,
matched control calls and candidate norms, speaker exclusion, native float and
quantized head extraction, encoder/state preservation, complete teacher bundle
copying and unchanged export contracts.

A six-frame, two-class generated example has a stationary complete objective
with both preservation terms retained and a negative weight-direction HVP
(approximately −0.19835 for the tested coordinate). The curvature arm accepts
a decreasing step while the stationary first-order arm stays put. This proves
that the built solver can perform the specified escape. It does not establish
negative curvature at the real shipped head, recover any planted consonant,
or measure ordinary-reading safety.

All three supervisor requests failed before command execution with
`NotFoundError: No such file or directory`, reporting USD 0.00. Run 1 requested
runtime/data inspection; run 2 requested the required six-shard core/SDK suites
through `bash tools/autolab_tests.sh`. Run 3 retries supplied bundle/data
inspection and the synthetic Python suite; it failed with the same error and
also reported USD 0.00.

No real source bundle or training audio has been inspected, no real cache or
head has been fitted, no complete real model bundle has been exported, and no
full-session target comparison has run. There have been zero practice-score
invocations. Target wins, losses, listener change and overall change are all
unknown. Service failure neither confirms nor falsifies the acoustic premise;
there is no acoustic kill result.

Once the supervisor can execute, prepare the shared authorized cache, select
all three heads using training speakers only, and verify the exported bundles.
Then compare full-session curvature and matched first-order streams for the
six registered additions, checking previously missing planted consonants and
the unchanged engine warnings. The card's zero-recovery kill condition and the
user's stricter minimum of three target wins remain unevaluated. If fewer than
three targets are won or the premise is wrong, revert the engine changes and
record the measured kill. If at least three are won, run the supplied practice
score before deciding; the fold gate owns collateral-loss decisions.

No `.autolab/model.json` is present. Decoding the shipped model would not
evaluate this implementation. No scorer, lab tooling, supplied manifest, test
data, evidence or experiment-card changes were made, and no fold was performed.

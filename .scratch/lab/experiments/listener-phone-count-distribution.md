# Listener phone occurrence-count distributions

Status: training implementation built; acoustic premise, training and target
evaluation blocked by supervisor execution failures. No replacement model is
declared. The experiment card is unchanged.

## Prediction and premise

Predict listener −1 and overall −1 wrong calls: one recovered added word and
zero losses. This is within the noise band and below the standalone fold floor.

The premise is that a missing consonant can have an unconditional expected
CTC-collapsed count closer to one than zero while assigning more probability
to zero occurrences than exactly one. A mean can conceal missing/surplus
probability mass. The initial kill criterion needs this pattern on at least
two of the six targets, using the planted word's local audio interval.

The targets are p07 at 144.12 s, p33 at 156.16 s, p48 at 39.98 s, p56 at
106.76 s, and p57 at 95.26 and 112.26 s. They appear only in this report and
untracked read-only analysis inputs, never in engine or training code.

## Built mechanism

`modal/student_count_distribution.py` implements a log-domain count-augmented
automaton for each consonant. Its states are previous-frame-is-this-phone and
previous-frame-is-another-label, crossed with occurrence count. Blank is an
ordinary different preceding label. Entering the consonant increments count;
continuing the same run does not. A blank-separated or different-label-
separated repetition increments again. The implicit left-edge label is blank.

The returned distribution contains exact probabilities for counts zero through
a supplied cap and one saturated overflow bin containing every larger count.
Overflow retains both preceding-label states, so subsequent frames cannot
lose or double-count its mass. The automaton sums unrestricted independent CTC
frame paths; it never conditions on the transcript alignment. Log-domain
arithmetic avoids underflow. Unreachable-state log additions mask both operands
to avoid undefined gradients through `logaddexp(-inf, -inf)`.

The auxiliary loss is the mean negative log probability of the transcript's
actual count across consonants, including absent consonants. The cap is the
largest transcript consonant count, so every supervised bin is exact; surplus
counts remain represented by overflow. Performed transcripts retain genuine
gemination. Help reference notation uses the existing SDK held-sound conversion:
madd/ikhfa duration runs become one occurrence and duration-expanded noon/meem
gemination becomes two. Expected counts are calculated separately with
`Σ p[t,c] × (1 − p[t−1,c])`, with implicit initial probability zero, rather
than approximating overflow by its minimum count.

`modal/student_count_train.py` prepares cached encoder features and fits three
objective families: ordinary transcript CTC, CTC plus expected-count error, and
CTC plus count-distribution negative log probability. All retain the shipped
output KL preservation term and frame-normalized transcript CTC. Impossible
enrolled transcripts fail explicitly. The expected-count control uses squared
error divided by actual count plus one, averaged over the same consonants.

The encoder remains the shipped quantized streaming ONNX encoder. Only its
existing affine phone head is refitted. Native head weights are dequantized;
head activation quantization is removed identically in all three arms. This
common export difference cannot be credited to the distribution objective.
No capacity, alphabet, timing, decoder, tracker, checker or insertion rule is
changed, and no count constraint is imposed at inference.

All arms share initialization, recordings, three fixed views per recording
(original plus two shipped augmentation views), seeded update order, SGD at
0.02, gradient bound 5, and twelve epochs. Auxiliary strength candidates are
0.1, 1 and 10 for each auxiliary family. Every arm evaluates both auxiliary
statistics and their zero-weighted gradients, matching computational work.
Regularization is selected by ordinary CTC plus shipped KL on selection
speakers, without including either count auxiliary in the selection metric.
The shared checkpoint minimizes the mean of ordinary and the two selected
auxiliary arms' validation objectives. Epoch zero is eligible. The same
selected and total update counts apply to all three exported arms. The refit
raises on exceeding one day; actual runtime is unmeasured.

## Data and export contracts

The preparation code is reused from `exp/listener-unconditional-count-training`;
that predecessor was infrastructure-blocked too, so its training was not
assumed successful. It requires help-train training-speaker none takes,
authorized verified performed transcripts, recorded real pauses with empty
transcripts, explicit speaker exclusions, and the shipped augmentation banks.

Help none takes must have empty mistake, non-error and unsupported arrays.
Additional recordings require speaker, train split, authorization, verification,
CTC-articulation transcript format, audio path and SHA-256. Silence must be
recorded rather than synthesized. Speakers in any nontraining split across
either corpus, plus release exclusions, are excluded before enrollment.
Whole speakers are assigned by hash to fit or selection. Both partitions must
contain both speech sources and real pauses. Duplicate audio hashes and changed
cached views are rejected. These input contracts have not been verified against
the actual remote corpus because inspection never executed. Training does not
fall back to practice recordings or planted transcripts when inputs are absent.

`modal/student_count_export.py` exposes the native feature tensor without
changing the encoder or state outputs and grafts the selected affine head into
a separate bundle. It inherits the alphabet, clock, frontend and streaming
geometry. Existing insertion-teacher and external-weight files must be present,
copied and hash-verified. Preparation checks unchanged original outputs and
recurrent states over multiple windows. All three complete exports must match
their fitted heads on every selection view within 2e-4 probability tolerance.
The final training report is written only after all three bundles pass parity.
Generated artifact manifests describe their new graphs; existing manifests,
evidence and test data are never edited.

The runner provides `prepare --model --help --transcribed --exclusions
--augmentation --out` and `fit --prepared --out`. Cache and output paths must
be supplied under `/scratch/listener-phone-count-distribution` using the
authorized supervisor. No `.autolab/model.json` has been written: the required
selected real model bundle does not yet exist.

## Validation and limits

Python compilation and whitespace checks pass. The synthetic tests cover full
CTC path enumeration, zero and overflow mass, true repeats versus held runs,
exact first moments, equal means with different disappearance probability,
enumerated and finite-difference gradients, pause suppression, matching arm
initialization/order/selected updates, speaker exclusions, frozen feature
gradients, native float/int8 head extraction, state preservation, and complete
teacher/external-data export.

All 28 synthetic tests passed locally with no skips in 9.679 seconds:
`.autolab/count-venv313/bin/python -m unittest discover -s modal -p
'test_student_count*.py' -v`. The isolated workspace environment uses Python
3.13.7, Torch 2.8.0, ONNX 1.19.0, ONNX Runtime 1.22.1 and NumPy 2.5.4.
No audio decoding or training computation ran on the Mac. These tests
establish numerical and synthetic export contracts, not acoustic transfer.

Five supervisor requests failed before command execution with the same
`NotFoundError: No such file or directory`; every response reported USD 0.00:

1. Supplied data/model/training-root inspection on the default image.
2. Required six-shard core/SDK build and test suites.
3. Count-distribution numerical tests on the training image.
4. Supplied data/model/training-root inspection on the training image.
5. Updated complete count/preparation/export numerical suite on the training image.

There are no remotely inspected posteriors or training records, no measured
premise census, no prepared real feature cache, no trained control or
distribution head, and no verified replacement model bundle. Target wins,
collateral losses, studio change, and listener/overall score changes are
unknown. Practice scoring was invoked zero times. This is an infrastructure
blocker, not zero target wins or a falsified acoustic premise, so a killed
outcome would misrepresent the result.

## Required continuation

Restore supervisor execution and inspect the supplied data. Measure complete
local collapsed-count distributions for truly missing planted consonants.
Stored top-k posteriors require explicit accounting for unlisted mass; the
prepared untracked census uses the SDK's uniform residual policy, which is
an approximation to native logits and cannot establish a borderline premise
without full-logit confirmation. If fewer than two targets satisfy the premise,
revert the engine changes and record an acoustic kill.

Otherwise enroll only authorized training material, prepare the shared cache,
fit/select all three arms within speaker-grouped training partitions, and
verify every real bundle. Compare the selected distribution head with both
controls on the six targets using full-session decoding and unchanged engine
warnings. Fewer than three target wins requires rollback and a measured kill.
At least three wins requires the supplied practice score before deciding or
narrowing. Collateral losses are for the fold gate. Scoring the main listener
without a declared complete replacement bundle would not evaluate this change.

# Reversible upper acoustic memory

Status: implementation prepared; training and evaluation blocked by supervisor infrastructure. No trained model is declared, and no premise or kill decision is established. The experiment card is unchanged.

## Prediction

The card predicts listener −2 and overall −2: preserving upper-layer acoustic
state might recover faint consonants in missed additions. Its criterion requires
two target intervals to recover a previously missing planted consonant relative
to a matched residual control. The execution instructions additionally require
three target wins before proceeding to practice scoring. Neither requirement has
been measured. No practice score has been used.

## Implementation

`modal/student_reversible.py` replaces the native stacks in the last two
Zipformer stages with additive coupling blocks. Split the entire channel vector
into equal halves and compute `y1 = x1 + F(x2)`, `y2 = x2 + G(y1)`. The inverse
is `x2 = y2 - G(y1)`, `x1 = y1 - F(x2)`, given the incoming causal caches.
Layer normalization, expansion, three causal depthwise convolutions and the
output projection all live inside each coupling function. Neither half of the
preserved state is normalized, projected to fewer channels or feature-masked.
Padding and feature masks act only on the additive updates.

The matched ordinary residual control computes both updates in parallel:
`y1 = x1 + F(x2)`, `y2 = x2 + G(x1)`. It has exactly the same parameters,
initialization, convolution operations and caches, but can collapse distinct
inputs. Both replacement arms start at identity through zero-initialized output
projections. This comparison tests coupling versus an ordinary residual
transformation; it does not compare parameter capacity against the original
attention-based stacks.

The reversible boundary is each replacement block at its native temporal
resolution. The shipped convolutional stem, first four encoder stages, stage
down/up-sampling, final output downsampling and CTC head remain in place. Those
operations are not claimed to be invertible. Each block has six real causal
caches, fitting existing Zipformer state routing. The ONNX manifest describes
their new shapes through the SDK's existing generic state contract. The phone
inventory, 80-bin input, streaming window/shift and 40 ms output clock are
unchanged. No tracker, checker, candidate-generation or insertion-verification
rule changes are made.

`student_train.build_model` accepts an optional `upper_blocks` mode and
`load_student` reconstructs it from checkpoint configuration. Existing
checkpoints without that field continue to build the shipped architecture.

## Matched training and export

`modal/student_reversible_train.py` loads clean-match Quran transcription
records exclusively from the supplied training split. It retains speaker-cluster
identity and excludes both listed gold clips and every other clip from an
identified excluded clip's cluster. Learner input admits only training speakers'
none-labelled takes with no events, repeats, unsupported labels or exclusion
flags. Any learner speaker with a nontraining or excluded row is blocked.

A fixed SHA-256 speaker partition provides training and validation subsets;
the two speaker sets must be nonempty and disjoint in both data sources. Learner
takes supplement ordinary training records. Checkpoint selection uses greedy
CTC error on actual cached streams of validation speakers from these training
sources. The program never opens practice sessions or their labels.

Each paired batch shares the same materialized augmented features. The shipped
RIR probability 0.2, noise probability 0.5, 5–25 dB SNR, −10 to +6 dB gain and
SpecAugment chain are retained. Only replacement parameters train. Frozen
modules use evaluation behavior so their stochastic masks cannot change the
shipped stem. Both arms receive identical updates, CTC loss, the checkpoint's
existing teacher top-five KD loss where those targets exist, and weight-5 KL
preservation against the frozen shipped model on the same features. Learner
takes have CTC and preservation loss, with KD explicitly masked because they
lack prepared teacher targets. The pairing checks parameter counts and exact
initial state equality and records partitions and training metrics.

Selected checkpoints export through the existing streaming ONNX wrapper and
MatMul-only dynamic int8 quantization. Export verifies three consecutive warmed
windows on held-back training audio, including fp32 output and all cache parity,
int8 state shapes and finite values. Quantization error is recorded; no int8
recognition-equivalence claim is made. Both `teacher.onnx` and
`teacher.onnx.data` are required, copied unchanged and hash-verified before a
generated manifest is published. Missing assets or invalid parity abort export.

## Validation and remaining work

Python syntax compilation and `git diff --check` pass. With PyTorch 2.8.0 on the
local CPU, all ten focused unit tests pass; one native icefall integration test
is skipped because the pinned icefall environment is unavailable locally. The suite
covers nontrivial inversion with incoming caches and masks, full-versus-streaming
equivalence, gradients through both halves, a singular ordinary residual
counterexample, six-cache routing, padding preservation, deterministic disjoint
speaker partitions, speaker-level exclusions and KD masking on none takes.
A native integration test additionally rebuilds both modes and checks the
unchanged stem, output clock and warmed caches against pinned icefall.

Supervisor runs 1 and 4 (six-shard core/SDK tests), 2 (training-data/checkpoint
probe), and 3 (training-image probe/focused tests) all failed before execution
with `NotFoundError: No such file or directory`. All reported zero GPU spend
and the entire $120 budget remaining. Run 5 requests the focused/native tests
and direct paired trainer on H100, using the conventional
`/student/prior-runs/spliced-clean/ckpt.pt` initialization path. The checkpoint
path itself remains unverified because the inventory probes never executed.

The CPU test dependencies are isolated in an ignored virtual environment.
Native icefall, full export parity, core tests and SDK tests still require the
supervised image. The CPU unit result is not a target-recovery measurement.

The infrastructure error is not an acoustic result. No replacement checkpoint,
complete exported bundle, target recovery count, aggregate score, clean-session
regression or studio regression has been measured. In particular, an untrained
identity replacement cannot establish this experiment's premise. There is no
`.autolab/model.json` selecting an asset, so scoring the main listener would not
evaluate this implementation. No killed outcome is written without a measured
premise failure.

To finish, restore the supplied supervisor, identify the checkpoint corresponding
to the shipped stem from `/student/prior-runs`, run the native and core/SDK tests,
and train/export both arms in a fresh persistent experiment directory using
`student_reversible_train.py --checkpoint ... --base-bundle
/lab/data/models/spliced-clean-int8 --output ...`. Freeze the selected
training-speaker checkpoints before decoding any target. Compare exported
full-session streams and missing planted consonants against the matched residual
control. If the execution floor is met, declare only the verified reversible
bundle in the untracked model selector and run the authorized practice score.
Otherwise apply the user-specified measured kill/rollback procedure.

No scorer, lab tooling, supplied data, supplied manifest or evidence edits are
included, and no fold is requested or performed.

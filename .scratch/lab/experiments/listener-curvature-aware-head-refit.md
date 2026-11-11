# Curvature-aware frozen phone-head refit

Status: implemented; acoustic experiment blocked and unmeasured. No replacement
model has been declared. The experiment card is unchanged.

## Prediction

Listener −1 wrong call and overall −1, with one recovered addition and no
predicted losses. This is within the lab's noise band and below its fold floor.
The acoustic premise is that correlated frozen encoder features leave useful
consonant directions poorly optimized by an ordinary head refit.

## Mechanism

`modal/student_curvature_train.py` prepares one shared cache, then fits ordinary
SGD and curvature-preconditioned arms. Both refit the existing affine CTC phone
projection, initialized from the shipped weights and bias. Quantized projection
weights are dequantized in both arms; the encoder remains frozen and quantized.
There is no residual head, new feature, vocabulary change, or extra parameter
capacity. Removing the head's activation rounding is common to both arms and
is not evidence for curvature by itself.

Each enrolled recording has three fixed views: original and two using the
shipped waveform augmentation and SpecAugment recipe. Each recording receives
one update per epoch and each view has equal weight. Both arms use the same
CTC loss, full-distribution teacher cross-entropy, and KL preservation against
the shipped listener. CTC is normalized by transcript length; KD and
preservation average real frames, including pauses. Targets never change
between arms. The update order is shared and seeded; there are twelve epochs,
with learning rate 0.02 and no momentum or arm-specific clipping.

For augmented features `z = [x, 1]`, the fixed factors are `A = E[zzᵀ]` and
`G = E[diag(p) − ppᵀ]`, estimated from initial-head model probabilities on fit
speakers only. Views are averaged within each recording, then recordings are
averaged equally, so long recordings do not dominate the factor estimate.
The complete weight-and-bias gradient `J` is transformed by solving
`G U A + λU = J` in the factors' eigenspaces. Curvature is normalized by
`c = mean(eigenvalues(A)) × mean(eigenvalues(G))`; the applied direction is
`cU`, and damping is `λ = ratio × c`. This gives ordinary-SGD learning-rate
units and finite updates in singular softmax and correlated-feature directions.
This is a framewise model-Fisher Kronecker approximation, not an exact CTC
Hessian. It omits temporal correlations and feature/output dependence.

Damping candidates 0.1, 1, and 10, plus epoch zero, are specified before practice
analysis. Damping and one shared checkpoint epoch minimize the pair's mean
selection objective, averaged equally across held-out training speakers. The
ordinary control and selected curvature model thus have identical exported
update counts. Selection uses the same complete objective as training, never
practice targets or scores. Curvature is neither estimated nor refreshed on
selection speakers.

## Data separation and export

Preparation requires both authorized, verified performed-transcript recordings
and help-train training-speaker `none` takes. Additional transcripts are
explicitly enrolled JSONL rows with speaker, split, authorization, verification,
phones, WAV path, and its SHA-256. None takes need empty events, non-errors and
unsupported lists. Mistake scripts and repeats are not admitted as performed
transcripts. Speakers appearing in any non-training split across either corpus,
and the explicit release-exclusion list, are excluded globally before fitting.
Audio hashes prevent duplicates across partitions; cache hashes protect the
shared evidence. Whole speakers are assigned deterministically to fit or
selection, and both partitions must contain both source types.

The export replaces the existing projection and prunes its unused old nodes.
Encoder operations, state outputs, ordered alphabet, frontend, streaming
window/shift and 40 ms clock are inherited. SDK decoding, tracker, checker and
insertion verification are unchanged. Every teacher ONNX/external-weight file
is required, copied and checksum-verified. Both exported models must reproduce
their fitted heads on all selection views within 2e-4 probability tolerance;
the final training report is written only after both bundles pass. Feature
extraction must also preserve the source outputs and states exactly before
training starts.

The runner exposes `prepare --model --help --transcribed --exclusions
--augmentation --out` and `fit --prepared --out`. These commands must run
through the supplied supervisor with cache and export directories under
`/scratch/listener-curvature-aware-head-refit`. They do not fall back to
practice audio or guessed transcripts when an enrolled input is missing.

## Validation and current result

All 19 local synthetic tests passed with no skips:
`.scratch/curvature-venv/bin/python -m unittest discover -s modal -p
test_student_curvature.py -v`. The environment uses Torch 2.15.0, ONNX 1.23.2,
ONNX Runtime 1.31.1 and NumPy 2.5.4. These are tiny generated examples, with no
training audio or heavy computation on the Mac.

The tests cover the dense damped Kronecker solve, singular and weak feature
directions, bias cross-terms, invalid curvature, speaker grouping, none-take
exclusion, global cross-corpus holdout exclusion, equal-recording factor
weights, selection-speaker independence, no encoder/teacher gradients, and
identical ordered examples and exported update counts. Native ONNX tests read
float and dynamically quantized heads, replace the projection, preserve state
outputs, and verify complete teacher/external-weight copying and unchanged
manifest contracts. Python compilation and whitespace checks also pass.

Three supervisor requests failed before command execution with
`NotFoundError: No such file or directory`; each reports USD 0.00:

1. Run 1: inspect supplied data, model bundle, persistent scratch and dependencies.
2. Run 2: required six-shard core/SDK suites via `bash tools/autolab_tests.sh`.
3. Run 3: retry data inspection plus the new synthetic unit tests.

No training recordings or source model were inspected remotely. No cache was
prepared, no ordinary or curvature model was fitted, no complete real model
bundle was exported, and no full-session target decode was performed. There
were zero practice-score invocations. Target wins and losses are unknown;
there is no measured component or overall change.

## Limits and next validation

The service failure does not falsify the optimization premise and is not an
acoustic kill. No `.autolab/model.json` is present: evaluating the shipped model
would not evaluate this implementation. Training-schema and real native-graph
compatibility remain unverified until the supervisor can execute.

Once compute is restored, prepare the authorized shared cache, select the
matched heads without practice access, and validate the complete exported
bundles. Then compare full-session exported decodes on the six target spans
against the identical-data ordinary control. The card asks for at least two
spans with a previously missing planted consonant recovered. The user's final
rules also require at least three target wins before practice scoring; if that
floor is reached, run the supplied practice score before narrowing or deciding.
The fold gate owns collateral-loss decisions. Otherwise document an acoustic
kill and revert the engine implementation as instructed. Synthetic solver
correctness alone establishes neither target recovery nor ordinary-reading
safety.

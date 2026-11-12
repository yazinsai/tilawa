# Listener evidence locality refit

Status: implementation present; acoustic training and export blocked, unmeasured.
The experiment card is unchanged. No replacement model is declared.

## Prediction

Listener −1 wrong call and overall −1, with one recovered addition and no
predicted losses. This is within the lab noise band and below the fold floor.
The premise is that an upper acoustic decision about a weak phone depends too
much on distant observations, even when those observations are available and
the streaming state is correct.

## Mechanism

`modal/student_locality.py` freezes the embedding, lower Zipformer stacks and
final output pooling. It refits the last two stacks and the existing phone head.
The shipped lite architecture's last stack already supplies the full output
width, so caching the input to the upper stacks preserves the native dimension
merge. Other dimension layouts fail explicitly instead of dropping bypass
features. The lower cache is obtained from the native full-recording forward
with the installed chunk size and left context. Nothing masks attention or
resets inference state.

For phone occurrence i, the score is log P(full performed transcript | audio)
minus log P(the same transcript with occurrence i deleted | audio). Both scores
use the CTC forward sum. The CTC forward–backward calculation preserves distinct
occurrences of repeated phones and sums over all valid boundaries. Its state
occupancies supply the own-emission probability and the probability of being
inside the occurrence plus its immediate preceding/following phones and
interior blanks. Leading/trailing silence is outside that neighbourhood. The
probabilities map to the lower cache using the unchanged factor-two pooling
clock; there is no hard forced-alignment cut.

Fixed Rademacher probes are scaled by the lower cache's channel RMS. Own probes
use the square root of own-emission probability; remote probes use the square
root of one minus neighbourhood probability. This estimates total sensitivity,
without dividing remote influence by the number of distant frames. Symmetric
perturbations with radius 0.01 estimate directional derivatives. Their CTC sums
use float64 to limit cancellation; gradients into upper parameters remain
ordinary first derivatives, avoiding second derivatives through Icefall's
custom autograd functions.

For own response a, remote response b, and the initial student's fixed own
response a0, the grounding loss is
`(b² + max(0, a0² − a²)) / max(a0², 1e-4)`. Thus flattening every response does
not produce zero loss when the initial own response was nonzero. The anchor
retains only a magnitude of the initial student's own acoustic sensitivity. It
copies neither the teacher's sensitivity nor any remote-context direction.
These are finite sampled directions, not a proof of locality in every direction.

## Matched training and selection

`modal/student_locality_train.py` prepares the identical input views and lower
caches for both arms. Each full recording has its original view and two fixed
views using the shipped RIR/noise/gain and SpecAugment recipe. Both arms use CTC
normalized by output frames plus full-frame KL preservation against the initial
native model, including pause frames. Upper layers run deterministically in
evaluation mode with gradients enabled; augmentation provides input variation.

The ordinary and grounded arms start at identical weights, use the same
recording order, three views, occurrence probes, Adam learning rate 1e-5,
gradient bound 5, and four complete epochs. Each view has up to four fixed
evenly distributed occurrences, rotated across epochs. Both arms execute the
same probe forwards and backwards. Only the grounded arm adds grounding loss,
with weight 0.01. These values were chosen before target analysis and have not
been fitted against practice.

Whole training speakers are deterministically partitioned into fit and
selection groups; every fifth speaker in hashed order is assigned to selection.
There must be at least five speakers and both authorized data sources in each
partition. One shared checkpoint epoch, including epoch zero, minimizes the
pair-average ordinary CTC-plus-preservation objective, averaged equally within
recordings and then speakers. Selection never reads occurrence penalties,
practice labels or target recoveries. The exported arms therefore have the same
number of updates. Preparation and fitting have a combined forty-five-minute
bound, leaving fifteen minutes for export within the supervisor's sixty-minute
job limit and the card's one-day budget. A timeout produces no accepted pair
or declared model, rather than unequal partial arms.

## Data and export contracts

The runner requires a native checkpoint that reproduces the installed int8
model's recurrent outputs and states on eight fixed random windows before
training. It does not attempt to guess upper weights from an ONNX head or train
a different architecture when the checkpoint is missing.

Additional performed transcripts are enrolled JSONL records with speaker,
split=train, authorized=true, verified=true, phones, corpus-relative WAV and
SHA-256. Help-train admission requires a training-speaker, scored `none` take
with empty events, non-errors and unsupported lists. Mistake scripts and
repeats are never treated as performed transcripts. Any speaker appearing in
a non-training split in either corpus, plus the explicit release-exclusion
list, is excluded globally. Audio hashes reject duplicate recordings and
transcript/audio identity mismatches; paths must stay inside their enrolled
corpus. Entire recordings, including their pauses, are retained.

Both selected models use the existing streaming exporter and int8 MatMul
quantization. New manifests inherit the inventory, frontend, window/shift,
40 ms clock and state contract. The SDK's decoding, tracker, checker and
insertion verification code is untouched. All source auxiliary files,
including the teacher ONNX and its external weights, are required, copied and
checksum-verified. Incomplete or truncated teacher weights fail export.

Before a final training report is written, both bundles must load in ONNX
Runtime, pass recurrent quantization checks, and reproduce native recurrent
PyTorch outputs and states on all three views of every selection recording.
Native streaming parity is checked separately from the chunk-masked training
forward. The lower-cache forward must also reproduce the native masked forward
before fitting. No supplied model, manifest, evidence or test data is rewritten.

The runner's required arguments are `--model --checkpoint --help --transcribed
--exclusions --augmentation --out`; invoke it through the supplied supervisor
with output under `/scratch/listener-evidence-locality-refit`. Enrollment paths
and dependencies cannot yet be confirmed because the service has not executed
an inspection request.

## Validation and result

All nineteen synthetic tests passed with no skips:
`.scratch/student/locality/venv/bin/python -m unittest discover -s modal -p
test_student_locality.py -v`. They cover exhaustive CTC path/occupancy agreement, repeated
occurrences, soft neighbourhoods and their clock, deleted-occurrence gradients,
finite differences against autograd, deterministic probe masks, the own-response
floor, exact cached/native toy-forward parity, frozen-prefix gradients,
speaker grouping, matched updates, selection independence from probes,
training-only admission, global holdout exclusion and truncated external teacher
assets. The optimization regression also passed: the penalty reduces remote
dependence while retaining own response on its generated acoustic example.

Local checks use only tiny generated tensors and a generated teacher graph,
with Torch 2.15.0, NumPy 2.5.4, ONNX 1.23.2 and ONNX Runtime 1.31.1 in an ignored
environment. No acoustic training or recording processing runs on the Mac.
Compilation and whitespace checks pass.

Supervisor runs 1, 2 and 4 failed before command execution with
`NotFoundError: No such file or directory`, reporting USD 0.00. They requested
dependency/data inspection, checkpoint/training-corpus inspection, and the new
synthetic tests remotely. Run 3 requests the required six-shard core/SDK suites
and remains pending at this point; no suite result has been returned.

No checkpoint or real training recording was inspected, no acoustic refit was
run, no complete real model bundle was exported, and no target was decoded with
a selected model. There have been zero practice-score invocations. Target wins,
losses, component change and overall change are unknown. Evaluating the main
model would not evaluate this implementation, so `.autolab/model.json` is absent.

## Limits and remaining experiment

The service outage does not falsify the premise and is not an acoustic kill.
Real training-schema/native-checkpoint compatibility, pinned-Icefall gradients,
SDK export parity, ordinary-reading drift and target transfer remain unverified.
The implementation preserves available context; the learned penalty cannot
guarantee that every acoustic decision becomes local or loss-free.

Once the supervisor can execute, inspect the eligible inputs, run the matched
refit, and verify both complete bundles before declaring the grounded export.
Only then compare full-session grounded and ordinary decodes on the five target
spans for missing consonants restored in correct order and resulting warnings.
Follow the user's stopping rule: fewer than three target wins kills the built
change; three or more requires practice scoring before any decision or narrowing.
The fold gate owns collateral-loss tradeoffs. A measured kill requires reverting
the engine changes and recording the actual target count; no zero-win number or
killed outcome is asserted from an infrastructure failure.

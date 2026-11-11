# Retain the scale removed by instantaneous BiasNorm

Status: blocked on supervisor execution. Implementation and tests are committed;
training, exported bundles and target validation are unfinished. The acoustic
premise has not been accepted or killed.

## Preregistered prediction

Listener −1 wrong call, overall −1: two recovered additions and one collateral
wrong call on practice `lab-v1.5u`. The experiment card is unchanged. This
prediction is below the fold gate's minimum gain and is not a measured result.

## Mechanism implemented

The pinned icefall revision is `3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`.
Its BiasNorm multiplies the original activation by a learned output gain divided
by `sqrt(mean((x - bias)^2))`, over the channel axis of each activation. It does
not subtract the bias from the numerator or estimate running statistics.

`modal/student_scale_bypass.py` retains that same biased RMS calculation, takes
its logarithm and projects this scalar back to the site's channel dimension with
a bias-free, zero-initialized linear layer. The residual is added immediately
after normalization, so the following acoustic operation receives both the
normalized pattern and its removed magnitude. The learned output gain stays
separate. The expression preserves the original normalization and its gain
constraint; it uses ordinary differentiable operations instead of icefall's
memory-saving custom backward. This can increase training memory use.

Every BiasNorm site is upgraded while retaining the existing parameter names and
values. Checkpoints record the mode in their configuration, and the ordinary
student loader restores it, and the shared training image includes its module.
The default model constructor is unchanged when no
mode is requested. The constant arm has identical additional parameters but
feeds each projection a scalar one.

The paired trainer uses the same augmented tensors, batch order, Torch seeds and
Python seeds for both arms. It restores the original training schedule's batch
count, fits only projections for one epoch, then permits three epochs of joint
updates. Its fixed objective is CTC plus five times the KL divergence from the
frozen shipped acoustic model on all valid frames, including ordinary pauses.
Original parameters use learning rate `1e-5`, projections `1e-3`. These choices
were made without practice observations. The command refuses to reuse old paired
checkpoints or export a checkpoint selected before joint training.

Data admission is limited to prepared `s1` training/clean-match recordings with
the gold exclusion list applied, and help-train's training-split, none-labelled
takes with no events, repeats, exclusions or unsupported labels. Original clips
use the existing teacher-anchored canonical transcript function; learner takes
use canonical phones. Both sources retain speaker identifiers. A fixed hash of
the speaker assigns a validation partition, and overlapping training/validation
speakers are rejected. Only this partition's PER selects checkpoints. The CLI
uses at most ten hours of original audio, oversamples learners at approximately
one item per four original items, and validates on up to 128 original items plus
the learner validation partition. No practice labels or audio enter this path.

The waveform reverb, noise, gain, Kaldi fbank and SpecAugment operations reuse the
shipped augmentation functions and settings. Export uses the existing streaming
wrapper and dynamic MatMul int8 quantization. All scale-carrying operations stay
inside the primary ONNX graph, with the same inventory, input interface, caches
and 40 ms frame clock. Both export directories must include byte-identical
copies of the existing insertion teacher and its external weights; absent files
fail instead of silently removing verification. Tracker, checker, gap decoding
and insertion acceptance rules have no changes.

## Checks implemented and actual results

The Python unit suite covers zero-initialization output and gradient parity over
multiple channel axes, the exact biased log-RMS residual, scale discrimination
versus the constant control, all-site installation, parameter round trips,
speaker separation, learner admission and exact auxiliary copying. It also
checks an isolated nonzero projection's ONNX output and a small complete acoustic
model's initial streaming outputs and warmed caches. Export checks three
streaming windows against PyTorch, asserts fp32 parity, checks finite int8
outputs and cache shapes, and records int8 probability differences. These are
graph checks, not evidence of recognition quality.

An SDK integration test, enabled by `TILAWA_SCALE_BYPASS_DIR`, loads both exported
arms through `loadNodeModel` and `createListener`, checks primary and auxiliary
hashes, and tests inventory, frame count and invariance to audio push size. It
requires real exported bundles and is skipped when the directory is absent.

Local Python syntax compilation and `git diff --check` passed. No Python runtime
test, SDK runtime test or core runtime test has a successful result. The provided
supervisor service returned `NotFoundError: No such file or directory` on each
attempt, without returning useful shard logs:

| Run | Requested work | Result |
| --- | --- | --- |
| 1 | Training-image inspection of normalization, prior runs and available data | NotFoundError |
| 2 | L4 training-image inspection of pinned BiasNorm and prior checkpoints | NotFoundError |
| 3 | Six-shard `bash tools/autolab_tests.sh` for core/SDK build and tests | NotFoundError |
| 4 | Focused Python normalization tests in the training image | NotFoundError |
| 5 | Plain CPU `python3 -c "print(123, flush=True)"` to distinguish command failure from service failure | NotFoundError |

Run 4 predates the later full-model and data-admission test additions. The last
probe also failed, so no further GPU reservations were made. Reported GPU spend
is $0.13, with $119.87 remaining. No training checkpoint was located or loaded;
no fitting, export or acoustic comparison ran.

## Limits and continuation

The six named target additions remain unmeasured. No recovered-consonant count,
target-win count, listener change, overall change, clean-session effect or studio
effect is available. The card's consonant-restoration kill check and the final
instruction's target-win threshold cannot be applied to infrastructure errors.
No practice score was requested; all three practice-score slots remain unused.

There is no complete model bundle and no `.autolab/model.json`. A replay with the
main listener would not evaluate this implementation. The ignored outcome file
records `needs-human` for the execution blockage, without invented measurements.
The engine changes remain available for continuation rather than being reverted
as an acoustically killed experiment.

After the supervisor service is restored, first run the Python suite and core/SDK
tests, then identify the checkpoint corresponding to the shipped listener and
the complete baseline bundle with its existing teacher. The paired command is:

```sh
python modal/student_scale_bypass.py \
  --checkpoint /student/prior-runs/<shipped-run>/ckpt.pt \
  --base-bundle /lab/data/models/<complete-baseline-bundle> \
  --output /scratch/listener-normalization-scale-bypass/paired
```

Use the authorized supervisor training image and GPU wrapper for that command;
the placeholders above have not been verified. The default command permits
50 minutes of fitting plus validation/export within a 60-minute supervisor run.
If the budget expires before joint training, it fails without declaring a model.
Run the Python suite in the training image with:

```sh
python -m unittest discover -s modal -p 'test_student_scale_bypass.py' -v
```

The SDK bundle check is:

```sh
TILAWA_SCALE_BYPASS_DIR=/scratch/listener-normalization-scale-bypass/paired \
  pnpm --filter @tilawa/sdk exec vitest run test/scale-bypass.test.ts
```

Finish both bundle checks before declaring the scale arm. Compare matched full
phone sessions for shipped, constant and scale arms through the ordinary SDK,
including existing gap decoding and insertion verification. Inspect restored
planted consonants in text order and actual warnings at the target keys. Select
no checkpoint or numeric setting from that comparison. Apply the user's target
threshold and practice-score instruction only once these observations exist.

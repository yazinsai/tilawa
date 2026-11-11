# Mask artificial training tails before temporal mixing

Status: ready-for-human

Implementation is prepared, but supervisor execution is blocked. No refit,
complete model bundle or acoustic comparison exists. The acoustic premise has
not been killed or accepted.

## Prediction and stop rules

The unchanged card predicts listener −1 and overall −1 wrong call on practice
`lab-v1.5u`, with no losses. This forecast is below the fold gate's minimum gain;
it is a prediction, not a measurement.

The card asks whether at least two named additions regain a previously missing
planted consonant in full-session exported decoding relative to an identical-data
continuation. The final task instructions additionally require three target
warning wins to proceed to practice scoring. Infrastructure errors supply
neither observation. No practice score has been requested; all three slots
remain unused.

## Source audit and implemented mechanism

The student uses Icefall revision
`3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`. Existing CTC lengths and the KD
validity mask exclude artificial frames from direct supervision. This experiment
addresses hidden temporal computation instead.

The encoder's `ConvolutionModule.forward` already zeroes padded positions after
the input projection and gate, immediately before depthwise convolution. Its
convolution guard remains in use. Two other paths do not respect item boundaries:

- The embedding's three temporal convolutions have no temporal padding, so a
  valid output from them cannot read artificial input frames. They nevertheless
  create an invalid hidden tail in a batch. The following ConvNeXt depthwise
  convolution has three frames of symmetric temporal padding and can read this
  tail into an item's last three valid embedded frames.
- `SimpleDownsample` fills a singleton's partial final group by repeating its
  last frame. In a mixed-length batch, that item's partial group instead reads
  artificial hidden positions. This occurs in reduced-rate encoder wrappers
  and at the final output downsampler.

`modal/student_masked_convolutions.py` propagates exact temporal lengths through
each embedding convolution and zeros invalid positions immediately before the
next convolution, after intervening biases and nonlinearities. It masks the
ConvNeXt input and resulting embedding. Valid lengths remain `(L - 7) // 2`.
Masking the tail at ConvNeXt recreates the singleton's intentional zero boundary;
no convolution kernel or receptive field changes.

Length-aware pooling gathers each item's last valid frame to complete its final
partial group, applies the existing learned softmax weights, and zeros only
completely invalid output groups. It preserves last-frame repetition rather
than substituting zero padding. Reduced-rate wrappers now receive the full-rate
mask, derive their pooled lengths, and pass an appropriate mask into their
existing encoder. Validity is retained between stacks and at output pooling.

Installation replaces the embedding, encoder and reduced-rate wrapper classes
with subclasses. It introduces no parameters or state-dictionary keys. Their
streaming methods, cache layout and ordinary streaming downsampling remain
inherited unchanged. `student_train.build_model` has an opt-in flag and the
checkpoint loader restores that flag. The training image includes the new
module. Existing checkpoints and the default constructor remain compatible.

## Matched refit and export prepared

`modal/student_masked_refit.py` starts both arms from the supplied shipped
checkpoint. It freezes the embedding and lower encoder stacks and trains only
the final two stacks and CTC head. Both arms receive identical augmented tensors,
batch order, Torch/Python seeds, optimizer settings and update counts. The fixed
objective is valid-frame CTC plus five times valid-frame KL from the frozen
shipped model, preserving ordinary readings and their pauses. Learning rate is
`1e-5`, with at most three epochs and 45 minutes of fitting before final
validation/export. These settings were chosen without practice observations.

Only prepared training/clean-match recordings with the gold exclusion list
applied, and help-train's training-split `none` takes with no events, repeats,
unsupported labels or exclusions, are admitted. Original recordings use the
existing teacher-anchored canonical transcript helper. Learner takes use
canonical phones; scripted mistakes are never substituted for performed
transcripts. Mono 16 kHz audio must be long enough for the embedding and at most
30 seconds. Speaker identifiers survive admission. A fixed speaker hash creates
disjoint training and validation groups, with overlap rejected.

The CLI admits at most ten hours of original audio by default and mixes learner
takes at approximately one per four original items. Selection uses up to 128
speaker-held original items and the learner validation partition, with ordinary
CTC plus preservation KL as its objective and PER logged only as a diagnostic.
Both arms are selected together at the same update using their mean validation
objective. This keeps their selected data exposure and update counts matched.
Admitted content/order and audio hashes are fingerprinted in the training log.

Augmentation reuses the shipped reverb, noise, gain, Kaldi fbank and SpecAugment
operations. Export uses the ordinary streaming wrapper and MatMul int8
quantization. Both arm directories must retain exact copies of `teacher.onnx`
and `teacher.onnx.data`; missing inputs or copy mismatches fail. Export compares
three streaming windows and warmed caches against PyTorch, asserting fp32
parity, finite int8 outputs and matching shapes, and recording int8 probability
differences. A completion marker is written only after both bundles finish.
The inventory, 40 ms clock, tracker, checker, gap path and insertion acceptance
rules have no changes.

## Validation attempted and limits

Local Python syntax compilation, `git diff --check`, and independent pure-Python
algebra checks passed. The algebra checks verify the embedding length formula
for input lengths 9–2000 and last-frame-repeat gathering for every residue at
pooling factors 2, 4 and 8. They do not execute the acoustic modules. A first
attempt at the algebra check using local NumPy encountered an incompatible
installed binary; the successful check uses only the Python standard library.

The Python suite covers the unmasked embedding's batch-boundary negative
control, isolated-versus-masked valid outputs, arbitrary tail perturbations,
zero gradients into artificial input frames, strided lengths, partial pooling,
full-model/state-dictionary round trips and identical streaming outputs/caches.
It also checks restricted trainable blocks, speaker separation, data admission,
deterministic augmented tensors, missing teacher failure, and a small primary
ONNX export. The ONNX unit test uses byte-copy fixture auxiliary files; it does
not substitute for the real bundle check.

`packages/sdk/test/masked-convolutions.test.ts`, enabled by
`TILAWA_MASKED_CONVOLUTIONS_DIR`, loads both complete arms through the ordinary
SDK. It checks primary and insertion-asset hashes, inventory, frame count,
finite posteriors and invariance to audio push size. It is skipped without
exported bundles.

No runtime test has a successful result. Supervisor requests returned no shard
logs or executable diagnostics:

| Run | Request | Result |
| --- | --- | --- |
| 1 | Initial CPU environment probe | `NotFoundError: No such file or directory` |
| 2 | Training-image CPU `python3 -c "print(123)"` | Same service error |
| 3 | Six-shard `bash tools/autolab_tests.sh` | Same service error |
| 4 | Focused Python boundary tests in the training image | Same service error |
| 5 | Expanded Python suite, including refit/data/export checks | Same service error |

Run 1 had unnecessarily escaped diagnostic code, so run 2's minimal command is
the clean service probe. Runs 3 and 4 failed without test execution; their
errors are not failing assertions. No training checkpoint could be inspected or
loaded. GPU training was not reserved after the CPU service failed.

Masking temporal forward computation does not prove full training invariance:
Icefall Balancer/Whiten custom backward regularizers can reduce over batch/time,
and stochastic training or a singleton shorter than the configured convolution
chunk can differ in existing chunk geometry. This implementation preserves
that geometry. The deterministic full-model boundary test uses items longer
than a configured chunk at every reduced frame rate. Runtime confirmation of
these tests is still required.

No restored-consonant count, target-warning win count, listener change, overall
change, clean-session effect or studio effect is available. There is no
`.autolab/model.json`; replaying the main model would not evaluate this change.
The ignored outcome records `needs-human` for execution restoration, without
invented measurements. Engine changes remain available for continuation.

## Continuation after supervisor restoration

Run the Python suite and required core/SDK checks first:

```sh
python3 -m unittest discover -s modal -p test_student_masked_convolutions.py -v
bash tools/autolab_tests.sh
```

Identify and verify the checkpoint corresponding to the shipped primary
listener and its complete baseline bundle, then run the paired refit in the
authorized training image on L4, with a 60-minute supervisor timeout:

```sh
python3 modal/student_masked_refit.py \
  --checkpoint /student/prior-runs/<verified-shipped-run>/ckpt.pt \
  --base-bundle /lab/data/models/<verified-complete-baseline> \
  --output /scratch/listener-masked-training-convolutions/paired
```

Those checkpoint/bundle placeholders have not been verified. Complete the real
SDK bundle test after both exports:

```sh
TILAWA_MASKED_CONVOLUTIONS_DIR=/scratch/listener-masked-training-convolutions/paired \
  pnpm --filter @tilawa/sdk exec vitest run test/masked-convolutions.test.ts
```

Compare shipped, control and masked full-session phone decoding through the
ordinary SDK, including existing insertion verification. The five card targets
are p07 `37:36:1` at 144.12 s, p33 `26:208:1` at 156.16 s, p56 `32:21:1` at
106.76 s, and p57 `51:18:1` at 95.26 s and `51:20:1` at 112.26 s. Inspect
previously missing planted consonants in order relative to control, and actual
warning wins relative to shipped. Select no checkpoint or setting from those
observations. Apply the acoustic/target stop instructions only when those
measurements exist; at three or more warning wins, declare the fully verified
masked bundle and run practice scoring before a verdict.

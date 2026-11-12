# Asynchronous frequency-band fusion

Status: implementation awaiting remote validation and matched training.

## Prediction and premise

Weak added-word consonants may be lost when frequency regions are fused at
identical timestamps despite different locally varying acoustic delays. The
prediction is two recovered additions and one collateral wrong call: listener
-1, overall -1. The experiment card is unchanged. The base listener is
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`, on `lab-v1.5u`.

The card requires at least two registered targets to regain a previously
missing planted consonant in correct sequence order against the matched
synchronous control in full-session exported decoding. The execution brief
also requires at least three target wins before practice scoring. These are
different measurements: hearing an additional consonant does not by itself
establish a confirmed warning. Neither measurement has completed.

## Implementation

`modal/student_band_fusion.py` places a trainable input adapter before the
shipped student's Conv2d subsampling/frequency fusion. Eight fixed contiguous
regions each contain ten of the existing 80 log-mel channels. A grouped causal
predictor reads each region's mean energy and first difference over three
frames, producing one bounded, audio-dependent delay per region and frame.
Its readout is initialized to zero, giving an exact identity at initialization.
The one-sided clamp has a nonzero derivative there, so zero initialization
does not make the offset head untrainable.

Delays range from zero to four existing 10 ms feature frames. This past-only
alignment can delay early regional cues to meet later cues without adding
future lookahead. It cannot advance a late cue or repair missing waveform
information. Linear interpolation uses the preceding/current sampling interval;
frame indices remain integers and fractional offsets stay float32 under bf16
encoder autocast. The layer leaves the
waveform, feature count, 40 ms output queries, phone inventory and timestamps
unchanged. It can still fuse unrelated sounds and change recognition.

The matched synchronous arm has the same initialized predictor and trainable
parameter count. Its offsets are fixed to zero; predictor outputs instead
modulate instantaneous within-region spectral contrast. It executes the same
sampling operations and keeps every predictor parameter active, rather than
padding the control with unused parameters. Both arms start from the same
acoustic checkpoint and identical predictor weights.

Streaming carries four raw log-mel frames as an additional `band_history`
ONNX cache. Cache values are represented relative to the existing silence
padding value so SDK zero initialization represents silence. Each call saves
only the last four frames of its committed input prefix, excluding the 13
preview frames. The SDK's existing generic manifest-state interface carries
this cache; no production SDK, tracker, checker or insertion-rule changes are
needed. Overlapping windows must reproduce the full causal sampler.

## Frozen training and export recipe

The runner admits prepared original `clean-match/train` recordings outside
the shipped gold exclusion list, with the shipped transcript/teacher-edge
target construction. It additionally admits help-train `train/none` takes
without events, repeats, excluded labels or unsupported labels. Mistake
scripts never become performed transcripts. The available authorized pool and
its speaker exclusions still require verification in the supplied container.

Original speaker clusters and learner speaker IDs are hash-partitioned within
the supplied training pool, with one fifth reserved for checkpoint selection.
The bounded recipe selects four hours of original recordings, mixes learner
takes at one per four original items, and selects checkpoints on up to 128
original validation recordings plus the reserved none-labelled learner takes.
Selection speakers are disjoint from update speakers. Both arms see exactly
the same batches, augmentation draws and seeded encoder dropout, for at most
two epochs/360 paired updates. Reverb, noise, gain and SpecAugment retain the
shipped distributions. Bounds, learning rates and selection criteria are
fixed before any practice decoding.

Both arms optimize transcript CTC plus a weight-5 posterior KL preservation
objective against the frozen shipped acoustic model on the same ordinary
readings and pauses. Acoustic parameters use learning rate 1e-5; predictor
parameters use 1e-3; gradient norm is clipped at 5. Checkpoints minimize PER
within the training-speaker validation partition, independently per arm.
Chunk masking retains the shipped encoder geometry; actual carried-state
export parity is checked separately.

The exporter embeds the sampler and its history cache in the primary ONNX
graph, applies the existing dynamic MatMul int8 export, and verifies fp32
ONNX against carried-state PyTorch streaming on training-only recordings.
It also records int8 top-1 agreement and checks all declared graph input names.
Each bundle must include the unchanged folded insertion teacher and external
data, validated against the SDK registry's existing sizes and SHA256 hashes.
Supplied manifests, models, evidence and benchmark tools are not edited.

## Validation and limits

Added Python tests cover exact identity and nonzero readout gradients in both
arms, independent bounded regional delays, absence of future dependency,
streaming overlap and committed-prefix state ownership, new-stream reset,
padding and SpecAugment, long-recording bf16 coordinate precision, and standalone
ONNX sampling/state parity. A SDK test exercises the additional cache through
ordinary listener pushes/end, checking propagation once per window and reset
without changing feature or output clocks.

Local Python compilation, Node's TypeScript syntax check and `git diff --check` passed. PyTorch and ONNX Runtime
are not installed locally; heavy execution remains on the authorized supervisor.
Remote execution results, matched checkpoints, complete bundle verification and
full-session target comparisons are pending. No practice score has been used.
No acoustic kill or zero-target result is inferred from missing execution.

Supervisor run 2 requested the supplied six-shard core/SDK build and tests;
run 3 requested only the sampler unit tests on the training image; run 4
requested only a default-image Python command that prints and writes an
execution marker to `LAB_OUT`. All three returned exactly
`NotFoundError: No such file or directory`, without a returned container log
or marker. Their result files record zero spend for those requests. These
failures do not identify whether the problem is startup, execution or artifact
retrieval. The GPU inventory (run 1) and GPU sampler test (run 5) are still
pending. No alternate Modal access has been attempted.

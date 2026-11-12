# Online peer phone-head learning

Status: implementation retained, acoustic validation blocked. The supplied
supervisor fails before executing commands. No trained model or model
declaration exists; this experiment is not ready for scoring or folding.

## Prediction and mechanism

The card predicts two recovered additions with one collateral wrong call:
listener −1, overall −1. Fixed-teacher imitation may carry blank-dominated
consonant errors into the student. The experiment instead freezes the shipped
encoder and learns three independent nonlinear acoustic readouts together.

`modal/student_peer_head.py` implements a 64-unit tanh readout followed by the
unchanged phone alphabet, including blank. For the 256-dimensional encoder and
43 classes, each head has 19,243 parameters. Both layers are independently
initialized; these are complete readouts rather than residual adjustments to
the old linear head. No head averaging or inference voting is used.

All arms receive 18 complete recording exposures per head, with a shared
six-epoch transcript-CTC warmup and twelve further epochs. Each recipient has a
different deterministic recording order. At each joint step, its two peers
predict that recipient's exact cached views. The arithmetic mean of their
probabilities is detached and supplies a KL agreement target. All three losses
and gradients are computed before any optimizer advances. Thus live targets
come from the current peers, without within-step update-order effects.

Three arms use the same architecture, initializations, AdamW updates, recording
orders and observations:

- **Live:** reciprocal agreement with the other heads' current predictions.
- **Independent:** transcript CTC and preservation, without agreement.
- **Frozen:** agreement with the other heads' snapshots at the warmup boundary.

Independent training still executes the detached peer forwards with zero
agreement weight. Frozen targets are evaluated rather than cached separately.
All arms take the same warmup snapshots. Explicit accounting verifies identical
recording/view exposure, differentiable and detached forward counts, optimizer
steps and parameter counts, including selection runs.

## Training inputs and selection

`student_peer_train.py prepare` accepts the supplied help-train training
speakers' complete none-labelled takes. Mistake scripts, repeats, unsupported
labels, excluded takes and every speaker with any non-training label are
rejected. Optional additional transcripts require verified/authorized flags,
identified training speakers, audio checksums and an explicit authorized
release-held-out speaker exclusion list. No practice-session input is exposed
by preparation or fitting.

Each recording supplies its original view and two draws from the shipped
RIR/noise/gain/SpecAugment recipe. The full recordings retain ordinary speech
and natural pauses. A fixed 0.1 shipped-output KL preservation term applies to
all frames in all arms. It limits drift but can also retain shared blank
errors; it is not a new teacher transcription target.

Native encoder features and shipped posteriors are cached from the exact
streaming ONNX graph. Preparation checks eight stateful windows for unchanged
primary and cache outputs, hashes the audio/transcripts/views/augmentation
banks, and rejects impossible complete CTC alignments. Cached features and
preservation targets never receive gradients.

Three speaker-grouped training partitions select agreement strength from
{0.1, 1, 10} and one initialization slot, separately for each arm. Feature
normalization is fitted on the training side of each partition. Selection
minimizes equal-speaker mean(PER + 0.1 shipped KL), with deterministic ties.
Blank occupancy is logged as a collapse diagnostic. Independent selection
repeats the same grid work to match compute. Each selected arm is then refitted
on all admitted training recordings; practice incidents never select a strength,
head or checkpoint.

`student_peer_export.py` grafts only the selected nonlinear readout into a new
copy of the shipped encoder graph. It prunes the old projection, preserves
public inputs, cache outputs, frame timing, alphabet and front end, and writes a
new model manifest/checksum. Existing source manifests are never rewritten.
The unchanged insertion teacher and external weights must be present in the
source bundle, copied and checksum-verified. No insertion rule, decoder,
tracker or checker changes are made. Export validates complete-utterance ONNX
outputs against the selected Torch head. `selection-final.json` is written
only after all three selected bundles pass those checks.

## Validation and limits

Read AGENTS.md, the card, shipped training/export code, help-label schema,
supervisor isolation/compute contract and SDK model-loading/streaming contract.
Feature extraction and bundle copying were adapted from the earlier
`exp/listener-interacting-phone-code-head` implementation; its unexecuted
checks are not evidence that this experiment works.

Six dependency-free unit tests pass locally: clean-take admission, mixed-split
speaker rejection, stable grouped partitions, identical exposure with different
orders, speaker-weighted deployment selection, and deterministic selection ties
with invalid-metric rejection. All seven new Python files pass `py_compile`.

Eleven additional Torch/ONNX tests cover independent initialization, detached
probability-mean agreement gradients, evolving versus frozen targets, recipient
view correspondence and simultaneous update order, frozen observations under
CTC, matched control trajectories/compute, single-readout ONNX parity,
unchanged caches, stream draining and auxiliary-bundle integrity. They have
not executed; Torch and ONNX are unavailable on the Mac. They were
requested through the supplied supervisor, as were the core and SDK suites.

Supervisor requests:

| Run | Requested work | Result |
| --- | --- | --- |
| 1 | Read-only supplied-data/runtime probe | `NotFoundError: No such file or directory`, before command execution |
| 2 | Required core/SDK build and tests, six shards | Same startup error |
| 3 | Training-image help/model/augmentation/runtime probe | Same startup error |
| 4 | All peer data/head/export unit tests, training image | Same startup error |
| 5 | Frozen-feature preparation and all matched fits/exports, training image | Same startup error |

No training recording has been read, no head fitted, and no bundle exported.
Target cases acoustically evaluated: **0 of 5**. Consonant recoveries relative
to either control, target warning wins, aggregate deltas, clean-session warnings
and studio drift are **unmeasured**. No practice score has been requested
(all three allowances remain). No `.autolab/model.json`, killed outcome or fold
is recorded. Infrastructure failure does not establish the acoustic premise
or a zero-gain result.

Once compute works, finish preparation, matched grouped selection and all three
complete verified bundles before evaluating the registered targets. The card
requires previously missing planted consonants restored in order on at least
two targets relative to both selected controls; the task's additional stop rule
requires at least three target-warning wins. If either measured premise fails,
revert the engine changes and record the killed result. With three or more
target wins, declare only the verified live-head bundle and run the authorized
practice score before deciding. The current untrained source is not a scored
model change.

# Learned frequency pooling

Status: ready-for-human; implementation unvalidated, execution service unavailable.

## Prediction and premise

Fixed mel aggregation may combine a weak consonant's frequency structure with
stronger neighbouring vowel energy. Learning that aggregation predicts three
recovered additions and one collateral wrong call: listener -2, overall -2.
The experiment card is unchanged. Its base is `0f74f7b9`, selecting
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0` on `lab-v1.5u`.

The card's acoustic kill criterion is fewer than two target spans regaining a
previously missing planted consonant against a matched fixed-pooling control
in full-session exported decoding. The execution brief additionally requires
at least three target wins before practice scoring. Neither criterion has been
evaluated; absence of execution is not evidence that the acoustic premise fails.

## Implementation

`modal/student_frequency_pooling.py` extracts original 512-point FFT powers
using the existing 16 kHz, 400-sample Povey windows, 160-sample shift, frame-local
DC removal and 0.97 pre-emphasis. No floor precedes aggregation. Each of the 80
filters is a softmax distribution within the positive contiguous support of its
shipped Kaldi mel filter. Its total weight is fixed to that original filter's
mass. This preserves the shipped feature scale and mel initialization while
allowing frequency weights inside each region to learn. Region boundaries do
not move, and bands containing only one FFT bin cannot learn a different shape.

The paired fine-tune starts both encoders from the same shipped checkpoint.
Both arms allocate the same pooling parameters; the control freezes them.
The encoders train jointly with CTC and a weight-5 KL preservation loss against
the frozen shipped model on ordinary-reading frames, including pauses. The
encoder learning rate is 1e-5 and pooling learning rate 1e-3; these choices are
fixed before target evaluation. The recipe allows four epochs within a
50-minute training limit, with identical seeded batches, dropout, and the
shipped reverb/noise/gain and SpecAugment distributions in both arms.

Only prepared original `clean-match/train` clips outside the gold exclusion
list and help-train `train/none` recordings without events, repeats or excluded
labels are admitted. Mistake scripts do not become performed transcripts.
Original clips retain the shipped spliced-reference target construction.
Every selected original training item is retained; learner takes are mixed
at 1:4 by item count. Speaker clusters and learner speaker IDs are deterministically
partitioned by hash within supplied training speakers. Checkpoint selection
uses only disjoint validation speakers in that training pool, not release
held-out speakers or practice sessions. This admission path still needs to be
exercised against the supplied data; the inventory request never ran.

The exporter writes the encoder's normal dynamic-MatMul-int8 ONNX graph and
normalized filters plus fixed scales in its generated input contract. SDK
`Fbank` validates and snapshots that contract. Every primary or fresh cropped
listener uses the model's filters before logarithmic flooring, with the same
80 feature channels and 40 ms output clock. Existing assets and manifests were
not edited. Tracker, checker, token decoding, and folded insertion verification
are unchanged. The existing teacher uses its own unchanged front end.

## Validation and execution limits

Added Python unit tests cover mel identity on silence/quiet speech transitions,
learned-weight gradients, positive contiguous support, preserved filter mass,
and exported-weight/floor parity. Added SDK tests cover normalized filter
validation, filter ownership, energy-scale ratios, frame timing and propagation
from the primary manifest to independent listener instances.

Python compilation and `git diff --check` passed locally. A lightweight direct
Node check passed streaming-split invariance and learned-filter energy-scale
ratios on a synthetic tone (98 frames). The unchanged default front end matched
all 40 stored reference frames with maximum error `0.0000209808349609375`, and
retained the reference recording's 587-frame length. This checks frontend
behavior only; it is not a substitute for the required suites or trained-model
export parity.

Supervisor run 1 requested a read-only training/model inventory on the training
image (one CPU shard, ten-minute limit). Run 2 requested the supplied core/SDK
build and tests (six CPU shards, twenty-minute limit). Both returned exactly
`NotFoundError: No such file or directory`, without any returned container log.
Their result files report zero GPU spend and the full $120 budget remaining.
Run 3 requested only `pwd`, listing `/lab/data`, and disk availability under
`/scratch` on the default image, to isolate service startup from training imports.
It returned the same `NotFoundError` with zero spend and no container log. All
three attempts therefore failed before project code could execute. Restoration
of the supervisor execution service is required; no alternative Modal access
was attempted.

No GPU training, checkpoint selection, model export, target diagnosis or practice
score has completed. No trained model bundle exists. The ignored
`.autolab/model.json` declares the intended `paired/learned` export location,
which is currently unavailable: this prevents a later supervisor run from
silently scoring the unchanged shipped model as the experiment. The ignored
outcome records an infrastructure block, not an acoustic kill. Running the shipped model
would not evaluate learned frequency pooling, and no zero-gain or zero-target
result is claimed. The implementation requires restored supervisor execution,
successful full tests and Python parity tests, a verified shipped starting
checkpoint, both trained/exported arms, and full-session SDK target decoding.
Before handing a candidate to scoring, its complete primary bundle must also
retain the shipped folded insertion teacher assets and pass export/frontend
parity. No fold or discard was performed.

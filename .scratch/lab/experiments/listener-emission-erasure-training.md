# Consonant emission erasure training

Status: implemented; real training and acoustic validation blocked by the
supervisor service before container execution. No selected real head or complete
model bundle is available. The experiment card is unchanged.

## Prediction and mechanism

The card predicts listener −1 and overall −1, with no losses predicted. Its
premise is a valid consonant depending on one fragile CTC emission. This remains
an untested acoustic hypothesis, not an established cause of the five misses.

`modal/student_erasure.py` computes a transcript-conditioned Viterbi alignment
with a separate CTC state for each phone occurrence. Each recording/view/update
samples one transcript consonant occurrence, then finds its strongest emission
among the frames assigned to that occurrence. Repeated consonants separated by
the required blank have separate states and separate peak candidates.

The auxiliary sequence loss substitutes a uniform log-probability row,
`−log(vocabulary size)`, at exactly that frame before recomputing the complete
transcript CTC forward sum. All other rows, the transcript, acoustic inputs and
timesteps stay intact. Selection is detached from autodifferentiation. The
selected row has no acoustic gradient, while surviving rows can learn additional
support. This is neither blank forcing nor waveform masking. It does not delete
a timestep or invent a new transcript.

The alignment checks exact path feasibility, including zero-probability support;
an infeasible authorized transcript aborts rather than silently becoming a zero
loss. An infeasible auxiliary exclusion is skipped. Uniform erasure cannot
remove an existing legal path when the other scores are finite: it preserves the
frame and gives every label positive probability. Minimum-length transcripts
therefore remain legal, rather than being treated as shorter acoustic sequences.
No-consonant examples receive the ordinary losses and preservation term.

`modal/student_erasure_train.py` compares three arms initialized from the same
shipped head: ordinary training, uniform erasure of a random recording frame,
and erasure of the selected consonant's strongest emitting frame. The random arm
uses the same occurrence draw, then selects a frame uniformly over the complete
real output sequence. All arms do the same alignment/feasibility work and two
complete CTC evaluations per example. The ordinary arm's second evaluation uses
the original rows, giving the same sequence-loss weight and evaluation count.

The head objective is ordinary CTC plus the arm's second CTC likelihood plus
twice the full-recording KL preservation of shipped posteriors. CTC losses are
normalized by real output frames. Ordinary speech and real pauses are retained
as complete recordings with their original output anchors; no new pause labels
or synthetic masked-waveform positives are introduced. The architecture and
phone inventory are unchanged.

## Data, selection and export

The frozen recipe admits up to 512 authorized original clean-match training
recordings and help-train training speakers' none-labelled takes, at most 30
seconds each. Original transcripts come from the prepared reference field;
help transcripts come from their labelled canonical ayahs. Mistake, repeat,
unsupported and explicitly unscored takes cannot supply canonical supervision.
Non-training and excluded/gold original speaker clusters are removed in full;
non-training or explicitly excluded help speakers are also removed in full.
The loader rejects unavailable speaker provenance and cross-speaker waveform
duplicates. Actual enrollment and source schemas have not been inspected here.
The original-source adapter expects prepared `/student/s1/*.parquet` recordings
and the existing `/student/aug/noise.npz` and `rir.npz` banks. Adjacent held-phone
runs follow the shipped transcript-collapse convention.

Speakers are reserved separately within each source by a stable hash before any
acoustic comparison. Every example has one unmodified view and two cached views
using the existing RIR/noise/gain/SpecAugment functions and ranges. All three
arms share the views, order, batches of eight, clipping bound five, FP32 head
storage, SGD updates at learning rate 0.0001 and twelve epochs. The source
identities, encoder graph, feature caches, partitions, recipe, checkpoints and
auxiliary assets carry checksums. The recipe is fixed before target inspection.

The strongest-erasure arm's ordinary CTC and full-recording preservation on
reserved training speakers select the checkpoint, including epoch zero. The
control heads are retained at that same epoch and update count. Selection uses
the exported int8 weight rounding where applicable, and never practice incidents
or release-held-out speakers. The intended compute budget is one day; no real
preparation/refit runtime has been measured.

`modal/student_head_refit.py` reuses feature capture and head export from
`exp/listener-noise-aware-teacher-loss`, without its observation channel. The
feature cache exposes the dequantized activations actually consumed by the
shipped integer classifier. The encoder is never a trainable module. Export
changes only existing head constants and preserves all graph nodes, encoder
tensors, state shapes, input geometry, alphabet, output clock and frontend.
SDK decoding, tracker, checker and insertion verification are unchanged.

Every export requires and copies the registered insertion teacher and all its
external weights with their original checksums. Source manifests are read only;
new bundles receive their own model checksums. Checks include capture parity,
unchanged recurrent states, quantized projection parity and complete streaming
parity on held-training-speaker frontend caches. Missing teacher files are an
error, never an excuse to validate the primary head without the folded verifier.

## Validation

Python compilation passed. The numerical and synthetic ONNX suite covers exact
tiny CTC path sums and logit gradients, repeated occurrence alignment, one-row
gradient disconnection, surviving consonant gradients, support infeasibility,
random-control reproducibility, equal sequence-evaluation counts, no-consonant
examples, clean-label filtering, speaker/cache provenance, shared checkpoint
epochs and update counts, frozen feature files, float/int8 head capture,
per-channel rounding, final stream trimming, complete exported streaming parity,
unchanged graph/state contracts and missing/corrupt teacher bundles.

All 19 local tests passed without skips: 13 erasure/training-contract tests in
42.07 seconds and six synthetic ONNX tests in 20.70 seconds. Commands:

```sh
.autolab/erasure-test-env/bin/python -m pytest -q modal/test_student_erasure.py
.autolab/erasure-test-env/bin/python -m pytest -q modal/test_student_head_refit.py
```

The isolated test environment used Python 3.13.7, Torch 2.8.0, NumPy 2.5.4,
ONNX 1.19.0, ONNX Runtime 1.22.1 and pytest 9.2.0. No real audio processing or
model training ran on the Mac. Compilation and whitespace checks passed. These
synthetic checks do not demonstrate restored acoustic consonants or target wins.

The four authorized supervisor requests were:

1. Normal-image data/model/dependency probe, five-minute limit: pre-execution
   `NotFoundError: No such file or directory`, USD 0.00.
2. Training-image authorized-data/dependency probe, ten-minute limit: the same
   pre-execution error, USD 0.00.
3. Supplied six-shard core/SDK build-and-test harness, thirty-minute limit: the
   same pre-execution error, USD 0.00. Core/SDK tests did not execute.
4. Training-image unit suite, prepare, matched refit and complete export,
   sixty-minute limit: the same pre-execution error, USD 0.00.

There are no remote command logs or successful real-input inspections from
these failures. They establish an execution block, not missing training data or
missing model assets. No practice score was requested; all three remain available.

## Limits and continuation

No authorized acoustic cache, matched trained heads, real checkpoint selection,
complete model bundle, full-session target decode or measured practice change
was produced. Target wins are unknown, not zero. The acoustic premise has not
been falsified, so no killed outcome or engine rollback is justified. The code
is retained for continuation but does not yet change the scored listener.

Viterbi supplies one maximal alignment rather than marginal occurrence
occupancies. Only one consonant occurrence is sampled per view/update. A neutral
row can still carry a transcript symbol in the likelihood, so the objective does
not guarantee that an unrestricted decode acquires additional acoustic support.
Both that transfer and recognition drift require real exported streaming tests.
Ordinary output anchors limit drift; they do not guarantee clean-session safety.

Once supervisor execution works, run the new Python suite and the supplied
core/SDK harness, then prepare, fit and export through the authorized training
image. The commands are:

```sh
set -e
python3 -m pytest -q modal/test_student_erasure.py modal/test_student_head_refit.py
python3 modal/student_erasure_train.py prepare \
  --model /lab/data/models/spliced-clean-int8 \
  --out /scratch/listener-emission-erasure-training/refit-v1
python3 modal/student_erasure_train.py fit \
  --out /scratch/listener-emission-erasure-training/refit-v1
python3 modal/student_erasure_train.py export \
  --model /lab/data/models/spliced-clean-int8 \
  --out /scratch/listener-emission-erasure-training/refit-v1
```

Preparation, fitting and export may be separate supervisor requests so each
fits its sixty-minute limit. A complete frozen cache can be reused unchanged.
Only after all three real exports verify should identical full-session SDK
streams compare the strongest-erasure arm with both matched controls at p07
37:36:1, p33 26:208:1, p56 32:21:1, p57 51:18:1 and p79 20:40:25. Check newly
recovered planted consonants in the correct order and the unchanged engine's
addition verdicts. Targets are reporting cases, never checkpoint selection.

The card asks for two consonant-recovery targets; the final task rules require
three target wins before practice scoring and take precedence. Fewer than three
wins or a disproved premise calls for rollback, a measured kill write-up and the
specified killed outcome. At least three wins requires the authorized practice
score before deciding or narrowing. Collateral losses belong to the fold gate.

There is no `.autolab/model.json`: declaring an absent bundle would block
validation, and a main-model decode would not evaluate this training change.
After verification, declare `refit-v1/strongest-bundle` for the supervisor's fresh
decode. No scorer, lab tooling, supplied data, evidence, tracked manifest or
experiment-card edits were made, and no fold was performed.

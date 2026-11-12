# Noise-aware teacher observations

Status: training and export implementation complete; real execution blocked by
the supervisor's Modal service. No trained model or measured target result is
available. The experiment card is unchanged.

## Prediction and mechanism

The card predicts one recovered addition with no losses: listener −1 and overall
−1. Its premise is systematic teacher recognition errors becoming student phone
identity through direct distillation. This prediction remains untested.

`modal/student_noise.py` estimates a normalized observation channel whose rows
are latent student phones and whose columns are teacher observations, including
blank. It does not align teacher argmax labels as transcript truth. Its
forward/backward calculation sums every legal CTC alignment of the authorized
transcript, including blank states and the intervening blank required for an
actual repeated phone. Each frame's soft teacher distribution contributes its
expected log observation likelihood. EM accumulates fractional phone/teacher
counts under the resulting transcript-placement posterior. A diagonal
Dirichlet prior retains an identity preference, including for unobserved rows.

`modal/student_noise_train.py` selects that prior on whole reserved training
speakers and freezes the resulting channel. The refit objective compares teacher
probabilities with `student probabilities @ channel`, alongside ordinary
transcript CTC, independent real-silence blank supervision, and preservation of
the shipped outputs across ordinary speech and pauses. Thus a latent consonant
can explain a systematically blanked teacher observation without itself having
to become blank. The ordinary-distillation control uses an identity channel.

Both arms start from the same shipped linear phone head and use identical
recordings, cached augmentation views, batches, clipping, learning rate, and
optimizer update count. Only the channel in the KD term differs. The encoder
has no trainable path: the runner caches its streaming outputs and exposes the
rounded activations actually consumed by the shipped integer projection.

The data recipe admits at most 512 authorized original clean-match recordings
and help-train training speakers' none-labelled takes, at most 30 seconds each.
Original targets use their authorized transcripts; help targets use their
canonical ayahs. Held notation is collapsed using the shipped transcript
convention. Takes containing labelled mistakes, repeats, unsupported events or
unscored content never supply canonical transcripts. Original gold/excluded or
non-training speaker clusters are removed in full; excluded/non-training help
speakers are also removed in full. Exact waveform copies under different
speaker identities abort preparation.

Each recording has one ordinary view and two views from the existing RIR,
noise, gain and SpecAugment functions and ranges. Cached original teacher
top-five probabilities remain soft and are normalized. Help takes obtain
full-distribution targets from the existing inference-mode ONNX teacher, with
the SDK's SeamlessM4T frontend. Augmented views inherit their original take's
targets using the existing zero-delay, clamped 25 Hz mapping. Independently
labelled real silence requires a 120 ms waveform neighborhood entirely below
one 16-bit PCM quantization step; teacher blanks and padded frames never define
silence labels. Other pauses retain the ordinary-output preservation objective.

Whole speakers are reserved separately within each source. Only unaugmented
training views estimate the channel. Prior fractions of 0.01, 0.1 and 1.0 times
the training-frame count per inventory class are compared by held-training-
speaker observation likelihood. Twelve EM iterations and twelve paired refit
epochs are fixed before target queries. Ordinary CTC, pause and preservation
loss on reserved training speakers select the corrected checkpoint, including
the unchanged initial checkpoint as a candidate. Both arms are retained at
that same epoch/update count. Selection includes the actual exported int8
weight rounding, rather than evaluating only a float head. Speaker partitions,
input/cache hashes, channel hashes and selection history are recorded.

`modal/student_head_refit.py` reuses the feature capture and parameter export
from `exp/listener-transfer-weighted-head`, without its gradient weighting. It
changes only existing phone-head constants and preserves the graph nodes,
encoder tensors, streaming states, alphabet, frame clock and front end. No SDK
decoding, tracker, checker or insertion-verification rule changes. Export
requires the unchanged registered insertion teacher and all its external
weights, checks their source and copied checksums, and verifies streaming-state
parity. Supplied manifests are read only; complete new bundles receive new
model checksums and sizes. Neither a bundle nor its model declaration is
created as a fallback when required files are missing.

## Validation

All 23 synthetic numerical, training-contract and ONNX tests passed locally
with no skips in 36.48 seconds:

```sh
.autolab/noise-test-env/bin/python -m pytest -q \
  modal/test_student_noise.py modal/test_student_head_refit.py
```

The isolated test environment used Python 3.13.7, Torch 2.8.0, NumPy 2.5.4,
ONNX 1.19.0 and ONNX Runtime 1.22.1. No real audio processing or model training
ran on the Mac. A preliminary exhaustive NumPy calculation also matched every
tiny CTC path sum. Python compilation and whitespace checks passed.

Tests cover exhaustive alignment and occupancy sums with repeated phones and
known pauses; rejection of impossible transcripts; a transcript-required phone
whose teacher observation is blank; normalized channel rows and an identity
prior for unseen phones; identity equivalence to ordinary KD; a frozen channel;
opposite corrected/direct KD gradients on the missing-consonant fixture;
padding exclusion and latent-blank pause supervision; scripted-label filtering;
soft top-five mass and frame mapping; frozen-cache/speaker-exclusion audits;
paired update counts and training-only checkpoint selection; float/int8 head
reconstruction; per-channel quantization; final drain trimming; unchanged graph
nodes and streaming state; and rejection of incomplete/corrupt auxiliary
bundles. These are implementation checks, not acoustic target recoveries.

Five authorized supervisor requests returned
`NotFoundError: No such file or directory` before container execution, each
reporting USD 0.00:

1. Supplied data/model/dependency inventory, normal image, 5-minute limit.
2. New numerical tests, training image, 10-minute limit.
3. Supplied training/help/model inventory, training image, 10-minute limit.
4. Required six-shard core/SDK build and test harness, 30-minute limit.
5. Final Python suite followed by prepare, matched refit and complete export,
   training image, 60-minute limit.

There were no remote command logs or real input inventories. Core and SDK tests
therefore have not run successfully. The failures do not establish missing
training data or missing teacher assets; neither was inspected remotely.

## Limits and continuation

No authorized feature cache, real channel estimate, matched trained head,
training-selected checkpoint, complete real bundle, target stream decode or
practice score was produced. Actual source schemas, speaker coverage, teacher
and encoder memory use, preparation duration, reconstruction tolerance and
ordinary-reading preservation remain unverified on real inputs. The one-day
budget is the intended bound, not a measured runtime. Target wins and
listener/overall changes are unknown, not zero.

Restore supervisor execution and first run the final unit suite and input
inventory. The submitted preparation/refit/export command is:

```sh
python3 -m pip install -q pytest
python3 -m pytest -q modal/test_student_noise.py modal/test_student_head_refit.py
python3 modal/student_noise_train.py prepare \
  --model /lab/data/models/spliced-clean-int8 \
  --out /scratch/listener-noise-aware-teacher-loss/refit-v1
python3 modal/student_noise_train.py fit \
  --out /scratch/listener-noise-aware-teacher-loss/refit-v1
python3 modal/student_noise_train.py export \
  --model /lab/data/models/spliced-clean-int8 \
  --out /scratch/listener-noise-aware-teacher-loss/refit-v1
```

Run this with fail-fast shell handling in the authorized training image.
Preparation requires a complete source bundle with the registered insertion
teacher files. Once both exports pass, use identical full-session SDK streams
to compare the training-frozen corrected and ordinary heads on the five named
targets: p07, p33, p48, p56 and p57. Check whether a previously missing planted
consonant is restored in correct order and whether the unchanged engine catches
the addition. Teacher agreement alone is correlated evidence, not an
independent listening judgment.

The final task rules require at least three target wins before practice scoring;
this overrides the card's two-stream pilot threshold. Fewer than three wins or
an absent acoustic premise calls for engine rollback and a measured kill note.
At least three wins calls for an authorized practice score before deciding or
narrowing; collateral losses belong to the fold gate. All three practice
scores remain available.

There is no `.autolab/model.json` and no killed outcome. A main-model decode
would not evaluate this refit, and infrastructure failure does not justify an
acoustic kill. The code is retained for continuation, but it does not yet change
the scored listener. No scorer, lab tooling, supplied data, evidence, experiment
card or tracked manifest was edited. No fold was performed.

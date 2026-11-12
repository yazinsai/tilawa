# Paired-view latent transcript paths

Status: implementation and synthetic contracts validated; real-data execution
blocked. No replacement listener is declared. The experiment card is unchanged.

## Prediction and mechanism

Predict one recovered addition, zero losses, listener −1 and overall −1. This
prediction is inside the lab noise band and below the fold floor. The proposed
failure is incorrect temporal placement of weak consonants in degraded-only
training: clearer observations of the same performance may disambiguate their
placement without supplying additional words or teacher-generated transcripts.

Freeze the installed streaming encoder and refit its existing affine phone head.
For original/degraded observations at corresponding sample times, the shared
arm optimizes

```text
log Z_shared(y) = log sum_{q: CTC(q)=y} product_t p_original(q_t) p_phone(q_t)
```

The control optimizes the sum of two independent transcript log-likelihoods.
`modal/student_paired.py` uses log-domain CTC forward-backward, including blank
states and the intervening-blank restriction for repeated target phones. Joint
emissions are the **sum of the two separate log-probabilities**. Their product
is not renormalized. Consequently the shared objective includes alignment
agreement mass; it is not ordinary CTC on an averaged posterior distribution.
Assignments are recomputed jointly at every current-head objective evaluation.
The analytic gradient differentiates the complete path sum, rather than a
Viterbi alignment, a fixed assignment surrogate, or a target predicted by one
view for the other.

Sample timestamps are integer offsets in the original recording. A union grid
groups exactly coincident observations, with no rounded delay or interpolation.
Each view contributes each observed frame once. At a time not sampled by a
view, that view supplies a neutral factor of one, rather than an invented blank
or a duplicated emission. The shared path advances on the union grid; each
independent path advances on its own view grid. This defines the behavior for
offset grids, whose discretization can differ from the zero-delay case.

Both arms compute **both** path sums on every update; only the selected arm's
sum contributes its loss/gradient. This deliberate extra computation gives
identical lattice and projection work for the same data in both arms, without
unequal optimization steps or approximate path refreshes. It does not claim
identical measured wall time.

## Data, fitting and export

`modal/student_paired_train.py` supplies `prepare` and `fit`. Preparation admits
help-train none-labelled training-speaker takes with no events, repeats or
unsupported labels, and no exclusion flags. Canonical duration repetitions
collapse within each ayah, as in the shipped transcript representation; an
articulation repeated across an ayah boundary remains repeated. Scripted
mistake takes are not treated as verified performed transcripts.

Additional authorized performed transcripts may be enrolled as JSONL with
`speaker`, `split`, `authorized`, `verified`, `phones`, `wav` and audio `sha256`.
The existing help split and every non-training split in additional enrollment
exclude those speakers globally before any recording is admitted. An explicit
excluded-speaker list can supplement this. The supplied help-train corpus is
also documented as excluding release-held-out speakers. Audio hashes reject
duplicate recordings; paths must remain within their enrolled corpus.

At least five training speakers are required. Whole speakers are deterministically
partitioned into fit and selection groups by a generic hash, independent of row
order. The current runner can use help-none alone, or supplement it with verified
transcripts; it does not require unprovided additional recordings.

Each recording supplies one original and one deterministic phone view, shared
by both arms. The phone view uses enrolled noise/RIR banks, direct-path RIR
trimming, zero-phase 120–7000 Hz filtering, the existing 5–25 dB noise range,
gain variation and peak limiting. It preserves sample count. Direct-path
trimming and zero-phase filtering make its declared delay zero. The cache still
records explicit delay and exact native sample timestamps, and validates that
correspondence when loaded. No temporal SpecAugment or time stretching is used.

The loss is paired CTC divided by twice transcript length, mean-view full-frame
KL toward shipped outputs, and L2 toward the initial native head. Full recordings
retain ordinary speech and real pauses in both transcript and preservation
losses. These examples limit drift; they do not guarantee clean-session safety.
No teacher posteriors or guessed off-text phone strings enter the objective.

Both arms start from the same dequantized native head on the frozen int8 encoder.
The predeclared schedule is 60 complete-corpus Adam updates at learning rate
0.01 for each of regularizations 0.001, 0.01 and 0.1. Both arms receive all
updates, with no single-arm early stopping. Epoch zero and every tenth update
are eligible checkpoints. A shared round and regularization minimize the pair's
mean PER, averaged equally across selection speakers, then recordings and
views. No practice result selects either checkpoint. Fitting enforces a one-day
deadline, including checks within complete-corpus passes and after selection.

`modal/student_head_export.py` reuses native feature extraction and complete
bundle-copy safeguards from `exp/listener-alternating-path-head-fit`. The
streaming window, state graph, alphabet, frontend and output clock are inherited
from the installed listener. Preparation requires exact original output/state
parity after exposing pre-projection features. Export replaces only the existing
projection, retaining a frozen int8 encoder and fitted fp32 head in both arms.
Removing the native projection's activation rounding is common to both arms;
it cannot establish a shared-path benefit by itself.

Existing insertion teacher ONNX and external weights are required, copied and
checksum-verified. Generated bundle manifests describe the new artifacts; source
manifests are never rewritten. Both selected exports must reproduce their head
posteriors on all selection views within 2e-4 probability tolerance before a
successful training report is written. Tracker, checker, greedy decoding, gap
decoding and insertion verification code are unchanged.

Example commands once the supplied service and enrolled banks are available:

```sh
python3 modal/student_paired_train.py prepare \
  --model /lab/data/models/spliced-clean-int8 \
  --help /lab/data/help-train/labels.jsonl \
  --augmentation <enrolled-noise-and-rir-bank-directory> \
  --out /scratch/listener-paired-view-latent-paths/prepared
python3 modal/student_paired_train.py fit \
  --prepared /scratch/listener-paired-view-latent-paths/prepared \
  --out /scratch/listener-paired-view-latent-paths/fitted
```

## Validation and limitations

All 21 synthetic tests pass, with no skips:

```sh
.scratch/paired-venv/bin/python -m unittest discover -s modal \
  -p test_student_paired.py -v
```

The tests cover exhaustive CTC sums and occupancies, shared versus independent
path mass, rejection of normalized-product substitution, weak-consonant temporal
disambiguation, integer and sub-frame delay, unequal edge coverage, repeated
phones, long peaky lattices, finite-difference gradients in both arms with three
delays, real-pause preservation terms, recording/view weighting, matched update
and lattice counts, selection exclusion, deadline failure, speaker and transcript
admission, native float/int8 head extraction, state preservation and complete
teacher copying. Generated fixtures use no corpus audio. The local environment
uses NumPy 2.5.4, SciPy 1.18.1, ONNX 1.23.2 and ONNX Runtime 1.31.1. The system
NumPy initially had incompatible architecture; isolated workspace wheels fixed
the synthetic-test environment. Python compilation and whitespace checks pass.

All three authorized supervisor requests failed before their commands executed,
each reporting `NotFoundError: No such file or directory` and USD 0.00:

1. Supplied data, help enrollment and source model discovery.
2. The required six-shard `bash tools/autolab_tests.sh` core/SDK harness.
3. Supplied-data retry, bank/asset discovery and the new synthetic suite.

Thus the real help enrollment, augmentation-bank availability and native source
graph have not been inspected. No training cache was prepared, no real-data
head was fitted, and no complete replacement model bundle exists. The required
core/SDK suite has no result. There have been **zero practice-score invocations**;
target consonant recoveries, confirmed warning wins, collateral losses, listener
change and overall change are unknown. Synthetic disambiguation is a solver
example, not evidence that the real targets have this failure.

The implementation remains on the experiment branch for execution when the
authorized service works. There is no `.autolab/model.json`, because declaring
an unfinished bundle would be invalid. Decoding the main listener cannot evaluate
this training change. Infrastructure failure does not falsify the premise, so
no acoustic killed outcome or fabricated zero-win measurement was written.

After real-data preparation, training-only selection and complete export parity,
compare both frozen selected heads on the five registered full-session phone
targets. Measure previously missing planted consonants recovered in correct
order relative to the independent control separately from confirmed warnings.
The card's floor is two consonant-recovery targets; the user's final stopping
instruction additionally requires at least three target wins. If the premise
fails or fewer than three targets are won, roll back engine changes and record
the measured kill. With at least three wins, run the authorized practice score
before deciding or narrowing; the fold gate owns collateral-loss tradeoffs.
The current service errors establish neither a kill nor a successful experiment.

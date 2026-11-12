# Alternating CTC paths and phone-head fitting

Status: implemented; real-data execution blocked and acoustic effect unmeasured.
No replacement listener is declared. The experiment card is unchanged.

## Prediction and premise

Listener −1 wrong call and overall −1, with one recovered addition and no
predicted losses. This is within the lab noise band and below the fold floor.
The speculative premise is that weak consonants have useful frozen encoder
features but remain trapped in self-reinforcing transcript-conditioned CTC
assignments. The experiment changes when assignments are refreshed, rather
than consonant weights, clipping, entropy, or update preconditioning.

## Implementation

`modal/student_alternating.py` implements log-domain CTC forward-backward,
including blank states, repeated-label skip restrictions, empty transcripts,
and impossible-path rejection. Each expectation phase infers frame-class
occupancies for **every fit recording and fixed view** before returning an
immutable assignment snapshot. Selection speakers do not enter that snapshot.

The maximization phase minimizes the complete equal-recording, equal-view
multinomial objective with full-data L-BFGS. It retains teacher full-posterior
cross-entropy, shipped-output KL on every real frame (including pauses), and
L2 regularization toward the initial shipped affine head. Fixed-occupancy
cross-entropy is scaled by transcript length, matching the exact CTC loss
`−log P(transcript) / transcript_length`; it is not averaged over frames.
Both preservation terms average real frames. Features remain frozen and no
new head parameters or output labels are introduced.

A phase must reach a gradient infinity norm of at most 1e-5 before the next
expectation phase. Positive L2 regularization makes the fixed-assignment
linear-head problem strictly convex. The limit is 2,048 complete objective
evaluations per maximization phase; reaching that limit or a line-search
termination without stationarity raises an error and prevents export. It is
never recorded as a converged fitting phase.

The ordinary-CTC control uses the identical L-BFGS implementation, data,
initial head, recording/view weights, preservation terms, and L2 objective.
It computes its own exact CTC assignments at every complete objective
evaluation. Both arms receive a matched deterministic compute allowance:
the proxy counts feature projection/backpropagation work and CTC lattice
work, including the alternating arm's separate expectation pass. Rounding
remainders carry across phases and remain below one ordinary full-data
evaluation. A stationary control can stop with unused allowance. This is
an operation-count proxy, **not measured wall-time or exact hardware FLOPs**;
real compute parity remains to be checked when the service runs.

Six phases, regularization candidates 0.001, 0.01 and 0.1, and epoch zero are
specified before practice access. A shared phase and regularization minimize
the pair's mean selection objective, averaged equally across training-only
selection speakers. Thus target performance cannot select either checkpoint.
The solver enforces a one-day fitting deadline and checks it during expectation
passes, objective evaluations, and phase transitions.

## Data and export

`modal/student_alternating_train.py` exposes `prepare` and `fit`. Preparation
requires an explicit enrolled corpus of authorized, verified performed-phone
transcripts (speaker, split, phones, audio path and SHA-256), help-train
training speakers' none takes, an excluded-speaker list, and the shipped
noise/RIR banks. Scripted mistake takes and repeats are not supervised
transcripts. Speakers appearing in any non-training split in either corpus
are excluded globally before enrollment, as are explicitly excluded speakers.
Whole speakers are deterministically partitioned into fit and selection;
both partitions must contain both supervised source types.

Canonical reference duration repetitions are collapsed within each none
take's ayah, following the shipped transcript representation, while boundary
repetitions between ayahs are retained. Verified additional transcripts must
already describe performed articulation phones. Audio hashes reject duplicate
recordings. Both arms use the same three fixed views: original audio and two
views with the shipped waveform augmentation and SpecAugment recipe. Source
model, manifest, teacher assets, augmentation banks and cached views have
recorded hashes; fit checks source and cache identity before optimization.
The actual supplied corpus schema and enrollment availability remain unverified.

`modal/student_head_export.py` reuses the native feature/existing-head extraction
and complete bundle-copy safeguards from the curvature-head experiment. It
exposes the shipped encoder's pre-projection features without changing its
operations or streaming states. Quantized head weights are dequantized in
**both** arms; removing projection activation rounding is common to both and
cannot by itself establish an alternating-path benefit. The encoder remains
quantized. Export replaces the existing projection with its fitted fp32
weight and bias and prunes the unused old head.

Ordered alphabet, blank, frontend, sample rate, streaming window/shift,
40 ms output clock, tracker, checker, gap decoding and insertion verification
are inherited unchanged. Every existing teacher ONNX/external-weight member
is required, copied and checksum-verified. Preparation first checks exact
source output/state parity with feature extraction. Both fitted bundles must
reproduce their head posteriors on every selection view within 2e-4 probability
tolerance; `training-report.json` is written only after both pass.

Commands, once enrolled inputs are available through the authorized supervisor:

```sh
python3 modal/student_alternating_train.py prepare \
  --model /lab/data/models/spliced-clean-int8 \
  --help <training-help-labels.jsonl> --transcribed <enrolled-transcripts.jsonl> \
  --exclusions <release-excluded-speakers.json> --augmentation <shipped-bank-directory> \
  --out /scratch/listener-alternating-path-head-fit/prepared
python3 modal/student_alternating_train.py fit \
  --prepared /scratch/listener-alternating-path-head-fit/prepared \
  --out /scratch/listener-alternating-path-head-fit/fitted
```

The runner does not substitute practice audio or guessed performed transcripts
for missing training inputs. There is no `.autolab/model.json`; the shipped
listener would not evaluate this implementation.

## Validation and limits

All 18 generated unit tests passed locally with no skips:

```sh
.scratch/head-fit-venv/bin/python -m unittest discover -s modal -p test_student_alternating.py -v
```

These are tiny synthetic arrays/graphs, with no training audio or heavy Mac
computation. The environment uses NumPy 2.5.4, SciPy 1.18.1, ONNX 1.23.2 and
ONNX Runtime 1.31.1. Tests cover occupancy against exhaustive paths, repeated
labels, peaky long lattices, exact and fixed-path gradients against finite
differences, initial-gradient parity, immutable whole-corpus assignments,
stationary fitting and observed-loss improvement, compute-allowance accounting,
equal recording/view weights, budget/deadline failures, speaker separation,
none-take admission, global cross-corpus holdout exclusion, native float/int8
head extraction, encoder/state preservation and complete teacher copying.
Python compilation and whitespace checks pass.

All three supervisor requests failed before command execution with
`NotFoundError: No such file or directory`, each reporting USD 0.00:

1. Request 1: supplied data/model discovery.
2. Request 2: the required six-shard core/SDK suites via `bash tools/autolab_tests.sh`.
3. Request 3: retry supplied-data discovery plus the new synthetic tests.

The local synthetic suite also passed again after adding expectation-phase
deadline checks and source-manifest identity validation. Core/SDK suite results
are unavailable; their remote request never executed.

No training cache has been prepared, neither real-data head has been fitted,
and no complete real replacement bundle or full-session target comparison
exists. There have been zero practice-score invocations. Target wins/losses,
listener change and overall change are unknown. Infrastructure failure does
not falsify the premise and is not an acoustic kill.

When execution is restored, validate enrollment and the real native graph,
prepare the shared cache, fit/select both arms without practice access, and
verify their full bundles. Then compare full-session SDK-exported decodes at
the six named target spans against the matched ordinary control, measuring
previously missing planted-consonant recovery and actual target warnings.
The card names a two-span consonant floor; the user's final rules additionally
require at least three target wins before proceeding. Fewer than three wins
or a disproven premise requires rollback, a measured kill write-up and ignored
`outcome.json`. At least three wins requires the authorized practice score
before narrowing or stopping. The fold gate owns collateral-loss decisions.
Synthetic solver correctness establishes neither acoustic recovery nor safety.

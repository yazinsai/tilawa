# Implicit sequence-loss phone-head refit

Status: implementation and synthetic validation complete; acoustic evaluation
blocked by the supplied supervisor before execution. No replacement model is
declared. The experiment card is unchanged.

## Prediction and premise

Predict one recovered addition, listener −1 and overall −1, with no predicted
losses. This is inside the lab noise band and below the fold threshold. The
speculative premise is that explicit training steps destabilize weak-consonant
CTC emission paths. Earlier teacher recoveries do not establish this defect,
and numerical solver correctness does not establish acoustic recovery.

## Implementation

`modal/student_implicit.py` refits only the existing affine phone projection.
The encoder is an unchanged frozen ONNX graph; cached encoder outputs are
detached before evaluating the head. The objective for a recording is complete
marginalized transcript CTC, normalized by transcript length, plus KL
preservation against the shipped listener on every valid frame. Real speech and
pauses both contribute to preservation. Fixed views are averaged within a
recording; each recording receives one outer update per epoch in every arm.
Impossible transcripts fail explicitly rather than becoming zero CTC loss.

For preceding parameters θ₀ and outer interval τ, the implicit update locally
minimizes `L(θ) + ||θ − θ₀||² / (2τ)`. L-BFGS with strong-Wolfe line searches
evaluates the entire recording loss at proposed new parameters. The final
penalized objective must not increase, and the maximum stationarity residual
must be at most `1e-6 * max(1, initial maximum gradient)`. An unconverged solve
restores θ₀; nonfinite failures restore θ₀ and fail the run. No failed solve is
replaced with an explicit step. This is a bounded approximate local solve of a
nonconvex sequence objective, with no global-optimum or safety guarantee.

Each candidate has two explicit controls. Ordinary SGD takes one gradient step
of length τ. The multi-step control takes N explicit steps of length τ/N on the
same recording, where N is the implicit solve's actual number of complete loss
and gradient evaluations, including its final convergence diagnostic. Thus
each control covers the same outer interval, and the multi-step control spends
the same number of sequence-objective evaluations. The ledger records N per
recording, candidate and epoch. This matches objective/gradient compute, not
wall-clock time or L-BFGS vector-algebra overhead. All arms share initialization,
cached views, losses, recording weights, order and outer epoch count.

The predeclared constant proximal intervals are 0.003, 0.01 and 0.03, with eight
epochs and seed 1729. Whole training speakers are deterministically partitioned
into fit and selection sets by hashed speaker identity. A shared checkpoint
epoch and candidate interval minimize the three arms' mean complete selection
objective, averaged equally across selection speakers. Epoch zero is included.
Selection cannot consult practice labels, audio, target recoveries or scores.

## Data and export contract

`modal/student_implicit_train.py prepare` accepts the supplied help-train labels
and enrolls only scored training-speaker `none` takes with empty error,
non-error and unsupported lists. Repeats and ambiguous/excluded recordings are
excluded. Speakers appearing in any non-training split are excluded globally.
An optional explicit exclusion list and optional additional performed-transcript
JSONL can extend enrollment. Additional rows require identified speakers,
`split=train`, `authorized=true`, `verified=true`, phones, a contained audio
path and matching SHA-256. Non-training speakers in that corpus also exclude
their help takes. Audio hashes reject duplicates across the shared cache.

The supplied none takes are sufficient authorized training material; another
corpus is optional rather than an invented prerequisite. Canonical reference
duration/gemination runs are collapsed using the shipped spliced-target
convention before forming these takes' CTC targets. Explicitly enrolled
performed transcripts retain their supplied articulation sequence.

Without enrolled augmentation banks, each recording has one original view.
With the shipped noise and RIR banks, it has the original plus two fixed views
using the existing waveform and SpecAugment recipe. Bank and view hashes are
recorded. No practice audio, practice noise, synthetic splice or new channel
recipe enters preparation. `fit` reads only the shared cache and validates its
hashes and speaker partition.

`modal/student_implicit_export.py` reuses the native-head extraction/export
implementation from `exp/listener-curvature-aware-head-refit`. Quantized head
weights are dequantized in all three arms, and all three exported heads use
float32 affine projections on the unchanged quantized encoder. Removal of
activation rounding is therefore common to every arm and cannot demonstrate
the implicit-update premise by itself. The original head is pruned; no extra
head capacity is introduced.

Alphabet, frontend, streaming geometry, state outputs and 40 ms posterior clock
are inherited from the source manifest. SDK decoding, tracker, checker and
insertion verification are unchanged. Every insertion teacher graph/external
weight must be present, copied and checksum-verified. Missing or truncated
assets fail export. Preparation checks that exposing encoder outputs leaves
source posteriors and state outputs identical. Each exported arm must reproduce
its fitted head on all selection views within 2e-4 probability tolerance. The
training report is written only after all three complete bundles pass.

Commands for the supplied supervisor, once it can execute:

```sh
python3 modal/student_implicit_train.py prepare \
  --model /lab/data/models/spliced-clean-int8 \
  --help-labels /lab/data/help-train/labels.jsonl \
  --out /scratch/listener-implicit-sequence-head-fit/prepared
python3 modal/student_implicit_train.py fit \
  --prepared /scratch/listener-implicit-sequence-head-fit/prepared \
  --out /scratch/listener-implicit-sequence-head-fit/fitted
```

Both commands require the source bundle's unchanged insertion teacher. Optional
`--transcribed`, `--exclusions` and `--augmentation` refer only to enrolled
training inputs. The runner needs Torch, NumPy, ONNX, ONNX Runtime, torchaudio,
soundfile and SciPy when waveform augmentation is enabled.

## Validation and measured limits

All 17 synthetic unit tests pass, with no skips:

```sh
.scratch/implicit-venv/bin/python -m unittest discover -s modal \
  -p test_student_implicit.py -v
```

The local environment uses Torch 2.15.0, NumPy 2.5.4, ONNX 1.23.2 and ONNX
Runtime 1.31.1. Only tiny generated numerical/ONNX examples ran on the Mac.
Tests include exhaustive CTC path summation, a finite-difference CTC gradient,
the analytic implicit quadratic solution, independent CTC stationarity checks,
candidate-parameter evaluation, nonfinite/unconverged rollback, equal-view
weighting, matched recording/evaluation ledgers, shared checkpoint selection,
speaker exclusions, duration collapse, native quantized/float head extraction,
state preservation and complete auxiliary-bundle copying. Python compilation,
CLI parsing and whitespace checks also pass.

Three authorized supervisor requests failed before command execution with
`NotFoundError: No such file or directory`:

1. Run 1: supplied data/model/dependency inventory.
2. Run 2: required six-shard core/SDK suites via `bash tools/autolab_tests.sh`.
3. Run 3: retry supplied-data inspection plus the new unit tests.

No remote training inputs or source bundle were inspected. No real encoder
cache, fitted checkpoint or verified real-model bundle exists. Core/SDK suites
could not execute. There were zero practice-score invocations. Target wins,
losses, clean-session warnings, studio changes and overall/component deltas
are unknown. This is an infrastructure block, not a falsified acoustic premise
or a measured zero gain.

No `.autolab/model.json` is present. Running the main listener would not evaluate
this implementation. Real native-graph compatibility and the full SDK export
path remain unverified despite the synthetic contract tests.

## Remaining experiment decision

After training-only selection and verification of all three real bundles,
compare full-session exported decoding on the five card targets against both
explicit controls. Identify previously missing, correctly ordered planted
consonants and actual target warning gains. Neither schedule nor checkpoint may
be reselected from those observations. The card's consonant premise must pass;
the user's final rule also requires at least three target wins before practice
scoring. If that floor is reached, run the authorized practice score before
narrowing or deciding; collateral losses belong to the fold gate. If the premise
fails or fewer than three targets are won, revert engine changes, record the
negative result, and write the requested killed outcome. Infrastructure failure
does not justify that outcome. No fold or discard was performed.

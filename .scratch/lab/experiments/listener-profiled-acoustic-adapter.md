# Profiled acoustic adapter

Status: implementation and synthetic checks complete; authorized remote
execution blocked, acoustic result unmeasured. No replacement model is
declared. The experiment card is unchanged.

## Prediction and premise

Predict listener −1 and overall −1, with one recovered addition and no
predicted losses. This is inside the lab noise band and below the fold floor.
The speculative explanation is classifier–representation co-adaptation:
features that could distinguish weak consonants may not become useful when
the classifier and representation move together. Neither the history nor the
synthetic checks establishes that this is the remaining acoustic failure.

## Implementation

`modal/student_profiled.py` implements the residual representation
`z = x + U tanh(D x + b)`, followed by the existing affine phone classifier.
The bottleneck has 32 channels. Its output matrix starts at zero, so both arms
start at the same native affine classifier. For a 256-dimensional encoder and
43 classes, each arm has 27,467 trainable parameters: 16,416 adapter parameters
and 11,051 classifier parameters. The shipped encoder remains frozen.

The profiled arm approximately minimizes the **complete marginalized CTC**
classifier objective at every proposed adapter state with full-data L-BFGS,
warm-starting from the last fitted classifier. A head-gradient infinity norm
of at most 1e-5 is required before using that state. Each head fit is bounded
at 512 complete objective evaluations; an unconverged fit raises an error and
cannot be exported as a successful experiment. CTC is nonconvex even with
fixed features: stationarity does not establish a global classifier minimum.

The outer L-BFGS uses the envelope gradient through the optimized classifier
to the adapter. At a stationary head, the head's own optimization derivative
does not contribute to the first derivative of this same objective. No
fixed CTC alignment replaces the marginalized sequence objective. The joint
control optimizes the same adapter and classifier together with L-BFGS.

Both arms use the same initial parameters, fixed cached views, transcript CTC,
full teacher-posterior cross-entropy, shipped-output KL over every real frame
(including pauses), and L2 preservation toward the initial parameters. CTC is
normalized by transcript length; teacher CE and output KL average valid
frames. Recording weights and view weights are identical. Impossible
transcripts are rejected rather than zeroing their loss.

All closures compute the complete adapter and head gradients, including
head-only profiling closures. The joint control receives the same cumulative
number of complete corpus evaluations, including the profiled arm's final
envelope-gradient passes. A stationary control can stop with unused allowance.
This matches an operation-count proxy; actual hardware FLOPs and wall time
are not measured. Profiling is expensive and may exhaust the budget.

Four phases, twelve outer evaluations per phase, regularizations 0.001, 0.01
and 0.1, and the initial checkpoint are fixed before practice access. A shared
phase and regularization minimize the pair's mean selection objective,
averaged equally across selection speakers. Neither arm's target performance
selects its checkpoint. Preparation, fitting and export share a one-day
compute-time budget, with checks during cache preparation, optimization,
selection and export.

## Training data and export

`modal/student_profiled_train.py` exposes `prepare` and `fit`. Preparation
accepts the actual help-import label vocabulary and an explicit JSONL of
authorized performed-phone transcripts with speaker, `split=train`,
`authorized=true`, `verified=true`, phones, audio path and SHA-256. Only
scored, none-labelled help-train takes without errors, repeats, unsupported
labels or exclusions receive canonical transcript supervision. Duration
notation collapses within an ayah while preserving articulation repetitions
across ayah boundaries. No scripted mistake take is assigned canonical
supervision.

Any speaker occurring in a non-training split in either input corpus is
excluded globally, along with the explicitly supplied release-exclusion list.
Whole speakers are partitioned deterministically into fit and selection;
both groups must contain both supervised source types. Audio identity and
duplicates are checked. The supplied corpus inventory and availability of
enrolled additional transcripts remain unverified because discovery never
executed.

Both arms use the same original view and two fixed views drawn from the
shipped waveform/SpecAugment recipe. Preparation caches native streaming
encoder features, full teacher posteriors, shipped posteriors and the
performed transcript. Source, manifest, teacher assets, augmentation banks
and cached views have identity hashes. Fit checks those identities before
optimization and reads no practice recordings or labels.

`modal/student_profiled_export.py` reuses the native-feature and bundle-copy
contracts from `exp/listener-alternating-path-head-fit`. It replaces the
existing projection with the adapter and fitted head inside the ordinary
ONNX graph; the SDK needs no new decoding interface. The encoder, streaming
states, ordered alphabet, blank, frontend and 40 ms output clock are inherited
unchanged, as are tracker, checker, gap decoding and insertion verification.
The native quantized classifier weights are dequantized in **both** arms;
removing its activation rounding is common to both and cannot establish a
profiling-specific gain relative to shipped decoding.

Export requires, copies and checksum-verifies the existing insertion teacher
and all its external weights. It rejects incomplete source bundles rather
than disabling verification. Both selected arms must reproduce their Python
head distributions on every selection view within 2e-4 probability tolerance
before `training-report.json` is written. No real bundle was exported here.

Commands for the authorized supervisor once enrolled inputs are available:

```sh
python3 modal/student_profiled_train.py prepare \
  --model /lab/data/models/spliced-clean-int8 \
  --help <help-train-labels.jsonl> --transcribed <authorized-transcripts.jsonl> \
  --exclusions <release-excluded-speakers.json> --augmentation <shipped-banks> \
  --out /scratch/listener-profiled-acoustic-adapter/prepared
python3 modal/student_profiled_train.py fit \
  --prepared /scratch/listener-profiled-acoustic-adapter/prepared \
  --out /scratch/listener-profiled-acoustic-adapter/fitted
```

These placeholders are input enrollment requirements, not discovered files.
No `.autolab/model.json` was written: running the main listener would not
evaluate the adapter implementation.

## Validation and limits

All 19 generated unit tests passed locally without skips:

```sh
.autolab/profiled-venv/bin/python -m unittest discover -s modal -p test_student_profiled.py -v
```

Only tiny synthetic arrays and ONNX graphs were used on the Mac. The local
environment used NumPy 2.5.4, SciPy 1.18.1, ONNX 1.23.2 and ONNX Runtime
1.31.1. Checks cover exhaustive CTC likelihood/occupancy including repeats
and blank, peaky lattices, every parameter gradient against finite differences,
the envelope gradient against independently reoptimized head minima,
nonstationary-head rejection, deadlines and evaluation budgets, equal
recording/view weights, unchanged observations, shared capacity and compute
allowance, fit-only optimizer access and speaker-only checkpoint selection.
Enrollment checks cover none-take admission, global cross-corpus exclusion,
duration notation, audio hashes, cache hashes and corpus containment. ONNX
checks cover native float/int8 projection extraction, nonlinear posterior
parity, preserved encoder/state updates, old-head removal, incomplete teacher
rejection and complete bundle copying. Python compilation and whitespace
checks pass.

Three authorized supervisor requests failed **before command execution** with
`NotFoundError: No such file or directory`, each reporting USD 0.00:

1. Request 1: supplied data/model and Python environment discovery.
2. Request 2: required six-shard core/SDK build and test harness,
   `bash tools/autolab_tests.sh`.
3. Request 3: corrected discovery command and the synthetic unit suite.

No alternate Modal access was attempted. Core/SDK suite results are unavailable.
There is no real feature cache, fitted adapter/head, complete replacement
bundle, SDK full-session target comparison or practice score. Practice-score
invocations: **zero**. Target wins, listener change and overall change are
unknown, not zero. Infrastructure failure does not falsify the acoustic
premise, so no killed outcome is asserted and the implementation remains.

The next empirical step is to prepare the training-only cache, fit/select both
arms and verify both complete bundles. Then compare full-session SDK decoding
at the six registered target spans, measuring previously missing planted
consonants in order relative to **both** shipped and joint decoding, and
separately measuring confirmed target warnings. The card requires at least
two such consonant recoveries; the user's final instructions additionally
require three target wins to proceed. A failed acoustic premise or fewer
than three wins requires rollback and a measured killed outcome. Three or
more wins requires the authorized practice score before narrowing the change.
Population losses belong to the fold gate. No training result, target gain,
clean-session safety, studio preservation or device readiness is established
by the synthetic tests.

No scorer, lab tooling, test data, supplied manifest, evidence or card edits
were made. No fold was performed.

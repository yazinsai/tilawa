# Listener temporal rank distillation

Status: implemented and synthetically validated; acoustic experiment blocked and unmeasured. Recorded 2026-11-12. No model is declared for scoring.

## Prediction and mechanism

The card predicts listener −1 and overall −1 wrong calls, with substantial zero-gain risk. Probability-matching distillation may transfer confidence poorly while losing the temporal ordering of weak consonant evidence. An auxiliary ordinal objective on the frozen encoder's phone head tests that hypothesis. The forecast lies within the stated noise band and below the fold gate.

`modal/student_temporal.py` marginalizes transcript-constrained CTC paths in float64, retaining separate state identities for repeated phone occurrences. An occurrence's acoustic neighbourhood runs from the preceding occurrence's posterior median through the following occurrence's median, including intervening blank/pause frames. Edge occurrences extend to the recording edge. The neighbourhoods are constructed independently for two augmented teacher views and intersected on their shared native student frame clock.

Within each neighbourhood, a comparison is eligible only when the same phone's teacher log-probability has the same strict frame ordering in both views. Ties and disagreements are rejected. Frame-pair SHA256 hashes deterministically select at most 64 comparisons per occurrence; probability magnitude does not affect sampling, loss margins or weights. Pairwise logistic loss penalizes reversed student order, averaging comparisons within an occurrence and then occurrences within a recording. This does not copy teacher probability values or select a single teacher emission path. It still encourages the teacher's local peak ordering and can transfer correlated teacher errors.

The ranking arm uses transcript CTC plus ranking and shipped-output KL preservation. The original unaugmented view retains CTC and preservation; ranking uses the two augmented views whose teacher ordering was checked. The ordinary-distillation arm uses full-distribution teacher cross-entropy instead of ranking; the CTC-only arm uses CTC and the shared preservation loss. All three compute the same objective graph, with unused auxiliary terms weighted zero, and share cached data, encoder, initialization, recording order, SGD schedule and update count. Thus additional ranking computation also occurs in the controls.

The trained head is a zero-initialized affine correction to the installed projection. The quantized source encoder and projection remain intact; the correction is added to its log-probabilities before LogSoftmax. This retains the source's quantization behavior at initialization while allowing a phone-head refit. No tracker, checker, insertion-verification or decoding rule changes.

## Training admission and frozen selection

`modal/student_temporal_train.py` provides `prepare` and `fit` commands. Preparation requires:

- A complete shipped model bundle, including the unchanged insertion teacher and its external weights.
- Help-train labels: consistently training speakers' scored `none` takes with empty events, non-errors and unsupported lists, and no exclusion flag.
- An explicitly enrolled authorized performed-transcript JSONL: identified speaker, `split=train`, `authorized=true`, `verified=true`, phones, relative audio path and matching SHA256.
- An excluded-speaker list and the existing noise/RIR augmentation banks.

Any speaker appearing in a non-training split in either corpus is excluded across both corpora. Audio paths must remain inside their enrolled corpus, duplicate audio is rejected, and every phone must belong to the unchanged inventory. Transcript targets use the shipped CTC recipe's adjacent-run collapse: repeated reference symbols denoting held duration do not become artificial CTC-separated articulations.

Speaker hashing assigns one in five training speakers to checkpoint selection; all takes from one speaker stay together. Both fit and selection partitions must contain both corpora. Preparation caches the original take and two deterministic augmentation draws using the shipped reverb, noise, gain and student SpecAugment distributions. The teacher sees each draw's waveform through its installed SDK frontend; student feature masks are applied in its own frontend. Complete teacher rows remain available for alignment. Ordinary KD uses the shared native prefix; student CTC and preservation retain every real output row, including pauses.

Pairing rules, audio/transcript identities, source model, teacher assets, augmentation banks and cached views are recorded or hashed. A changed pairing rule invalidates the cache. The fixed recipe is seed 1729, 12 epochs, learning rate 0.02, plain SGD without momentum and unit auxiliary/preservation weights. Each arm independently selects its lowest equal-speaker CTC-plus-preservation objective from epochs zero through twelve, with earliest checkpoint winning a tie. Selection never uses ranking loss, practice labels or target results. Preparation plus the three refits share an enforced 24-hour compute budget.

Actual corpus enrollment, the release-speaker exclusion audit and retained real speech/pause coverage could not be checked because the supervisor probes never executed. These requirements are implemented, not claimed as completed data validation.

## Export and validation

`modal/student_temporal_export.py` exposes pre-quantization encoder features for caching, then grafts the selected affine head into a separate bundle. It preserves source nodes, weights, streaming state inputs/outputs, alphabet and clock fields. It copies and checksum-verifies auxiliary members, rejecting missing or truncated teacher weights. New manifests describe generated models only; source manifests are never rewritten. The fit command checks each optimized exported graph against cached head predictions on checkpoint-selection recordings before completing its training report.

The cache/export infrastructure was adapted from `exp/listener-conflict-filtered-distillation`; the CTC marginalization was adapted from `exp/listener-duration-stratified-distill`. The synthetic fit test caught and fixed an inherited parity-path bug: loaded view dictionaries had replaced filenames before export validation.

Validation completed:

- Python compilation and `git diff --check` pass.
- **26 synthetic unit tests pass**, including exhaustive CTC path enumeration, repeated occurrence separation, full-teacher/shared-student edges, agreement/tie filtering, confidence-scale-independent pairing, deterministic comparison limits, equal-occurrence weighting, ranking gradients, exact objectives for all controls, detached evidence, and full-stream CTC/pause preservation.
- Admission tests cover speaker grouping, cross-corpus exclusion, authorization/checksums and duplicate audio. Selection tests cover equal-speaker weighting and read-only evaluation.
- Floating-point and dynamically quantized ONNX tests cover frozen source weights/states/clock, zero-head parity, optimized export parity and complete external teacher-file copying. A tiny synthetic cache runs all three refits, checkpoint selection and complete exports. These generated fixtures are not acoustic training or a practice measurement.

Local test command:

```sh
.scratch/temporal-venv/bin/python -m pytest modal/test_student_temporal.py -q --basetemp .scratch/temporal-test-tmp
```

Environment: Python 3.12.9, Torch 2.2.2, NumPy 1.26.4, ONNX 1.23.2 and ONNX Runtime 1.23.2. The supplied training image uses different pinned versions, so remote validation remains necessary.

## Supervisor blocker and stopping rule

All three requests through the supplied supervisor failed before command execution with `NotFoundError: No such file or directory`, reporting zero spend:

1. Available data, dependencies, model members and storage probe.
2. Training-image probe of authorized sources and their schemas.
3. Required six-shard `bash tools/autolab_tests.sh` run.

The core/SDK suites have not run. No real audio has been cached, no acoustic head trained, no complete real model bundle exported, and no target session decoded. No practice score has been requested (zero of three allowed calls used). There is no `.autolab/model.json` or killed outcome. Running the main listener would not evaluate this implementation.

The requested diagnostic sites are p07 `37:36:1` at 144.12 s, p33 `26:208:1` at 156.16 s, p48 `12:76:1` at 39.98 s, p56 `32:21:1` at 106.76 s, and p57 `51:18:1` at 95.26 s. These identifiers occur only in this report, not in training or export code.

After restoring the supervisor, resolve the admitted corpus paths, prepare one shared cache, fit the three arms, and verify their complete real bundles. Compare the training-selected exports' full-session streams against both controls, recording restored planted consonants separately from recovered warnings. The card requires at least two consonant recoveries relative to both controls; the user's final rule additionally requires at least three target wins before practice scoring. Neither threshold has been measured. Infrastructure failure does not falsify the acoustic premise and is not recorded as zero wins or a kill.

If the acoustic stopping rule fires, remove the engine changes and record the killed result. If at least three targets are won, declare only the fully verified rank bundle under the experiment scratch directory and request practice scoring before deciding whether to narrow. A new warning on a fully clean session is a hard stop; other losses are assessed by the fold gate rather than used for an early kill.

Temporal pairing and an unchanged encoder do not guarantee recovery or preservation. Teacher agreement is correlated supervision, and new head emissions can create false warnings. Target wins, listener/overall changes, studio behavior, clean-session warnings and reader robustness remain unmeasured.

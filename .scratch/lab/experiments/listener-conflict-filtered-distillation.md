# Conflict-filtered distillation of the frozen listener's phone head

Status: implementation and local unit tests complete; acoustic experiment blocked and unmeasured. No model is declared for scoring. Recorded 2026-11-11.

## Prediction and stopping rule

The card predicts listener −1 and overall −1 wrong calls on lab-v1.5u practice: transcript-supported weak consonants may be suppressed by conflicting teacher-imitation updates. That prediction is within the stated noise band and below the fold gate's required overall gain.

The card's early check requires at least two target full-session decodes to restore a previously missing planted consonant relative to an identical-data ordinary-distillation control. The user's final instruction additionally requires at least three target wins before practice scoring. Neither check has been measured. No practice score has been requested, and no zero-gain or killed acoustic result is claimed.

The five requested diagnostic sites are p07 `37:36:1` at 144.12 s, p48 `12:76:1` at 39.98 s, p56 `32:21:1` at 106.76 s, and p57 `51:18:1` at 95.26 s and `51:20:1` at 112.26 s. These identifiers occur only in this report; the training and export code has no practice-specific routing or selection.

## Implemented mechanism

`modal/student_conflict.py` computes transcript CTC, full-distribution teacher cross-entropy, and shipped-posterior preservation separately. For each recording it averages its original view and two fixed augmented views, then differentiates each objective with respect to the complete affine head, including both weights and bias. With transcript gradient `g` and distillation gradient `d`, a negative inner product replaces `d` with `d − (d·g)/(g·g) g`. Agreeing and orthogonal gradients are retained; a zero transcript gradient retains distillation unchanged. Projection accumulates in float64. CTC and preservation gradients are added unchanged.

The head is a zero-initialized affine correction to the installed phone logits, equivalent to an affine head refit while retaining the quantized shipped projection. Cached encoder features, installed logits and teacher targets are detached. The encoder, ordered 42-phone alphabet, 40 ms output clock, streaming states, SDK decoding, tracker and checker retain their existing implementation.

`modal/student_conflict_train.py` provides `prepare` and `fit` commands. Enrollment requires an explicit excluded-speaker list, training-speaker help-none takes with empty event/non-error/unsupported lists, and a separately enrolled authorized, verified performed-transcript JSONL with audio checksums. Any speaker appearing in a non-training split in either corpus is excluded from both. Duplicate audio is rejected. Both fit and checkpoint-selection partitions must contain both corpora. These admission requirements are implemented but have not been checked against the available real-data schemas.

Preparation caches the original recording and two views using the existing reverb, noise, gain and SpecAugment distributions. No waveform splices or phone relabels are introduced. The installed teacher uses its SDK frontend: PCM scaling, Kaldi fbank to Nyquist, unbiased per-band variance normalization and frame pairing. KD uses only the native shared frame prefix; complete transcript CTC and preservation retain all student rows. Source, teacher, audio, transcript, augmentation-bank and cached-view hashes are recorded.

Both arms start at the shipped head and use the same recordings, views and shuffled recording order. The frozen recipe is seed 1729, 12 epochs, learning rate 0.02, and unit KD/preservation weights, with plain SGD and no momentum. Speaker hashing assigns approximately 20% of the enrolled training speakers to checkpoint selection. Each arm independently selects the lowest equal-speaker CTC-plus-preservation objective, including epoch zero. Practice data and release speakers are absent from that selection path.

`modal/student_conflict_export.py` exposes pre-quantization encoder features and appends the chosen affine correction to the shipped ONNX outputs. It preserves native nodes, initializers, state inputs/outputs and manifest clock/vocabulary fields. It generates a new manifest for the new graph in a separate directory and copies/checks the existing teacher graph and external weights. Missing or truncated teacher members stop export. The fit command also checks the optimized full graph against cached head predictions on checkpoint-selection speakers before recording export parity.

## Validation completed

Python compilation passed for all four new Python files. Local synthetic unit tests passed: **23 tests**, using Python 3.12, Torch 2.2.2, NumPy 1.26.4, ONNX 1.23.2 and ONNX Runtime 1.23.2.

The tests cover agreeing/orthogonal/zero/conflicting gradients, joint weight/bias projection, per-recording ordering, ordinary-control equivalence to autograd, detached evidence, complete CTC at teacher edges, speaker partition stability, cross-corpus exclusion, transcript authorization/checksums, duplicate audio, equal-speaker selection metrics and read-only evaluation. Synthetic floating-point and dynamically quantized ONNX graphs exercise feature extraction, preservation of source weights and streaming states, the output clock, selected-head parity, zero-head parity and complete external teacher-asset copying.

Local command:

```sh
.scratch/conflict-venv/bin/python -m pytest modal/test_student_conflict.py -q
```

## Supervisor blocker

All three requests through the supplied supervisor failed before command execution with `NotFoundError: No such file or directory`, each reporting zero spend:

- Run 1: inspection of available data, dependencies and storage.
- Run 2: the required six-shard `bash tools/autolab_tests.sh` run.
- Run 3: a retry listing authorized training inputs and shipped bundle members.

No alternate Modal route or external training/exam files were accessed. The core/SDK suites have not run. No real recording has been prepared, either head trained, complete real bundle exported, or target session decoded. There is no `.autolab/model.json` or killed outcome. Scoring the main model would not evaluate this training change.

## Work remaining and limits

Restore the supplied supervisor, inspect the authorized corpus and its enrollment schema, audit the excluded speakers and retained speech/pause examples, prepare one shared cache, fit both arms, and verify both complete exported bundles. The training CLI requires `--model`, `--help`, `--transcribed`, `--exclusions`, `--augmentation` and `--out` for `prepare`, followed by `--prepared` and `--out` for `fit`. Their real input paths remain unresolved because the inspection commands never executed.

Compare the training-selected arms on the requested full sessions without selecting checkpoints from their results. Apply the two-restored-consonant premise check and the user's three-target-win floor. If the premise fails, remove the engine implementation and record a killed result. If at least three targets are won, declare the fully verified filtered bundle under the experiment scratch directory, run the required core/SDK suites and request practice scoring before deciding whether to narrow the change.

The local tests establish the update geometry and synthetic export behavior. They provide no evidence that real recordings contain the proposed destructive supervision, that native encoder features preserve the missing consonants, or that an affine head can recover them. Finite SGD steps and the independent preservation gradient can still trade off transcript accuracy. Retained speech and pause examples and an unchanged encoder do not establish safety on clean sessions or studio audio. Listener/overall changes, target wins, clean-session warnings, losses and reader robustness are all unmeasured.

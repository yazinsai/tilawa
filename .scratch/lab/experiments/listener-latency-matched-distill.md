# Occurrence-preserving temporal distillation

Status: training-loss implementation and unit validation complete; acoustic
premise, matched training, export and practice evaluation blocked by the supplied
execution service. This is an incomplete, unmeasured experiment, not a killed
hypothesis or a scoreable new listener. The experiment card is unchanged.

## Prediction

The card predicts three recovered additions and one collateral loss: listener
−2 and overall −2 on lab-v1.5u. The six registered targets are used only for the
future premise audit and evaluation. No target audio, labels or evidence have
entered calibration or training. No target-win count has been measured.

## Implemented mechanism

`modal/student_temporal.py` implements a normalized forward sum over monotone
teacher-to-student timing correspondences. Each teacher row contributes its
complete original renormalized top-k distribution exactly once. Corresponding
rows must belong to the same transcript-phone occurrence under detached CTC
forced alignments and lie within the training-calibrated displacement bounds.
Repeated identical phones retain separate transcript occurrence identities.
Alignment blanks belong to the following occurrence, with trailing blank
belonging to the last occurrence.

For a correspondence z, its cost is the sum over teacher rows t of
CE(p_teacher[t], p_student[z[t]]). The KD numerator is
`log(number of legal correspondences) - logsumexp(-cost(z))`.
The dynamic program sums every legal nondecreasing correspondence without
enumerating them. Its gradient supplies the posterior mixture of transported
teacher targets. Every source row's gradient weights sum to one, so neither a
phoneme occurrence nor its blank/alternative-phone mass is duplicated or
removed. Several source rows may use the same destination row; their weights
are added without destination renormalization. The correspondence prior is
normalized so merely widening the timing band does not buy evidence through
path multiplicity. Division by total teacher rows retains the mean KD scale.
With equal lengths and only identity correspondence, values and gradients match
the existing framewise loss exactly.

`modal/student_train.py` now accepts programmatic `kd_mode="temporal"` with
`target="reference"`, nonzero original `kd_weight`, zero `kd_delay`, and a
`temporal_calibration` JSON path. It uses transcript CTC and the new KD term,
retaining the existing augmentation, optimizer, model and export architecture.
The default remains ordinary framewise KD. The existing training image packages
the new helper. A matched continuation can use exactly the same data, target
source, schedule, checkpoint and KD weight with `kd_mode="framewise"`.
No matched continuation has actually run.

The module also provides the training-only calibration building blocks:

- Exact greedy emissions are edit-aligned to transcript occurrences. A missing
  or substituted phone has no measured emission time; forced alignment cannot
  turn disappearance under lookahead restriction into a delayed emission.
- The teacher restriction recomputes inference on the actual PCM prefix
  available at each deployed student chunk deadline. It includes the fbank
  analysis window, limiting future-dependent teacher normalization and
  convolutions as well as attention. Rows retain their original grid.
- For the deployed 45-feature-frame window and 32-frame advance, with a 10 ms
  hop and 25 ms analysis window, the first eight 40 ms output rows have an input
  deadline of 7,440 samples at 16 kHz; the next eight have 12,560 samples.
- Calibration records retain the ordinary exact-occurrence delay envelope and
  form the permitted transport envelope from ordinary student/teacher delays,
  restricted/full-teacher shifts and identity timing. Run centres define the
  delays. No empirical envelope was fabricated here.
- Training entry requires a training-only record identifying model and data
  hashes, nonzero ordinary and restricted measurement counts, and matching
  output/receptive-field geometry. These checks establish record completeness,
  not corpus authorization: its producer and training-speaker exclusions still
  require an actual data audit.

A transcript with no complete alignment, or a source row with no legal
mass-conserving correspondence, raises an error. There is no edge clamping,
framewise fallback or silent example removal. The hard occurrence partition is
an approximation: timing within its legal correspondence is marginalized, but
alternative CTC occurrence partitions are not. Its effect on weak consonants
and the feasible training population remains unmeasured.

## Validation

Read AGENTS.md and `docs/agents/audio-review.md`. Static inspection confirmed
that the original KD uses clamped frame indices and has no occurrence-dependent
transport. That is a code observation, not evidence that the target misses are
caused by timing. No independent listening judgment is claimed.

Twenty unit tests passed on tiny CPU tensors in an isolated environment under
this worktree, using Python 3.13.7 and torch 2.8.0. The command was:

```
.scratch/temporal-test-env/bin/python -m unittest discover -s modal -p test_student_temporal.py -v
```

Coverage includes exact repeated-phone occurrence identity, impossible repeats,
disappearing emissions, training-only delay construction, deployed PCM deadlines,
restricted inference on the original grid, and refusal to clamp or drop source
rows. Numerical tests compare the dynamic program with exhaustive legal maps,
check each teacher row's gradient mass, rule out a path-count advantage, reject
crossing-only maps, and preserve total teacher mass with unequal durations.
The actual production `losses()` function is also tested against the matched
framewise control for equal CTC, KD, total loss and gradients, including padded
batch rows. Its definition is loaded without the module-level Modal image and
volume declarations; only CUDA autocast is disabled for this tiny CPU test.

The isolated local environment is ignored and used for these small unit tests
only. Acoustic inference, corpus analysis and training were not run on the Mac.
Python compilation and `git diff --check` passed. No existing tests, timeouts,
scoring rules, test data, supplied manifests or evidence were modified.

## Execution service record

All remote requests used the user-supplied supervisor. No alternate Modal
access, credentials, exam access or outside-worktree data search was attempted.

The earlier attempt requested runs 1–4: CPU and training-image inventories,
`pwd`, and the six-shard core/SDK harness. All returned
`NotFoundError: No such file or directory` without container output. The earlier
pending run 2 also resolved with this error, charging $0.752.

This attempt retried independently of corpus or implementation assumptions:

| Run | Request | Result |
| --- | --- | --- |
| 5 | CPU, `pwd`, 3 minute timeout | Startup NotFoundError; no command output |
| 6 | L4/train, inventory of supplied checkpoints, model directories, help-train and torch, 5 minute timeout | Startup NotFoundError; no inventory or dependency output |
| 7 | Six shards, `bash tools/autolab_tests.sh`, 5 minute timeout | Startup NotFoundError; no build/test output |

Run 6 charged $0.1253; the last result reports GPU spend $0.88 and remaining
budget $239.12. Runs 5 and 7 charged no actual GPU cost. All requests have
finished. Core and SDK regression suites have therefore been requested but
cannot be reported as passing. These failures occur before training data or
the new loss is evaluated.

## Why the experiment cannot be completed here

The supplied execution service still cannot start a container, even for `pwd`.
The authorized training audio, initial checkpoints and practice audio are
available only inside those containers. There is consequently no way here to
measure an ordinary delay range, restrict the teacher on training recordings,
audit the six targets, or run either matched training arm. Local unit tensors
cannot substitute for that evidence. Guessing calibration bounds or declaring
a copy of the main model would not evaluate this experiment.

The implementation is committed for review, but there is no trained checkpoint,
SDK export, `.autolab/model.json`, or incomplete declared bundle. No practice
score was requested; all three opportunities remain available. A default-model
score must not be treated as evaluation of this loss change. Infrastructure
failure is not the card's acoustic kill criterion, so no killed outcome or
fabricated zero-target result is written.

After service restoration, the remaining work is to audit the original
training mix and help-train training speakers' none-labelled takes, verify
release-held-out exclusions, produce the training-only calibration, and test
the registered premise. Fewer than three targets with a missing teacher-retained
consonant shifted beyond the ordinary range under restriction kills the premise
and requires reverting the engine changes. Otherwise run identical-data
framewise and temporal continuations, select checkpoints on training-speaker
partitions, export and verify the complete primary SDK bundle with the existing
inventory and clock, and evaluate its fresh target streams. Fewer than three
built-model target wins also kills the experiment. Three or more require a
practice score before any narrowing; the supervisor fold gate decides whether
the measured gain and losses are acceptable.

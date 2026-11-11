# Complete CTC sequence-distribution phone-head distillation

Status: blocked before acoustic validation. This is an unvalidated implementation,
not a completed or killed experiment. No replacement model has been declared.

## Prediction and premise

The card predicts listener −1 and overall −1 on lab-v1.5u, with substantial
zero-gain risk. The proposed cause is loss of correlated, ordered consonant
alternatives during framewise teacher distillation.

The premise requires at least two of the five named targets to contain the
complete planted-word phone sequence in the teacher's retained transcription
distribution but not the student's, on the same engine-selected interval. It
must be checked after fixing candidate bounds on training speakers. Retaining
individual consonants, or using the planted timestamp to choose an interval,
would not establish this premise. That audit has not run.

## Implementation prepared

`modal/student_sequence.py` generates bounded acoustic-only CTC prefix beams
with separate blank and nonblank states. It rescores each retained complete
transcription with an exact CTC forward sum, including repeated-phone blank
requirements. Teacher and current student candidates form a bounded union.
There is no transcript, Quran lexicon or language model in this proposal path.

The differentiable loss compares the complete-sequence probabilities and their
complementary OTHER mass. It does not renormalize a tiny retained set as if that
set contained all teacher evidence. Teacher and student retain their own valid
frame counts; sequence supervision does not align their emission timing.

`modal/student_sequence_train.py` prepares frozen native encoder features from
authorized, hash-verified performed transcripts and strictly admitted help-train
training-speaker none takes. Repeat, ambiguous, excluded and nontraining takes
are rejected. Nontraining speakers discovered in either enrolled corpus are
excluded globally. Each recording has an original and two deterministic shipped
augmentation views. Full-utterance transcript CTC and preservation KL include
both speech and pause frames.

The existing native affine phone head is initialized identically in the two
arms. Only the KD unit differs: complete-sequence KL versus full framewise KL.
Both arms share data, view weights, recording order, optimization and selected
update count. Hash-based whole-speaker fit/selection partitions are training
only. A separate `limits` command records candidate bounds from the registered
8/16/32 grid before premise audits, using speaker-balanced retained teacher mass.
Checkpoint selection uses speaker-balanced phone error and a common sequence
validation objective, with epoch zero eligible. Preparation, limit selection and
refitting account for a combined one-day CPU budget.

`modal/student_sequence_export.py` exposes the native feature tensor and replaces
only the existing phone projection. It inherits the inventory, input geometry,
streaming state and clock. Complete bundle export requires and checksum-verifies
the unchanged insertion teacher and its external weights. Both arms must pass
native inference parity before a final training report is written. The SDK's
existing directory loader can load these generated bundles. Tracker, checker,
insertion verification and SDK acceptance rules have no edits.

The exporter and enrollment/frontend scaffolding reuse the prior
`exp/listener-curvature-aware-head-refit` implementation. The training runner
also fixes its stale parity-loop cache-field reference and retains the teacher's
entire valid output extent instead of truncating it to the student's frame count.

## Validation and infrastructure limit

Local command:

```
python3 -m unittest discover -s modal -p 'test_student_sequence*.py' -v
```

Seven tests pass. They exhaustively enumerate tiny CTC frame-path distributions,
check repeated phones, exact beam ranking, ordered candidate unions, invalid
posterior rejection and whole-speaker enrollment exclusions. Ten Torch/ONNX
tests are skipped because those dependencies are absent locally. Those tests
cover exact OTHER mass, gradients against exhaustive enumeration, timing,
frozen evidence, matched update counts, native head extraction and complete
bundle/state contracts. Skips are not validation passes.

All five new Python files compile. No acoustic computation ran on the Mac.

The supplied supervisor returned `NotFoundError: No such file or directory`
before execution for run 1 (data/model probe), run 2 (the required six-shard
core/SDK tests), run 3 (a minimal `pwd` probe), and run 4 (the new numerical and
export tests). All four result records report zero expenditure and no command
outputs. The runtime-access failure has been reported to the user. No remote
jobs remain pending.

No training assets have been inspected remotely, no candidate bounds have been
selected, and no premise coverage or target warning recovery has been measured.
There are no practice scores, selected checkpoints or verified model bundles.
`.autolab/model.json` is absent. Scoring the main model would not evaluate this
implementation. Infrastructure failure does not justify a killed outcome or a
claim of zero recovered targets.

## Remaining work

After supervisor access is restored, run the numerical/export tests and required
core/SDK suite. Discover and enroll the authorized training assets, prepare
caches, and record training-only candidate limits. Audit the five targets using
generic engine-selected intervals; stop and revert engine changes only if the
premise fails. Otherwise train and export both matched arms, verify the complete
bundles and compare target verdicts through the unchanged SDK. Fewer than three
target wins triggers the builder's kill rule. Three or more requires a practice
score before the fold gate decision. No scoring allowance has been consumed.

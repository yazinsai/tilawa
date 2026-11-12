# False-sequence unlikelihood head refit

Status: source implementation committed; training and evaluation blocked before
execution by the authorized Modal supervisor. Unmeasured. The experiment card is
unchanged. There is no selected model or declared model pointer.

## Prediction and mechanism

The card predicts listener −1 and overall −1 on lab-v1.5u. Freeze the shipped
streaming encoder and explicitly reduce the probability of complete erroneous
phone transcriptions that cause wrong-word accusations on clean help-train
recordings. This changes the phone head's learned readout.

For a mined sequence y and its original acoustic interval x, the negative loss is
`−log(1 − P_CTC(y | x))`. CTC sums every complete path for that specific
transcription. It does not penalize a list of consonant classes. Adjacent repeated
phones retain the CTC requirement for a separating blank; duplicate transcriptions
on the same interval contribute once.

Compare with a positive-only arm initialized from exactly the same shipped head.
Both use the same admitted recordings, fixed batch orders, update counts,
canonical transcript CTC on clean takes, and KL anchoring to shipped posteriors
on every real frame. Mistake, repeat and unsupported takes provide output
preservation only; their written reference is never treated as a performed
transcript. Real pauses remain in both arms.

The task's final stopping rule requires at least three of the four target
confirmations removed on complete streaming replay relative to the matched
positive-only head. Fewer than three wins or a false premise kills the experiment;
at least three wins requires an official practice score. This supersedes the
card's narrower zero-win criterion. Infrastructure failure cannot establish
either target count.

## Implemented training and export path

`packages/sdk/src/training/mine-sequence-cli.ts` replays the unchanged SDK free
recitation engine on clean training takes. Its helper excludes an entire speaker
if any source take is outside the training split or explicitly excluded. Before
mining, it reserves the first fifth of admitted speakers in SHA256 order for
training-only validation. Positive coverage comes exclusively from fit speakers.

A negative must be a finalized wrong-word warning inside the labelled ayah,
incompatible with existing lawful pronunciation variants, with a unique
contiguous complete-token witness in the original SDK evidence. It must also
have another clean fit take whose canonical CTC transcript contains the targeted
pronunciation. This stricter positive condition avoids counting a partial or
isolated form absent from the positive transcript as coverage. The implementation
contains no practice incident, reader or verse selection.

The CLI caches the actual SDK frontend's float32 fbank rows, including the
streaming resampler tail, alongside audio, label, reference and feature hashes.
It verifies primary and inherited auxiliary model assets before replay, so a
missing insertion teacher cannot silently change the training accusations.

`modal/student_sequence_unlikelihood.py` rechecks immutable source labels,
whole-speaker exclusion, deterministic partition, every recording/frontend hash,
and positive-coverage provenance before freezing its plan. It captures actual
dequantized activations consumed by the shipped integer phone head, checks
reconstruction of shipped log probabilities, and removes padding rows that the
SDK does not emit. Both arms share frozen encoder features and posteriors.

The fixed recipe is seed 831, 12 epochs, batch size 8, SGD learning rate 0.0001,
preservation weight 2 and negative loss weight 1. These were fixed before any
data or target replay. Epoch selection minimizes the unlikelihood arm's
speaker-balanced reserved-training objective; the control is taken at that same
epoch and update count. Selection records initial negative sequence mass, both
validation histories and selected checkpoint hashes.

`modal/student_head_refit.py` exports both selected heads by replacing classifier
constants only. It retains shipped quantization geometry, checks encoder tensor
and graph preservation, tests exact stream-cache output parity, and copies/checks
every inherited declared auxiliary file. Generated manifests inherit vocabulary,
architecture, timing and capability fields. No supplied manifest, tracker,
checker or warning rule is edited.

The training source implemented here is authorized help-train, including all
admitted clean, mistake and repeat recordings. The original large training corpus
is not assumed accessible through the supplied mount. No external corpus,
alternative compute route or files outside the authorized worktree/data route
were sought.

## Validation and execution blocker

Local lightweight validation passed:

- Python compilation of the trainer, utility and both test modules.
- Node TypeScript syntax checks for both training modules and the SDK test.
- Two standard-library-only unit tests: source-label/whole-speaker provenance
  checks and clean-transcript held-phone normalization.
- Git diff whitespace checks and inspection of all changed paths.

Remaining tests are implemented but have not executed. They cover exhaustive CTC
path sums, repeated articulations, duplicate negatives, numerical extremes,
sequence-order-sensitive gradients with identical phone classes, independent
reproduction of positive-only updates, frontend parity against actual listener
inputs, lawful-pronunciation exclusion, positive coverage, float/int8 capture,
cache preservation and complete auxiliary-bundle export.

The supervisor reports
`{"error": "NotFoundError: No such file or directory", "usd": 0.0, ...}`
before execution. There is no collected command log or training artifact.
The exact missing supervisor resource is unknown.

| Request | Work requested | Result |
| --- | --- | --- |
| 1–3, previous attempt | Inventory, minimal pwd, six-shard tests | Launch failure |
| 4 | Minimal runtime/module inventory | Same launch failure |
| 5 | Six-shard core/SDK build and tests | Same launch failure |
| 6 | Inventory through an ignored Python script | Same launch failure |
| 7 | New core/SDK build and Python loss/export tests | Same launch failure |
| 8–9 | Latest build/tests followed by training-only mining | Same launch failure |

Failures of both a minimal command and script-based inventory distinguish the
launch problem from shell quoting or training-code errors. Requests 8 and 9 run
Python tests and mining only on shard zero after the six-shard build/test harness.
Request 9 includes the final source changes. All nine requests have returned the
same launch error with zero compute spend; none executed a build, test or mining
command. The two small local tests and syntax checks above are the only executed
validation. This is an execution blocker, not a measured experiment kill.

Eligible negatives, positive coverage, actual gradients, selected heads,
float/int8 behavior on the supplied model and complete target replay are unknown.
No practice score was requested; all three permitted calls remain. No trained
bundle exists, so `.autolab/model.json` is absent. Scoring the main listener would
not evaluate this experiment. No killed outcome is written because neither the
premise nor the target stopping rule has been measured.

Positive labels establish training coverage, not an independent audio judgment.
Exact unlikelihood may produce negligible gradients when a complete rival has tiny
CTC probability, and positive/pause/preservation anchors cannot establish safety
on unseen substitutions. These remain empirical questions for the blocked runs.

## Resume

Restore the authorized supervisor and rerun request 9's build/tests/mining
command. Inspect training-only negatives before fitting. If eligible fit negatives
exist, run these commands through the supervisor:

```bash
python3 modal/student_sequence_unlikelihood.py prepare \
  --model /lab/data/models/spliced-clean-int8 \
  --mining /scratch/listener-false-sequence-unlikelihood/mining-v1/mining.json \
  --out /scratch/listener-false-sequence-unlikelihood/refit-v1
python3 modal/student_sequence_unlikelihood.py fit \
  --out /scratch/listener-false-sequence-unlikelihood/refit-v1
python3 modal/student_sequence_unlikelihood.py export \
  --model /lab/data/models/spliced-clean-int8 \
  --out /scratch/listener-false-sequence-unlikelihood/refit-v1
```

After training selection and complete bundle verification, compare both heads on
complete target sessions. Apply the stopping rule, declare the verified candidate
only if it survives, and obtain an official practice score if three or more
targets are won. The fold gate makes the final decision.

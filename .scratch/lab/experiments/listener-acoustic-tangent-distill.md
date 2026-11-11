# Acoustic tangent distillation

Status: training implementation and small unit validation complete; corpus audit,
matched pilot, export and acoustic evaluation blocked by the supplied execution
service. This is an incomplete, unmeasured experiment. The card is unchanged.

## Prediction

The card predicts three recovered added words and one collateral wrong call:
listener −2 and overall −2. Its premise is that the frozen teacher retains weak
consonant cues whose acoustic sensitivity static student distillation fails to
preserve. The six registered incidents remain evaluation targets only. No target
audio, labels, evidence or scores entered training or parameter selection.
No target-win count has been measured.

## Implemented mechanism

`modal/student_tangent.py` creates a small signed, tapered time-frequency patch
in the original recording's STFT. Inverse STFT synthesizes its PCM direction
`d`. Both models consume the exact same `x`, `x + epsilon*d` and
`x - epsilon*d`, through their respective original front ends. The direction
changes existing spectral energy rather than synthesizing phonemes. It preserves
sample count and timestamps, and zero-energy audio has a zero direction.
Its patch geometry and epsilon must come from an audited training-only recipe;
the tiny geometries in unit tests are not training choices.

For each model, let `o(x)` be every nonblank phone's log probability minus the
blank log probability on the original 40 ms output grid. The tangent is

```
[(o(x + epsilon*d) - o(x)) - (o(x - epsilon*d) - o(x))] / (2*epsilon)
```

The response loss is mean squared student/teacher tangent difference. Static
phone offsets and even response curvature cancel. Teacher responses are detached;
gradients train the student through its outputs, without second-order input
autograd. Response passes use fp32 and disable stochastic layers while restoring
the student's prior training flags and chunk/context configuration afterward.
The models use their shared valid output prefix; differing front-end edge rows
are counted and excluded, never repeated or interpolated. Whether their physical
frame origins match accurately enough is still an acoustic integration question.

The matched `ordinary` arm instead applies full-distribution teacher cross-entropy
to the same plus and minus PCM inputs. Both arms retain transcript CTC, original
cached ordinary KD, the original augmentation chain, and KL preservation against
a frozen copy of the initial ordinary student. Preservation includes blank rows
and real recording pauses. Augmentation is deterministic by seed, epoch and item
for these opt-in arms; perturbation draws use a separate key/step RNG. The two
arms must use identical data, anchor, geometry, loss weights and training schedule,
changing only response mode and run name. No matched arms have actually run.

`modal/student_train.py` exposes the mechanism programmatically through
`acoustic_response_mode`, `acoustic_response_recipe`, `acoustic_response_weight`,
`acoustic_preservation_weight`, `acoustic_anchor_run`, `acoustic_help_root` and
`acoustic_reference_index`, with `target="reference"` and positive CTC/static-KD
weights. The default response mode stays disabled. The image packages the helper,
canonical reference index, and existing pinned Muaalem/transformers versions.

The selected corpus combines audited `s1` clean-match training keys with
help-train none-labelled takes. Help selection requires the original `train`
split, the recipe's speaker, `kind="none"`, a scored/non-excluded row, and empty
events, repeats and unsupported-label lists. An empty event list alone is not
accepted. The complete mono 16 kHz recording is retained losslessly, including
pauses, and its canonical ayah phones supply transcript CTC. Frozen teacher
inference supplies its ordinary top-five targets. No help repeat or planted
mistake is used as a canonical clean transcript.

The recipe requires disjoint training/validation/excluded speaker lists,
speaker provenance for every selected key, both sources in each partition,
recorded training-speaker geometry and real-pause audits, and exact initial
student, teacher-state and combined corpus hashes. The combined hash includes
encoded PCM, transcripts and cached ordinary targets. Loader limits or exclusions
that change the frozen key set cause an error. These checks establish record
completeness and detect data drift; they do not prove that the producer correctly
identified every Quran-corpus reader or release-held-out alias. That actual audit
is still outstanding, and no recipe with invented provenance was created.

Checkpoint selection minimizes the frozen validation objective exclusively on
the recipe's training-speaker validation partition. `selected.pt` is separate
from the resumable `ckpt.pt`; `load_student()` requires and uses the selected
weights for these runs. Resume rejects a changed recipe/configuration, including
disabling a response continuation. Final metrics identify the training-speaker
validation set and the selected checkpoint's step. The existing streaming ONNX
exporter can consume that loader without an inventory or architecture change.
No actual exporter or SDK primary-listener bundle has been produced or verified.
Tracker, checker, SDK inference and insertion verification are unchanged.

## Validation and limits

Read AGENTS.md and `docs/agents/audio-review.md`. No acoustic incident triage,
independent listening review or teacher-retained target consonant audit could
run. The existing teacher diagnostics in the card motivate the hypothesis;
this attempt has not independently established it.

Twenty-five small CPU unit tests pass:

```
.scratch/tangent-test-env/bin/python -m unittest discover -s modal -p test_student_tangent.py -v
```

They cover analytic blank-relative derivatives, cancellation of static priors
and even curvature, nonlinear finite-difference convergence, student gradients
and teacher detachment, the identical-input ordinary control, padding exclusion,
perturbation locality/private RNG, sample preservation and silence. They also
exercise production loss dispatch with unchanged CTC/static KD, teacher identity,
selected-checkpoint loading, restoration after response/setup exceptions, and
shared-prefix handling without copying edge rows. Tiny help-ingestion fixtures
verify exact full-PCM retention, canonical transcripts/static targets, changed
data hashes, and refusal of repeats, unsupported labels, original eval speakers,
missing keys and paths outside the authorized corpus.

Tests run in an ignored environment inside this worktree with Python 3.13.3,
torch 2.8.0, NumPy and soundfile. Only tiny synthetic unit tensors and waveforms
ran locally. Compilation and `git diff --check` pass. No existing test, scorer,
lab tooling, supplied manifest, evidence or test data was changed.

Actual Muaalem/Zipformer integration, corpus schema compatibility, speaker
authorization, perturbation size, loss balance, streaming frame alignment,
memory/runtime cost, clean-pause preservation and quantized export remain
unverified. Unit tests cannot establish weak-consonant recovery or warning safety.

## Execution service record

Every request used the user-supplied supervisor. All four finished with
`NotFoundError: No such file or directory` before any command output:

| Run | Request | Result |
| --- | --- | --- |
| 1 | CPU, `pwd`, 3 minute timeout | Startup failure; no container output |
| 2 | Six CPU shards, `bash tools/autolab_tests.sh`, 5 minute timeout | Startup failure; no tests ran |
| 3 | L4/train, torch/checkpoint/s1/help-train inventory, 5 minute timeout | Startup failure; no inventory or inference ran |
| 4 | Six CPU shards, `bash tools/autolab_tests.sh`, 5 minute timeout | Startup failure; no tests ran |

Run 3 reports $0.2096 charged; the final budget record reports $0.21 spent and
$59.79 remaining from $60. The CPU requests report $0 GPU cost. There are no
pending runs. No alternate Modal access or outside-worktree data search was used.
Core and SDK regression suites were requested but cannot be reported as passing.

## Result and remaining work

The execution service cannot start even `pwd`, so the authorized audio and
initial checkpoints cannot be audited, either matched arm trained, or full
sessions decoded here. There is no checkpoint, export, declared model bundle,
`.autolab/model.json`, or practice score. All three scoring opportunities remain.
A score using the main model would not evaluate this implementation.

Infrastructure failure does not establish the acoustic premise or the kill
criterion. No killed outcome or fabricated zero-target measurement is written.
The implementation is retained for review with this explicit incomplete result.

After service restoration, audit corpus speakers and release-held-out aliases,
verify the true none-labelled help schema and real pauses, validate perturbation
geometry on training partitions, and freeze a recipe with measured hashes and
an identical configuration for both arms. Run the matched pilot, select within
training-speaker partitions, export and verify the complete primary model bundle
with the unchanged alphabet, clock and SDK streaming path, and evaluate the six
full-session target streams against the ordinary control. Under the user's latest
rule, fewer than three built-model target wins kills the experiment and requires
reverting engine changes and recording that measured outcome. Three or more
require a practice score before any narrowing; the supervisor fold gate decides
whether the gain and collateral wrong calls are acceptable.

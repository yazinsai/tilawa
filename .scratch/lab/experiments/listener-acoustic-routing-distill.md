# Acoustic attention routing distillation

Status: blocked and unmeasured. Training implementation and focused unit
validation are complete; corpus audit, matched training, export and acoustic
evaluation are outstanding. The supplied execution service failed requests
1, 2 and 4; GPU request 3 remained without a response after 30 minutes.
The card is unchanged.

## Prediction

The card predicts two recovered additions with no losses: listener −2 and
overall −2, with substantial zero-gain risk. The seven registered additions
are evaluation targets only. No practice audio, target transcript, incident
identity or practice score enters training or checkpoint selection.

The card asks whether the exported routing student restores missing ordered
consonants on at least two targets relative to ordinary distillation. The
user's final rule additionally requires at least three built-model target wins
before practice scoring. Neither count has been measured.

## Mechanism implemented

`modal/student_routing.py` observes the actual recurrent Zipformer attention
operator's direct `streaming_forward` calls. Ordinary module hooks cannot see
those calls. A scoped instance wrapper returns the original outputs unchanged,
captures the attention probabilities and input features, and restores the
method even after an exception. The teacher uses ordinary forward hooks on
selected inference-mode Wav2Vec2-BERT attention operators. Teacher targets and
features are detached, and only selected quadratic maps are retained.

Acoustic positions use the pinned frontends' physical sample grids. Kaldi
uses 400-sample windows with a 160-sample hop. The student's valid convolution
stack covers fbank rows `2j..2j+8`, centred at sample `840+320j`.
The teacher stacks rows `2j,2j+1`, centred at `280+320j`, before its CTC
adapter. A student downsample group represents adjacent observations. Teacher
query rows are averaged within its group; teacher source probability mass is
summed into the corresponding student source bins. Sequence lengths never
rescale time, and unpaired teacher edges are excluded rather than repeated.

Both distributions are restricted to each captured invocation's real cache
length and key-padding mask. Negative startup-cache positions and padded tail
positions are excluded. Each teacher head is conditioned on available keys
before correspondence, so different amounts of inaccessible teacher mass
cannot distort the mixture. Empty supported rows fail explicitly. This
restricts retrieved positions; it does not remove context already encoded in
the teacher's hidden states or establish causal equivalence of representations.

A learned row-softmax correspondence maps each student head to a convex
mixture of teacher heads despite their different widths. It is calibrated
using the frozen initial student on training speakers, then frozen before
acoustic refitting or checkpoint selection. The routing objective is mean KL
over the available positions. No value-vector, output-occupancy or emission-time
match substitutes for this objective.

`modal/student_routing_train.py` runs ordinary, feature-hint and routing arms
with the same initial student, records, augmentation draws, successful update
count, teacher calls and recurrent passes. All arms compute both auxiliary
objectives; zero coefficients prevent the irrelevant objective from training
the ordinary or feature control. The feature control learns a training-only
linear width projection and matches normalized hidden vectors.

All arms retain full-recording transcript CTC, fresh full-distribution ordinary
teacher supervision, and KL preservation against the frozen initial student's
recurrent output. Preservation covers every output row, including real pauses.
CTC keeps the existing augmentation and SpecAugment chain; teacher and auxiliary
recurrent passes consume the same channel-augmented PCM without SpecAugment.
An opt-in `routing_paired` dataset mode seeds augmentation by seed, epoch and
recording index. Existing dataset behavior remains the default.

The runner requires an audited, frozen training-only recipe: disjoint
training/validation speaker groups, release speaker exclusions, complete key
provenance, exact checkpoint/teacher/corpus hashes, audited real pauses,
operator pairs and fixed objective weights and schedule. Quran examples must
be original clean-match training rows outside the existing gold exclusions.
Help examples must explicitly have `kind="none"`, original `split="train"`,
the audited speaker, scored status, and empty events, repeats and unsupported
labels. Full mono 16 kHz PCM is retained, including pauses. An empty event
list alone is insufficient. No mistake or repeat becomes a canonical clean
transcript. These checks verify record consistency; the original reader and
alias authorization still needs an actual corpus audit.

Checkpoint choice minimizes the same transcript-CTC-plus-preservation criterion
for all arms on the speaker-separated training validation partition, in
deterministic mode. Nonfinite updates abort instead of silently altering
effective training data. Correspondence and feature projections are stored
separately from student checkpoints and never exported.

The export path creates a dynamic-int8 primary student with the original
inventory, 40 ms clock and streaming geometry. It compares fp32 ONNX with
the same PyTorch recurrent calls, checks int8 inventory/frame count and finite
output, and records quantization agreement on training validation recordings.
It resolves shipped insertion assets from the actual SDK registry, checks
their identities and copies the complete bundle. A manifest is written only
after these checks. No new auxiliary model is deployed. Tracker, checker,
SDK decoding and insertion verification code are unchanged.

## Validation and limits

Read `AGENTS.md` and `docs/agents/audio-review.md`. Reviewed the pinned icefall
and transformers attention/frontend source. Sixteen small helper tests passed
initially. They cover physical grids, probability-mass projection, independent
head conditioning, support exclusion, masked NaNs and gradients, learned
unequal-head correspondence, teacher detachment, capture neutrality, cleanup
after exceptions, and restoration of execution mode and geometry.

All 27 focused unit/integration tests pass. The extended suite exercises
production objective dispatch,
speaker partitions, whole-PCM none-take ingestion, data/state fingerprints,
and production paired augmentation:

```
.scratch/routing-test-env/bin/python -m unittest discover -s modal -p 'test_student_routing*.py' -v
```

Compilation and `git diff --check` pass. The small test environment uses
Python 3.13.7, torch 2.8.0, NumPy, pyarrow and soundfile. No corpus training
or acoustic-model inference ran locally. The final artifact-path validation
and nested-capture fixes also passed their five recipe and four capture tests.
Nested captures observe the same forward result and restore both wrapper
scopes correctly.

Tests use small synthetic tensors and temporary recordings within this
worktree, not practice or release data. No scorer, lab tooling, supplied
manifest, evidence, test data or experiment-card edits were made.

Real Muaalem/Zipformer integration, actual help schema compatibility, reader
authorization, loss balance, attention pairing quality, GPU memory/runtime,
ordinary-pause preservation, quantization drift and SDK loading have not
been verified. No training recipe with invented speaker provenance was made.
Unit agreement cannot establish restored consonants or safe warnings.

## Execution and result

All execution requests use the supplied supervisor. Runs 1 and 2 failed with
`NotFoundError: No such file or directory` before container output. Run 1
requested training data/checkpoint inventory; run 2 requested the supplied
six-shard core/SDK build/test harness. No tests or corpus operations ran there.
The CPU requests report zero GPU spending.

Run 4, the final six-shard build/test request with the training image, also
failed with the same `NotFoundError` before container output. Core and SDK
regression suites therefore cannot be reported as passing.

At 2026-11-12 16:20 UTC, run 3 (L4/train, unit tests and inventory, requested
15:50 UTC with a ten-minute runtime) still had no result file or container
output. The supervisor wrapper permits extra startup waiting, so this is an
unresponsive request, not a measured acoustic failure. No alternative Modal
access was attempted. The request remains supervisor-owned; its lifecycle is
handled by the supervisor when the executor ends.

There is no trained checkpoint, exported bundle, `.autolab/model.json`, target
win count or practice score. All three practice scoring opportunities remain.
Scoring the default listener would not evaluate this training implementation.

Execution-service failure and an unresponsive request neither falsify the
acoustic premise nor fire its
kill criterion. The implementation is retained as incomplete and unmeasured;
no killed outcome or fabricated zero-target measurement is written.

After execution is restored, audit authorized speakers and corpus schemas,
freeze the operator correspondence recipe and matched schedule within training
partitions, run all three arms, and verify each complete exported bundle through
the SDK. Compare the registered targets using full-session recurrent streams
against ordinary distillation. Fewer than three built-model target wins requires
rollback and a measured killed report under the user's final rule. Three or
more requires practice scoring before narrowing; the supervisor fold gate
decides whether the aggregate gain and losses permit folding.

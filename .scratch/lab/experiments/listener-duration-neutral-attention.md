# Duration-neutral acoustic memory

Status: implementation and small unit validation complete; matched training,
whole-model export, target-stream validation and core/SDK regressions blocked
by the supplied execution service. This is an incomplete, unmeasured experiment.
There is no scoreable new listener bundle.

## Prediction

The card predicts two recovered added-word misses and one collateral wrong
call: listener −1, overall −1 on lab-v1.5u. Its proposed cause is redundant
sustained-sound keys buying repeated attention mass at the expense of brief
consonants whose input evidence survived. That acoustic premise has not been
established here. No target recovery count or practice change was measured.
The experiment card is unchanged.

## Implemented mechanism

`modal/student_memory.py` wraps the attention-weight modules in the last two
Zipformer encoder stages. Each head learns a diagonal metric on adjacent
projected-key differences and a join bias. Hard adjacent joins define contiguous
groups in training and inference. A straight-through gradient uses products of
soft adjacent-join probabilities, so grouping learns jointly with attention.
Similarity cannot join separated occurrences across an intervening group.

For a query with original content-plus-position score s_i and visible group
G, its group score is `logsumexp(s_i for i in G) - log(|G|)`. Attention is
normalized over groups, then distributed within each group by the original
frame softmax. The equivalent expanded frame weight is proportional to
`exp(s_i) / |G|`. Repeating an identical key does not multiply the group's mass.
Original frame values and relative positions remain available; values are not
averaged before the query retrieves them.

The group score also receives a learned query-dependent coefficient multiplying
`log1p(visible group duration in seconds)`. Each stage's frame duration derives
from the existing 20 ms subsampling clock and stage downsampling factor.
The annotation starts at zero and lets attention learn legitimate duration
distinctions independently of repeated-key count. Duration describes retained
visible frames, not elapsed audio already evicted from the normal cache.

Counts exclude padding, future chunks and unavailable history. The adapter
requires contiguous visible intervals and rejects hidden holes in eager mode.
Prefix sums compute interval counts in O(K² + QK), avoiding the cubic dense
membership multiplication. All-hidden queries return zero. Group identities
and counts stay float32 under bfloat16 acoustic autocast.

This is an expanded hierarchical representation, not a reduction in the number
of cached frames. The existing cache layout, frame values, positional offsets,
phone inventory and output clock are retained. Its compute and memory overhead
on the deployed listener remain unmeasured. The module is compatible with the
inspected pinned icefall projection and zero attention-dropout interface, but
the complete pinned Zipformer has not executed with it here.

## Matched continuation support

`modal/student_train.py` adds programmatic `duration_memory`,
`preservation_weight`, `preservation_run`, fixed streaming geometry and seeded
augmentation options. Both ordinary and memory arms can start from the same
frozen ordinary checkpoint. The memory arm wraps its copied upper attention;
the control retains ordinary attention. A masked KL term preserves the frozen
listener's ordinary-reading posteriors on the same augmented inputs. Reference
CTC uses authorized native clean-match training clips and admitted help takes.

`modal/student_none.py` admits only scored, nonexcluded, training-split,
none-labelled help takes with no mistake events, repeats or unsupported labels.
A speaker appearing in any nontraining split is excluded altogether. It uses
canonical reference phones, refuses paths outside the supplied help directory,
and requires mono 16 kHz audio. Scripted mistakes are never converted into
transcripts. Missing teacher dumps have zero teacher-frame length and padded
storage rows, so teacher KD ignores those takes while CTC and preservation can
train on them.

The continuation requires a positive preservation loss, transcript supervision,
explicit help paths, fixed streaming geometry and an augmentation seed. It
refuses a second initialization source, missing augmentation banks, or
augmentation settings differing from the preservation checkpoint. Random
augmentation is indexed by seed, epoch and item; batch ordering already uses
the training seed. The two arms must still be launched with identical data and
schedule configurations. No actual matched configuration or checkpoint was
chosen without inspecting the supplied training assets.

Continuation diagnostics use native training-split recordings rather than
release/test speakers. They are training diagnostics, not independent
generalization evidence or a demonstrated checkpoint-selection procedure.
Existing ordinary-training defaults remain available. Loading a memory
checkpoint rebuilds its wrapped architecture before loading weights. The
existing exporter calls this loader, allowing the operation to be traced into
the primary ONNX graph consumed by the SDK. This path has not yet been verified
with a complete trained model; no SDK bundle or supplied manifest was changed.

## Validation

Read AGENTS.md and `docs/agents/audio-review.md`. Inspected the public pinned
icefall attention source at commit
`3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`. Key balancing and whitening preserve
forward values, including the keys stored by streaming attention. Source
inspection is not an independent listening judgment or target premise audit.

Eighteen unit tests passed using tiny CPU tensors and synthetic temporary audio
in an isolated worktree environment with torch 2.8.0. Reproduction after moving
that environment to an already-ignored development directory:

```
PYTHONPATH=modal:tools .scratch/student/duration-attention-dev/venv/bin/python -m unittest discover -s modal -p test_student_memory.py -v
```

Tests cover repetition invariance; explicit hierarchical weights and retained
value detail; singleton parity; contiguous groups; visibility and truncation;
duration retrieval; finite joint gradients; float32 group identities under
bfloat16; unchanged streaming caches; upper-stage installation and checkpoint
roundtrip; and standalone ONNX Runtime parity for normal, truncated and hidden
memory. Prefix counts and parameter gradients match the dense membership
oracle. Training tests cover preservation and shipped-augmentation requirements,
speaker/script exclusion, appended audio and buffer offsets, actual production
collation, and actual production CTC/KD/preservation loss with missing teacher
supervision and padded frames. The production definitions are loaded without
executing Modal image declarations; CUDA autocast alone is disabled for the
tiny CPU loss test.

Python compilation and `git diff --check` passed. No scorer, lab tooling,
supplied test data, manifest or evidence was edited. No acoustic inference,
corpus analysis or training ran on the Mac.

## Execution service and limits

All requests used the user-supplied supervisor. Four requests failed before
container startup with `NotFoundError: No such file or directory`, without
container logs:

| Run | Request | Result |
| --- | --- | --- |
| 1 | CPU inventory, 5 minute timeout | Startup failure |
| 2 | Training-image checkpoint, icefall and help inventory, 10 minute timeout | Startup failure |
| 3 | Six-shard core/SDK harness, 5 minute timeout | Startup failure |
| 4 | Retry of six-shard core/SDK harness, 5 minute timeout | Startup failure |

All four returned $0 charged; the last result reports $90 GPU budget remaining.
Core and SDK regression tests were requested twice and cannot be reported as
passing. The authorized audio and checkpoints are available only inside these
containers. No alternate Modal access or outside-worktree data search was used.

There is no trained checkpoint, whole-model or int8 export, target-stream
comparison, `.autolab/model.json`, or incomplete declared asset bundle. No
practice score was requested; all three scoring opportunities remain. A score
with the default model would not evaluate this attention change. Infrastructure
failure does not establish the acoustic kill criterion, so there is no killed
outcome and no fabricated zero-win result.

After service restoration, audit the supplied initial listener recipe,
transcripts, help schema and release-held-out exclusions; verify the pinned
whole-model integration; run identical-data ordinary and memory continuations;
export and verify both complete streaming bundles; and compare the six
registered full-session streams. The premise check must identify previously
missing planted consonants recovered relative to the ordinary continuation,
not merely improved crops or teacher agreement. The card's acoustic threshold
is fewer than two such target streams; the user's final instructions also
require killing a built change with fewer than three target wins. A premise
failure or insufficient built-model wins requires reverting the engine changes
and recording the measured killed outcome. Three or more wins require a
practice score before any narrowing, with the supervisor fold gate deciding
the gain/loss tradeoff.

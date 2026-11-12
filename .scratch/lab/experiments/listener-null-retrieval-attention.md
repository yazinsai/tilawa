# Query-conditioned null acoustic retrieval

Status: implementation and small-model validation complete; corpus fitting,
registered-target comparison, full trained model export and core/SDK regression
checks are blocked by the supplied supervisor. This experiment is incomplete
and unmeasured. There is no scoreable replacement listener bundle.

## Prediction

The card predicts two recovered added-word misses and one collateral wrong call:
listener −1, overall −1 on lab-v1.5u. The speculative cause is compulsory
contextual retrieval overwriting weak local consonants. No target audio or
posteriors were available here to establish that cause. The experiment card is
unchanged.

## Mechanism

`modal/student_null_attention.py` replaces only `self_attn_weights` in the last
layer of the final Zipformer stage. All pre-existing parameters are frozen,
including this layer's content and positional projections, values, feedforward
paths, convolutions, normalization and output head. The rest of the encoder is
unchanged.

For each head and query, original real-key logits remain the content dot product
plus the exact relative-position score. The adapter appends one score
`w_h · q_h + b_h`, where q is that head's existing projected acoustic query.
It normalizes real and null logits together. The null value is exactly zero:
returning only the real-key weights to the existing value consumers is
algebraically identical to appending a zero value and multiplying by the full
extended distribution. The same zero contribution applies to both ordinary
attention value streams and the block's shared nonlinear attention stream.
Existing residual and projection-bias paths are retained.

The null entry occupies no cache slot, position or acoustic frame. Real key
caches, context availability and positional offsets remain unchanged. Padding
and attention masks exclude real keys before normalization; the null stays
available. No pause labels, expected verse or candidate-word information enter
the scorer. Only the per-head query weights and biases are trainable: 132
parameters for the shipped four-head, 32-dimensional final attention.

The temperature arm has the same number and geometry of query parameters.
Its query-conditioned log temperature scales original logits before masking
and softmax; it has no null entry and retains unit real-key mass wherever a real
key is available. Temperature one exactly reproduces the ordinary attention on
valid queries. Both arms fit with the existing acoustic modules in evaluation
mode, so frozen dropout and other stochastic training behavior cannot differ
between them.

## Training and export path

`modal/student_null_fit.py` provides resumable `cache`, `fit` and `export`
commands for the supplied supervisor's training image. It does not launch Modal
jobs or access additional data sources.

Native supervision admits only train-split clean-match transcripts with speaker
clusters, at most 30 seconds per clip, within a fixed seeded two-hour subset.
It excludes entire clusters appearing outside train or containing a gold-excluded
clip. Help supervision admits only scored, nonexcluded, none-labelled train
recordings without mistake events, repeats or unsupported labels. A help
speaker present in any nontraining split is excluded altogether. Audio paths
must remain inside the supplied help directory and decoding requires mono
16 kHz audio. Scripted-mistake recordings are never turned into transcripts.

One seeded speaker-grouped 80/20 partition is fixed before fitting. Native
clusters are the existing pipeline's approximate speaker groups, not independently
verified people; their identity and release-exclusion coverage still require
an asset/provenance audit before real fitting. Help release speakers are absent
from the supplied training copy, with additional whole-speaker split exclusion
in the adapter. No actual admitted speaker counts have been measured.

Each admitted take has an ordinary view and one deterministic draw from the
initial checkpoint's shipped noise, RIR, gain and SpecAugment recipe. Both arms
reuse identical cached lower-block features for both views. The cache retains
input to the final layer, its real positional encoding and mask, final-stage
bypass input, and the lower-stage output channels needed by the existing head.
Fitting replays the final layer in recording order and carries its newly computed
attention-value and convolution states. It does not replace changed upper state
with baseline upper caches.

The loss is transcript CTC plus KL preservation of the unchanged listener's
posteriors on the same input, and a small pull of the new projection toward its
initialization. CTC and KL are computed on the original recording's output
clock, excluding padded output frames. Both arms use the same fixed optimizer,
data order and four-epoch schedule. Preservation weights {1, 10, 100} and epoch
checkpoints are selected exclusively by training-partition validation: reference
PER, followed by CTC plus preservation, with deterministic ties. Each arm is
selected independently from the matched candidate schedule; update counts are
recorded. The shared design verifies paired recordings, speakers, partitions,
geometry and schedule on resume. Pre-existing parameter fingerprints must
remain unchanged.

A persisted process-CPU budget caps cache construction, fitting and export at
one CPU day across requests. Checks occur between complete-recording operations
and export stages, so a single operation can finish across the boundary. No
actual corpus CPU time or fitted checkpoint exists here.

The existing `student_train.load_student` recognizes the new checkpoint flag
and restores the adapter before strict weight loading. The existing streaming
exporter traces the operation into the primary ONNX model. The generic SDK
loader therefore consumes it through its existing posterior/state interface;
no tracker, checker, SDK decoder, insertion verifier or alphabet change is needed.

Export prepares unchanged, null and temperature bundles with the shipped
MatMul-only dynamic int8 recipe. It checks training-only cached-suffix/FP32
streaming parity, finite int8 outputs, manifest inventory/clock/state contracts,
and reproduction of the supplied int8 listener by the unchanged export. It
requires, copies and hashes the unchanged insertion teacher and its external
data against the SDK registry. A verified-bundles marker is written only after
all three bundles complete. This full real-model path has not executed here.
No supplied manifest or evidence was edited.

## Validation

Read AGENTS.md first. Inspected the attention, residual, cache, export and SDK
loading paths, the card, and the prior duration-neutral and norm-neutral branch
write-ups. Checked public icefall source at pinned commit
`3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`.

Sixteen small CPU tests pass in the worktree-local development environment with
torch 2.8.0, ONNX 1.19.0 and ONNX Runtime 1.22.1. They cover:

- Explicit extended-softmax/zero-value equivalence, query conditioning, local
  residual retention and the temperature arm's inability to abstain.
- Original relative-position scores, masking, masked-history independence,
  unchanged real-key caches and zero-length history.
- Single-block installation, finite projection gradients, unchanged existing
  parameters, checkpoint roundtrip and the production checkpoint loader.
- Help admission, deterministic speaker-grouped partitions, complete-recording
  CTC/preservation and a persisted CPU-budget limit.
- Cached-prefix versus full streaming parity with changed carried upper state,
  matched fitting and validation-only selection, unchanged selection/weights
  after resume, and rejection of mismatched paired data.
- Standalone attention ONNX parity and complete tiny Zipformer primary ONNX
  streaming parity with unchanged output and state shapes.

The local tiny integration uses the pinned Zipformer, subsampling and scaling
Python source with analytic torch substitutes for unavailable k2 Swoosh kernels
and a disabled CPU autocast context. This validates replay and graph structure,
not native-kernel, real-checkpoint or real-audio behavior. The committed
integration tests use the actual dependencies in the supervisor's train image;
they skip when pinned Zipformer is unavailable. The ordinary local unit command
can be reproduced with:

```
PYTHONPATH=modal:tools .scratch/student/null-retrieval/venv/bin/python -m unittest discover -s modal -p 'test_student_null*.py' -v
```

The supplementary pinned-source local run is recorded in
`.scratch/student/null-retrieval/unit-validation.log` (ignored development
material). No acoustic training, corpus analysis or real-audio inference ran on
the Mac. Python compilation and `git diff --check` passed.

## Supervisor failure and remaining work

All execution requests used the supplied supervisor:

| Run | Request | Result |
| --- | --- | --- |
| 1 | Default-image model/data/dependency inventory | `NotFoundError`, before container startup |
| 2 | Training-image checkpoint/data inventory | `NotFoundError`, before container startup |
| 3 | Required six-shard core/SDK build and tests | `NotFoundError`, before container startup |
| 4 | Native training-image null-attention tests | `NotFoundError`, before container startup |

All four requests returned no command logs or container outputs and
charged $0. Service restoration was requested. These failures do not establish
an acoustic kill. Core and SDK tests cannot be reported as passing.

After restoration, audit initial checkpoint provenance, native speaker groups
and help metadata; run the native-kernel tests; build the shared cache; fit and
freeze both training-only selections; then export and verify all three complete
bundles. Compare the seven registered targets using full primary streaming
decode against both unchanged attention and the selected temperature control.
Previously missing planted consonants must actually reappear in order; a crop
or posterior-mass change alone is not a target win.

The card's acoustic threshold is two recovered consonant targets, and the user's
final instructions permit killing a built change below three target wins. Three
or more wins require practice scoring before a gate decision. Losses elsewhere
alone do not justify stopping; the supervisor's fold gate trades them against
wins, with fully clean-session false warnings a hard stop. If the acoustic kill
is established, revert the engine changes and record a measured killed outcome.

No target-recovery count, wrong-call change, clean-session precision or studio
result has been measured. No practice score has been requested; all three
opportunities remain. There is no trained model, `.autolab/model.json`, incomplete
declared bundle, or fabricated killed outcome. Scoring the default model would
not evaluate this attention change.

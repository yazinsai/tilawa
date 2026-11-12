# Common-descent frozen phone-head refit

Status: implemented; training and acoustic validation blocked by the supplied
supervisor. No replacement model has been declared. The experiment card is
unchanged, and no acoustic kill criterion has been evaluated.

## Prediction

Listener −1 wrong call, overall −1, with one recovered addition and no predicted
losses. This is inside the stated noise band and below the fold floor. The
speculative premise is cancellation between consonant, vowel and blank learning
in the existing CTC head, rather than missing encoder information.

## Mechanism

`modal/student_pareto.py` computes exact transcript-conditioned CTC
forward/backward occupancies, including the blank states, repeated labels and
their required intervening blank. For group k, let gamma_k be its frame/class
occupancies and m_k their sum over classes. Its logit-gradient contribution is

```
g_k[t, :] = m_k[t] * p[t, :] - gamma_k[t, :]
```

The three groups partition the ordered inventory into consonant identities,
vowels/madd/marks, and blank. The vowel/mark inventory follows the existing
core's phoneme grouping; nasal consonants and their tajweed variants remain
consonant identities. Each contribution includes competition with every output
class. It is not a mask over output gradient columns. Because the occupancies
sum to one per frame, the contributions sum to `p - gamma`, exactly the ordinary
CTC gradient. Multiplication by the frozen features augmented with a bias column
gives the complete weight-and-bias contributions. The code checks their
reconstruction of the ordinary gradient at every update.

The experiment projects that summed gradient onto the intersection of the three
half-spaces `dot(d, g_k) >= 0`. With only three constraints, enumeration of all
active sets and tiny Gram-system KKT solves gives the nearest feasible direction
in Euclidean head-parameter space. Dependent, opposing and zero contributions
are handled. Constraint normalization improves the solve's conditioning without
changing a half-space. There is no example reweighting, momentum, clipping or
post-projection rescaling. The occupancies are recomputed at each current head;
the decomposition describes current gradient responsibilities, not three
independent global objectives with fixed targets.

## Matched data, selection and export

`modal/student_pareto_train.py` prepares one shared frozen-feature cache, then
refits the existing affine projection in ordinary and common-descent arms.
Both start from the same shipped projection. Dynamic projection weights are
dequantized and activation rounding is removed identically in both arms; the
encoder remains frozen and quantized. Any improvement common to both refits
would not establish the common-descent premise.

Help-train admission requires training-speaker `none` takes, scored labels and
empty events, non-errors and unsupported lists. Their authorized full-ayah
transcripts come from the existing reference inventory. Optional additional
performed-transcript JSONL rows must explicitly provide `speaker`,
`split=train`, `authorized=true`, `verified=true`, `phones`, `wav` and a matching
audio `sha256`. Mistake scripts and repeat-labelled takes are not used as
performed transcripts. All speakers in non-training splits across either
corpus, plus any explicitly enrolled exclusions, are removed before enrollment.
The supplied service also guarantees release-held-out audio is absent. Audio
hashes reject duplicates, paths must stay inside their enrolled corpus, and
cache hashes bind both arms to the same evidence.

This bounded refit uses unmodified full recordings, retaining their ordinary
speech and actual pauses under transcript CTC. It introduces no augmentation,
teacher relabelling, pause positives, auxiliary loss or preservation-gradient
term that could invalidate the stated common-descent constraints. CTC uses the
existing sum divided by valid output-frame count. Each recording receives one
update per epoch in both arms. Training runs twelve epochs at learning rate
0.02, with deterministic identical recording order and no arm-specific weights.
These choices were fixed without practice access.

Whole training speakers are deterministically partitioned into fit and
selection groups. One shared epoch, including epoch zero, minimizes the mean of
the two arms' selection CTC objectives, with equal averaging across selection
speakers. Both exported checkpoints therefore have identical applied update
counts. No practice labels, target spans or practice scores enter preparation,
training or checkpoint selection. Training reports include per-group gradient
conflicts, projection displacement and numerical constraint violations.

`modal/student_pareto_export.py` exposes the native pre-projection features and
requires original probabilities and state outputs to remain byte-identical
before caching. Export replaces the projection and prunes unused old head
nodes. It inherits the original ordered vocabulary, frontend, window/shift,
state interface and 40 ms posterior clock. The existing SDK can load the
resulting primary model without a product-code change; decoding, tracker,
checker and insertion verification retain their existing rules.

The complete source insertion teacher and its ONNX external weights are
required, copied and checksum-verified in both bundles. Missing or truncated
members fail export. Both exported heads must reproduce their cached selection
predictions through actual streaming ONNX inference within 2e-4 probability
tolerance. The final training report is written only after both complete
bundles pass. Generated manifests describe the new graphs; supplied manifests
and evidence are never rewritten.

## Validation and observed result

The local synthetic suite passed 16 tests, with one explicit skip for the
PyTorch original-CTC oracle:

```
arch -x86_64 .autolab/test-venv/bin/python -m unittest discover -s modal -p test_student_pareto.py -v
```

It checks forward sums and occupancies against independent path enumeration,
complete CTC weight/bias gradients against finite differences, cross-class
softmax contributions, repeated and empty transcripts, impossible alignments,
disjoint phone groups, nearest projection against an independent SciPy QP
solver, dependent/opposing constraints, and unchanged feasible gradients.
Admission tests cover global cross-corpus speaker exclusions, repeat rejection,
unverified transcripts and path escapes. The matched-control test checks the
same ordered examples, applied update counts and shared selection epoch.

Synthetic ONNX tests use ONNX's reference evaluator to check float and dynamic
int8 head extraction/replacement, unchanged encoder-state outputs, complete
teacher/external-data copying, rejection of truncated weights, manifest hashes
and inherited contracts. A streaming-cache test checks carried state, padding
and valid-frame drain boundaries. Only tiny generated inputs were used locally,
with NumPy 2.4.4, SciPy and ONNX 1.23.2. Python compilation passes. These checks
do not verify the supplied real graph or its compatibility with the SDK.

All three supplied supervisor requests returned
`NotFoundError: No such file or directory`, USD 0.00, with no command logs:

1. Run 1: runtime, supplied data and source-bundle inspection.
2. Run 2: a simpler runtime/data probe.
3. Run 3: synthetic unit tests followed by the required six-shard core/SDK
   build-and-test harness (`bash tools/autolab_tests.sh`).

Consequently no training audio or source model was inspected remotely, no cache
was prepared, no model was fitted or exported, and no full-session target decode
was compared with the matched ordinary control. Core/SDK tests and the PyTorch
oracle remain unrun. There were zero practice-score invocations. Target wins,
losses and component/overall changes are unknown, not measured as zero.

## Limits and resumption

This is an infrastructure block, not evidence that the acoustic premise is
wrong. No `.autolab/model.json` or acoustic-kill outcome is present. Scoring the
shipped model would not evaluate this implementation. The implementation is
retained for resumption; a complete real replacement bundle is still required
before acoustic handoff.

Once the supplied service is restored, run the unit and core/SDK suites, inspect
the authorized corpus and source bundle, then prepare and fit entirely under
the experiment's persistent scratch directory. For example, through the
provided `modal-run` command, with BLAS threads limited to one:

```
python3 modal/student_pareto_train.py prepare \
  --model "$LAB_DATA/models/spliced-clean-int8" \
  --help "$LAB_DATA/help-train/labels.jsonl" \
  --out /scratch/listener-pareto-phone-head/prepared
python3 modal/student_pareto_train.py fit \
  --prepared /scratch/listener-pareto-phone-head/prepared \
  --out /scratch/listener-pareto-phone-head/refit
```

Enroll additional verified transcripts and exclusions with `--transcribed` and
`--exclusions` when supplied. Freeze the selected checkpoints before any target
analysis. Compare complete-session SDK decodes on the six card targets with the
ordinary-control bundle, distinguishing newly recovered planted consonants
from actual confirmed warning wins. The card requires two consonant-recovery
spans; the final user rules also require three target wins before practice
scoring. Apply the stated acoustic kill procedure if the premise or target
floor fails. If at least three targets are won, run the supplied practice score
before deciding or narrowing, and let the fold gate account for collateral
losses. First-order gradient feasibility and retained ordinary recordings do
not guarantee loss-free recognition or absence of new clean-session warnings.

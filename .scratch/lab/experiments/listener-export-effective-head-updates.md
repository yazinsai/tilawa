# Export-effective phone-head updates

Status: implementation and synthetic behavior tests complete; acoustic premise,
real training and deployment blocked by supervisor execution. Neither a measured
nor a killed experiment. The experiment card is unchanged.

## Prediction and mechanism

The card predicts listener −1 and overall −1, one recovered added-word miss and
no losses. The proposed failure is that a useful floating-point phone-head
correction disappears when the selected head is quantized for deployment.
Direct optimization of bounded int8 weight-code changes should make a correction
survive that boundary. This forecast is below the fold threshold.

The preceding float32-writeback audit rules out reduced-precision parameter
storage in the current trainer, but explicitly leaves export quantization open.
`tools/quantize_onnx.py` uses ONNX Runtime's dynamic MatMul-only signed-int8
recipe. `modal/student_export.py` exports the streaming head as an affine
projection followed by LogSoftmax. The source therefore permits this failure;
it does not establish that any registered target actually loses a correction.

The decisive premise audit is still required: select an ordinary floating-point
head refit using training speakers only, export that same selected head through
the existing recipe, and compare the two artifacts on all five targets. At least
one newly recovered consonant or eligible whole-word candidate must disappear
through export. If none does, stop and revert the engine additions. A small
numeric change or a different token without useful target evidence is insufficient.

## Implementation

`modal/student_quantized_head.py` extracts the existing dynamic-int8 phone-head
subgraph, including activation quantization, integer accumulation, dequantization,
bias and LogSoftmax. It accepts the current per-tensor signed-int8 recipe and
rejects unsupported or ambiguous layouts. Grafting removes only the old head
and checks that it shares no removed parameters or intermediate tensors with
the encoder. The original encoder and every streaming state output remain.

Cached features retain full SDK-sized chunks and the number of valid rows in
each. The last chunk includes features computed with final frontend padding;
invalid output rows are discarded only after head inference. This matters:
DynamicQuantizeLinear chooses one activation range over its input tensor, so
concatenating a recording or truncating its final chunk changes the computation.
Preparation compares exposed features against the unchanged primary graph,
including state outputs, then checks isolated-head posterior and argmax parity.

`modal/student_export_effective.py` provides separate `prepare`, `control` and
`discrete` commands. The ordinary control starts from the deployed head's
dequantized weights, refits them with Adam, and exports the selected snapshot
using the existing `tools/quantize_onnx.py` recipe. It writes both the selected
floating-head artifact and its int8 export for the premise audit. This avoids
requiring an unavailable historical source checkpoint; the encoder remains
the actual shipped quantized encoder in both artifacts.

The discrete arm starts from the same deployed weight codes. Transcript-loss
gradients evaluated at actual exported outputs rank one-code coordinate moves.
A floating affine tangent suggests directions; it cannot accept an update.
Each candidate runs through the actual ONNX head on every fit recording, and
is retained only if the aggregate transcript-plus-preservation objective
strictly improves. Codes stay within two integer steps of their starting values
and within the signed-int8 range. Weight scales and zero points stay fixed.
Both arms freeze the float32 bias as well as the encoder, isolating weight-code
updates without introducing bias corrections or extra model capacity.

Both arms use the same complete recordings, initial outputs, equal recording
weights, transcript CTC and full-recording KL preservation. Speech and real
pause frames both enter preservation. The control uses full-batch updates;
the discrete arm evaluates at most eight gradient-ranked proposals per step.
Defaults are 96 steps, a control learning rate of 0.001 and a two-code bound.
These are untested training defaults, never selected on practice. This is an
identical-data comparison, not an assertion of identical optimizer compute.
The one-day deadline begins with cache preparation and persists across commands.

Checkpoint selection is average transcript CTC on a deterministic, separate
group of eligible training speakers. The ordinary control includes its initial
snapshot among selections; the discrete arm includes the original codes.
No target IDs, reader IDs, verse references or practice incident scores occur
in the implementation.

The enrollment list must identify speakers, performed phone transcripts and
audio checksums, with `split: train` and `held_out: false`. Sources can be
authorized transcripts or help-train none-labelled takes. Help entries also
require empty events, non-errors and unsupported labels. Any nontraining or
unresolved-held-out entry excludes that speaker before audio is read. Explicit
release exclusions are required separately, and duplicate audio is rejected.
No enrollment list has been built here. Caller-supplied eligibility fields are
a contract, not independent proof of the release exclusion; that proof must be
established from corpus enrollment when remote access is available.

Before writing a bundle, the runner replays cached frontend windows through
the grafted full model and checks its posteriors against the isolated head and
its states against the original encoder. Bundle writing requires the unchanged
SDK insertion teacher and its external weights, copies source auxiliaries with
checksum verification, and writes a generated primary manifest with its hash.
It leaves the alphabet, frontend, clock, SDK decoding, tracker, checker and
insertion verification unchanged. The runner does not declare `.autolab/model.json`;
that declaration requires a real, complete, validated model and the target audit.

## Validation

All 12 small CPU tests in `modal/test_student_export_effective.py` passed using
PyTorch 2.8.0, ONNX 1.19.0, ONNX Runtime 1.22.1 and NumPy 2.2.6 in an ignored
worktree-local environment. The final suite took 2.7 seconds. It checks:

- A synthetic useful float change changes posteriors but disappears under the
  actual existing int8 export recipe.
- Actual exported loss improves for accepted code moves; suggestions are all
  rejected when the exported evaluator finds no gain. Codes respect bounds
  and saturation, and rejected proposals do not mutate them.
- Extraction and grafting preserve carried encoder states. Per-chunk activation
  ranges and full final chunks differ observably from incorrect whole-recording
  or truncated evaluations.
- Streaming cache/drain counts, padding and product parity checks, including
  rejection of misaligned cached features.
- Speaker exclusion before enrollment, deterministic speaker separation,
  rejection of mistake/repeat takes, repeated-label CTC and pause preservation.
- End-to-end synthetic control and discrete refits, selected-snapshot exports,
  generated primary hashes and unchanged auxiliary bundle copies.

These tests establish executable optimizer/export behavior on tiny models.
They do not establish native Zipformer parity, real acoustic improvement,
authorized corpus enrollment, silence on clean sessions or release portability.
Python syntax compilation, CLI loading and final diff checks passed.

Four supplied-supervisor requests failed with
`NotFoundError: No such file or directory`, without command output:

1. Inventory the actual shipped graph, model bundle and available training data.
2. Run the required six-shard core/SDK build and test harness.
3. Run just `pwd`, confirming the environment cannot start a minimal command.
4. Run the new Python behavior suite remotely with the pinned dependencies.

No remote test pass is claimed. No alternate route to Modal or external
recordings was used. Scorer, lab tooling, test data, existing manifests,
evidence and the experiment card were not edited.

## Limits and remaining work

The five registered targets remain unresolved:

- `lab-v1.5-p07/phone/mistake:added_word/37:36:1/144.12`
- `lab-v1.5-p33/phone/mistake:added_word/26:208:1/156.16`
- `lab-v1.5-p56/phone/mistake:added_word/32:21:1/106.76`
- `lab-v1.5-p57/phone/mistake:added_word/51:18:1/95.26`
- `lab-v1.5-p79/phone/mistake:added_word/20:40:25/136.10`

Restore supervisor execution, inventory the model and eligible corpora, and run
the remote tests. Enroll authorized transcripts and none-labelled help training
takes with release-speaker exclusions. Prepare the native-feature cache and
run the ordinary control. Audit its selected floating and actual int8 artifacts
through the SDK before proceeding to the discrete arm. If the premise is
absent, revert the engine additions and record a killed outcome.

If the premise holds, train the discrete arm, verify the complete real bundle,
then inspect the five targets through the unchanged product warning path.
Fewer than three built-model target wins requires reverting and killing;
at least three requires an official practice score before deciding. Losses
elsewhere are left to the specified fold gate, with fully clean-session new
warnings retaining their hard-stop rule.

No real refit, acoustic target replay, replacement model or target-win count
exists. No overall or listener delta is claimed. No `.autolab/model.json` is
declared: decoding the main model would not evaluate this implementation.
Infrastructure failure is not evidence of a killed premise, so no killed
`.autolab/outcome.json` is written. Zero of the three practice scores were used,
and no fold or discard was performed.

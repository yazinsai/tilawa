# Transfer-weighted frozen phone head

Status: implemented training and export pipeline; Modal execution is blocked.
The experiment card is unchanged. No trained listener bundle is available yet.

## Prediction and mechanism

The card predicts two false warnings removed and one caught mistake lost:
listener −1 and overall −1. The hypothesis concerns destructive transfer between
speakers, rather than insufficient fit of the examples used for an update.

`modal/student_transfer_head.py` caches features from the shipped streaming ONNX
encoder and refits only its linear CTC phone classifier. The feature extractor
in `modal/student_head_refit.py` exposes the exact rounded activations consumed
by an integer head. It verifies the original phone outputs and all encoder state
outputs against the unchanged graph. Whole recordings use the deployed chunk,
padding and cache geometry, with final padded output frames trimmed away.

The bounded recipe admits up to 512 authorized original clean-match recordings
and help-train training speakers' none-labelled takes. Original targets retain
the shipped spliced-target recipe, including teacher-observed recording edges;
none-labelled targets come from their authorized canonical ayahs. Each recording
has one ordinary view and two views using the existing student augmentation
functions. Speech and in-recording pauses remain in the cache. Gold/excluded
original speaker clusters and excluded/non-training help speakers are removed
in full. Whole training speakers are reserved for checkpoint selection, and
identical waveform copies under different speaker identities abort preparation.

Mistake and repeat recordings provide only shipped-output preservation targets,
including otherwise `none` takes with repeats recorded in `non_errors`.
Their scripts never become performed transcripts, phone labels or query targets.
They follow the same whole-speaker separation as the clean recordings.

For every update, supervised training speakers are divided into disjoint support
and query groups. Every cached supervised view supports once per epoch; group
roles and query recordings rotate. The query sample takes at most one recording
view per speaker. At the current head parameters, let `g_i` be a support example's
length-normalized CTC gradient. The virtual update is

`theta'(epsilon) = theta - lr * (g_preserve + sum_i epsilon_i g_i)`.

The negative query-loss derivative with respect to an example's weight at zero
is `lr * dot(g_query, g_i)`, with the query gradient evaluated after the virtual
preservation step. Positive derivatives receive normalized weight; a batch with
no positive transfer receives no transcript update. The ordinary preservation
update still applies. The chain rule is evaluated explicitly, avoiding an
unsupported second backward through PyTorch CTC. This is a differentiable
one-step bilevel weighting rule, not weighting by the support's own training loss.

Preservation recordings belonging to query speakers are excluded from each
update, so even transcript-free anchors cannot contaminate the transfer test.
Both arms use the same resulting preservation batch.

The weighted and uniform arms start from the same shipped classifier, consume
the same support batches and augmentation cache, use the same preservation bank
and SGD settings, and take the same number of optimizer steps. Only transcript
weights differ. Both preserve shipped probabilities across every support and
preservation frame. Twelve epochs and the other recipe settings are frozen in
code before any target queries. Reserved-training-speaker CTC plus preservation
loss selects the weighted checkpoint; the uniform arm is retained at that exact
epoch and update count. Cache checksums, speaker partitions, per-example transfer
benefits, weights and selection provenance are recorded in persistent artifacts.

Export replaces only phone-head constants, preserving graph nodes, all other
tensors, inventory, frame timing, streaming states and warning requirements.
Per-tensor or per-channel int8 weight quantization keeps its original geometry.
Both selected arms are exported for the comparison. The ordinary SDK directory
loader can load these unchanged-interface graphs; no tracker, checker, SDK
decoding rule or confidence requirement is modified.

Export also requires the complete inherited SDK auxiliary bundle, including the
folded insertion teacher and its external weights, with their registered sizes
and checksums. Missing assets abort export. Supplied manifests are only read;
new bundle manifests receive the new model checksum and size.

## Validation

All 15 synthetic unit tests passed locally with PyTorch 2.15.0, ONNX 1.23.2 and
ONNX Runtime 1.31.1:

```sh
.autolab/transfer-test-env/bin/python -m unittest discover -s modal -p 'test_student_*head*.py' -v
```

The suite covers whole-speaker exclusions, full per-epoch support coverage,
support/query separation, preservation updates excluding query speakers, and
the bilevel derivative against a finite-difference virtual update,
self-fit improving while cross-speaker fit worsens, harmful-batch rejection,
uniform recovery for equal-transfer examples, whole-speaker episode separation,
complete per-epoch support coverage, matched optimizer counts, training-only
selection, preservation controls without transcripts, and frozen-cache audits.
ONNX fixtures additionally cover float/int8 head reconstruction, per-channel
quantization, final-padding trimming, encoder-cache parity, head-only changes,
and complete auxiliary-bundle validation. These fixtures are not target results.
The bounded refit uses direct first-order SGD updates, with no momentum or
optimizer state. Python compilation and command-line import checks passed.
The staged diff passes `git diff --cached --check`.

Supervisor Modal requests 1 and 2 attempted read-only input inventories in the
normal and training images. Request 3 attempted the required six-shard core/SDK
build-and-test harness. Request 4 attempted the new Python unit suite followed by
prepare, fit and export. Request 5 retried the six-shard core/SDK harness;
request 6 retried the Python suite and training/export pipeline after reviewing
speaker separation. All six returned `NotFoundError: No such file or directory`
before container execution. No remote command log or real input
inventory was produced. This does not establish missing training data, a failed
acoustic premise, or passing core/SDK tests.

## Limits and remaining work

No real feature cache, cross-speaker weights, matched refits, training-selected
checkpoint or complete exported model has been produced. The actual training
schemas, speaker coverage, runtime, output reconstruction tolerances and
inherited teacher bundle have not been verified on the supplied inputs.

The target comparison is still required: use the training-selected weighted and
matched uniform exports on identical full-stream, engine-requested evidence,
and judge them with the unchanged tracker/checker. The card asks for at least
two removed target warnings that remain under the uniform refit; the task also
requires at least three target recoveries before practice scoring. Neither
condition has been measured. Target wins are unknown, not zero, and no kill
verdict is justified by the infrastructure failure.

After Modal execution is restored, run the final code's unit suite and input inventory first,
then prepare, fit and export under the training image. A complete installed
bundle must be supplied to export, rather than silently dropping the insertion
teacher. The intended persistent output root is
`/scratch/listener-transfer-weighted-head/refit-v1`; its `selection.json` and
`export-report.json` establish the chosen heads and bundle checksums. Verify the
normal Node SDK loader and paired target decisions before declaring
`{"path":"refit-v1/weighted-bundle"}` in the ignored `.autolab/model.json`.
If the target prerequisite passes, run the authorized practice score and let
the fold gate assess losses and gains.

All three practice scores remain available. No population gain, clean-session
safety or studio regression is claimed. There is no `.autolab/model.json` or
killed outcome: a main-model decode would not evaluate this refit. The retained
code implements the experiment for continuation, but does not yet change the
model the scored engine hears. No scorer, lab tooling, supplied data, evidence,
experiment card or tracked manifest was edited. No fold was performed.

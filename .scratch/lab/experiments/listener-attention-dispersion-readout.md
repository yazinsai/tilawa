# Attention dispersion readout

Status: needs-human. Implementation is present; the supervisor has not executed
model/data inspection or the required tests. No trained bundle, acoustic target
result or practice score exists. The experiment card is unchanged.

## Prediction and mechanism

Predict one false wrong-word warning removed, with no losses: listener −1,
overall −1. Attention's weighted mean can cancel opposing value-vector cues.
The proposed readout retains each selected head's weighted value variance beside
its existing mean, without altering attention weights, encoder computation,
primary phones or posteriors.

`modal/student_attention_export.py` adds parallel nodes to a separate copy of the
supplied streaming graph: `A @ (V * V) - (A @ V) * (A @ V)`, clamped at zero only
for floating-point subtraction. Both moments use the existing attention weights
and the complete existing value tensor, including its streaming cache. Padding
remains governed by the original attention mask. All heads in the final encoder
stage's ordinary value-attention branches form the architectural candidate set.
Streaming methods omit attention module scopes in ONNX node names. Selection
therefore uses completed-stage combiners and the ordinary `cached_val1`/
`cached_val2` input wiring, excluding nonlinear-attention caches. Statistics
normalize the singleton batch axis to `[heads, frames, channels]` without changing
values. The exporter checks normalized nonnegative attention, independently recomputes
variance, and requires bitwise parity of primary output and every recurrent state
across nine sequential windows, including the final serialized extractor before writing extractor metadata.

`modal/student_attention_dispersion.py` trains phone-conditioned full Gaussian
emissions on cached help-train training-speaker `none` takes. Both dispersion and
mean-only arms have sixteen projected coordinates and identical parameter count,
training iterations and boundary topology. Their mean projection basis is shared;
the dispersion arm additionally projects value variance. Fixed seeded projections
are chosen before training or target inspection. Three grouped speaker folds
select each arm's covariance regularization from a fixed grid and select one head
using the two arms' average complete-pronunciation contrast loss. Contrasts modify
training transcripts only; practice examples never choose any training setting.
The final arms are refit on eligible training speakers and frozen together.

Complete-pronunciation likelihood sums CTC phone-boundary paths. Legal outgoing
arcs are normalized; conditioning the timing prior on sequence completion removes
path-count advantages. Shipped posteriors initialize soft training occupancies,
then EM uses each fitted arm's own marginalized boundaries. Preparation excludes
every speaker with any non-training take and requires an explicit release-speaker
exclusion file; the trainer independently rejects missing speakers, excluded
speakers, mistake takes, non-training splits and invalid statistics.

The SDK requests only complete written and already-accused pronunciations on the
unchanged checker's existing supported string-path windows. A finite strict
reversal is required: dispersion prefers written, mean-only prefers rival. Core
matches both strings, word identity and exact frame bounds before downgrading the
public warning to uncertain. Internal confirmation/ownership state remains intact
to preserve later candidate generation and insertion merges. Graph confirmations
are exempt. The Node loader verifies the side-file declarations and hashes,
matched model dimensions and exact primary checksum; declared missing assets fail
loading. Ordinary models without a declaration use their existing path.

Audio-backed sessions compute the secondary pass at completion, retain certificates
for stored-evidence replay and publish explicit downgrades only for exact live
occurrences also removed by replay. The secondary graph has independent caches.
This buffered completion route has not demonstrated immediate warning latency or
device memory/CPU suitability.

## Validation and limits

Ten Python unit tests pass. They cover exhaustive boundary paths, normalized soft
occupancies, path-count cancellation, covariance regularization, masked cached
values, opposing-value cancellation, matched arm dimensions, unscoped ordinary
value-attention identification and rejection of non-training/mistake/excluded
speaker data. All five new core behavior tests pass. Both core and SDK TypeScript
builds pass. The full local SDK suite passes: 37 tests passed and one optional
help-data test skipped. This includes existing primary listener/teacher tests and
new certificate transport, loader integrity and session downgrade tests. The full
core suite remains uncertified because the prescribed remote harness never ran.

The repository's random-weight tiny Zipformer fixture successfully exports four
candidate heads from its two ordinary value-attention branches. Across nine
sequential windows, primary output and every recurrent state are bitwise identical;
the final serialized extractor's normalized statistics are bitwise identical to
the raw diagnostic statistics after reshaping. Weighted variance matches an
independent double-precision calculation with maximum absolute error
`3.5768779920175575e-06`. The first fixture check exposed missing module scopes and
four-dimensional attention tensors; both assumptions were corrected and the final
serialized graph was verified. This checks the extraction implementation on the
shared architecture, not the deployed quantized model or target acoustics.

Supervisor requests 1 and 2 attempted read-only model/data inspection. Both failed
before command execution with `NotFoundError: No such file or directory`, zero
compute cost and no command output. Request 3 retried the infrastructure with
`pwd`; request 4 requested the prescribed six-shard `tools/autolab_tests.sh` harness.
Both also failed before execution with the same error and zero compute cost.
No remote command executed in any of the four requests. The supervisor blocker has been reported to the user.

No supplied help-train label, primary graph or practice audio has been inspected.
The preparation utility's normalized `speaker`, `split`, `kind`, `wav` and `ayahs`
fields remain to be checked against the supplied labels. The deployed graph's
attention names, layouts and clock ratios remain to be established. Head
selection, fitted parameter stability, SDK/Python acoustic-score parity, target
comparisons and the complete model bundle remain unvalidated.

No `.autolab/model.json` or claimed completed side bundle is declared. Running the
main listener does not evaluate this experiment. Target comparisons: zero;
practice scores: zero of three. No fold/discard or premise-based kill is claimed.
After the supervisor is repaired, inspect and verify the source graph and eligible
labels, cache statistics, train/freeze both matched arms, verify the complete
bundle and inspect the four registered checker-requested comparisons. Apply the
user's continuation rule: fewer than three recovered targets kills the experiment;
at least three requires the official practice score before any gate decision.

Required bundle files are the unchanged primary model and manifest plus existing
primary auxiliary assets, `attention-dispersion.onnx`,
`attention-dispersion-emissions.json` and `attention-dispersion-asset.json`.
Training/parity reports record provenance. No scorer, lab tooling, supplied
manifest, evidence, test data or experiment-card file was edited.

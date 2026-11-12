# Activation-aware low-rank teacher inheritance

Status: blocked before implementation by the supplied remote execution route;
confirmed by minimal retries. No trained model, engine implementation or
accuracy result is claimed. The experiment card is unchanged.

## Prediction and mechanism

Predict two recovered added words and one collateral wrong call: listener −1,
overall −1 on the practice set. The proposed failure is loss of teacher
consonant features during distillation into a separately initialized narrow
student. Inherit the teacher's hidden dimensions, nonlinearities, normalization
and residual connections instead, replacing its large linear operators with
two dense low-rank factors.

For an operator with weight W and calibration input second moment C, the
initialization should minimize ||(W − AB) C^(1/2)||_F². A truncated SVD of
W C^(1/2), followed by unwhitening, gives that solution for positive-definite C.
Calibration must include padding exclusion and numerical treatment of singular
moments; it must use authorized training audio only. Keep the original output
bias rather than treating the bias as a second low-rank projection. Retain the
two products in export instead of materializing their dense product.

Fine-tune the factors with transcript CTC and frozen-teacher output
preservation. Include ordinary recitation and real pauses, and only help-train
training speakers' none-labelled takes as additional transcribed audio.
Mistake scripts are planted labels, not verified performed transcripts.
Release-held-out speakers must not enter calibration, fitting or selection.
Compare against a conventionally distilled student matched to the exported
storage budget and training compute; select ranks and checkpoints using
speaker-grouped partitions of authorized training speakers only.

The acoustic kill criterion requires at least two targets retaining a
teacher-supported consonant missing from shipped decode after factorization
and matched-compute fitting. The executor instructions additionally require
three target wins before proceeding to a practice score. Neither test has run.

## Repository audit

`modal/student_teacher_export.py` loads the fp32
`obadx/muaalem-model-v3_2` teacher and verifies its phoneme inventory against
the SDK registry. Its export takes `input_features` and `attention_mask`;
it exposes no recurrent state. `packages/sdk/src/insertion-teacher.ts` supplies
Nyquist-band mel features with per-bin clip mean and unbiased-variance
normalization, then packs pairs of frames into 160 channels. The auxiliary
teacher evidence uses the same nominal 40 ms output clock as the student.

The primary listener contract differs. `modal/student_export.py` exports a
causal Zipformer with explicit acoustic caches, 80-channel features, a
45-feature-frame window and a 32-frame advance for the default chunk size.
`packages/sdk/src/listener.ts` constructs those inputs and advances states.
`packages/sdk/src/model.ts` currently declares only that streaming architecture.
A factorized teacher cannot be installed merely by replacing `model.onnx`.
It needs a tested streaming execution/frontend adapter and the existing
timed-phone evidence interface. A stateless whole-session teacher would not
establish streaming compatibility.

The earlier `exp/listener-sparse-teacher-subnetwork` report records an audit of
the teacher configuration and Transformers attention implementation: a
24-layer Wav2Vec2-BERT Conformer, width 1024, feed-forward width 4096, and
bidirectional attention. This was read as prior evidence, not reverified
against loaded weights here. The main-line teacher insertion report records
a 2.43 GB auxiliary fp32 bundle; that auxiliary export does not establish
low-rank storage or primary-listener viability.

`tools/student_manifest.py` includes speaker-cluster IDs in prepared Quran
training rows. `modal/student_train.py` restricts loading by bucket/split and
excludes gold PER training clips. A new training-only selection partition
should group those clusters, rather than reuse release evaluation speakers.
`tools/help_import.py` records explicit speaker, split, kind, exclusion and
none/repeat/error fields. Actual remote schemas and the available training
speakers still need to be inspected before using these records.

## Execution and validation limits

On 2026-11-12, all four requests through the supplied supervisor failed
with `NotFoundError: No such file or directory`:

| Run | Request | Result |
| --- | --- | --- |
| 1 | Train-image environment/data probe; one CPU shard, ten-minute limit | NotFoundError; no command output or shard log |
| 2 | Required `bash tools/autolab_tests.sh`; six CPU shards | NotFoundError; no command output or shard log |
| 3 | Shell-only `pwd`; one CPU shard, two-minute limit | NotFoundError; no command output or shard log |
| 4 | Train-image Torch import; one CPU shard, three-minute limit | NotFoundError; no command output or shard log |

All four result files report $0 charged and the full $180 GPU budget
remaining. The returned error does not identify the missing path; it does
not establish which volume, image or staging step failed. No GPU job has
been requested.

Local Python lacks Torch, Transformers and ONNX. NumPy is installed but fails
to import because its extension is x86_64 while the interpreter requires ARM.
A small synthetic reconstruction check stopped at that import; no algebraic
or model numerical test passed. The worktree has no node_modules. Heavy
local training and alternative remote access are outside the authorized
execution arrangement.

No engine behavior changed, so no new behavior tests were added. Core and SDK
tests remain unverified: run 2 failed before the harness produced any log.
No calibration, rank selection, factor fitting, matched control, export,
SDK parity check or target diagnosis has run. No practice score has been
requested; all three opportunities remain. The acoustic premise is untested,
so infrastructure trouble does not justify a killed outcome or a measured
zero gain. No `.autolab/outcome.json`, model bundle or `.autolab/model.json`
is declared. Scoring the main model would not evaluate low-rank inheritance.

Activation-aware inheritance depends on actual teacher activations and
operator weights, not just the published layer dimensions. The unavailable
runtime also prevents inspecting the speaker metadata, fitting the matched
control, checking factorized ONNX storage and proving streaming numerical
parity. Consequently the requested complete engine change cannot be built
and verified through the currently available route. An untrained replacement
or an unused factorization helper would not implement this experiment.

Completion requires restored access to the supplied training runtime, teacher
weights and speaker metadata, then numerical factor/reconstruction tests, matched-arm
training and training-only selection. A surviving candidate requires a
complete verified bundle, streaming SDK parity, passing core/SDK tests and
the prescribed target and practice validation before the fold gate can judge
it. No scorer, lab tooling, supplied manifest, test data or evidence was edited.

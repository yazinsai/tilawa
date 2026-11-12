# Sparse teacher subnetwork

Status: blocked before training or implementation. The acoustic hypothesis remains
untested; this is not a killed experiment or a measured zero gain. The experiment
card is unchanged.

## Prediction and mechanism

Predict two recovered additions and one collateral wrong call: listener −1,
overall −1. Dense compression may lose rare consonant-feature interactions that
survive in the original teacher's wider representation.

The proposed experiment gradually prunes the original teacher while retaining
its layer widths, depth and residual connections. Surviving connections must be
selected by their contribution to transcript-conditioned phone-occurrence
likelihood, rather than magnitude alone, and retrained on authorized transcribed
recordings and help-train training speakers' none-labelled takes. Release-held-out
speakers must be excluded. An equal-storage dense-distillation control must use
matched training data and compute, with checkpoint selection confined to training
partitions. A complete sparse primary-listener bundle must preserve the inventory,
40 ms posterior clock, streaming context and downstream warning rules.

The card's comparison requires at least two target intervals with a recovered
planted consonant relative to that matched dense control on full-session decoding.
The executor instructions additionally require at least three target wins before
continuing to a practice score. Neither comparison has been run.

## Read-only architecture audit

`modal/student_teacher_export.py` identifies the original teacher as
`obadx/muaalem-model-v3_2`. Its published configuration identifies a
Wav2Vec2-BERT multilevel CTC model with 24 Conformer layers, hidden width 1024,
feed-forward width 4096, and 16 attention heads. The existing export takes
`input_features` and `attention_mask` and supplies no recurrent state.

`packages/sdk/src/insertion-teacher.ts` computes the teacher's Nyquist-band mel
features, normalizes each bin using the whole clip's mean and unbiased variance,
and packs pairs of feature frames into 160 channels. The teacher's adapter
subsamples again, giving the same nominal 40 ms clock as the student. This
frontend differs from the primary listener's raw 80-bin features.

The Transformers 4.55.2 Wav2Vec2-BERT source confirms that the encoder attention
mask excludes padding, not future frames. The configured left/right position
limits clip relative-position indices; they do not limit the attended context.
The Conformer depthwise convolution is causal, but encoder and adapter attention
are bidirectional. These findings mean that a sparse export needs an explicit
streaming execution adapter; merely zeroing teacher weights and replacing the
primary ONNX file would not preserve the current streaming context.

The current primary listener uses a causal Zipformer, a 45-feature-frame input
window, a 32-frame advance, and cached acoustic states. Its manifest and backend
interface cannot directly consume the existing stateless teacher export. The
previous teacher-insertion experiment's write-up also records a 2.43 GB fp32
auxiliary bundle, which does not establish viable sparse storage or primary
streaming execution.

These are implementation prerequisites, not evidence that sparse compression
cannot recover the target consonants. No accuracy conclusion follows from the
architecture mismatch alone.

Sources inspected: the engine files above, `modal/student_train.py`,
`modal/student_export.py`, `packages/sdk/src/model.ts`,
`packages/sdk/src/listener.ts`, and
`.scratch/lab/experiments/listener-teacher-insert-marginal-gate.md`. Public model
configuration: <https://huggingface.co/obadx/muaalem-model-v3_2/raw/main/config.json>.
Pinned attention/convolution implementation:
<https://github.com/huggingface/transformers/blob/v4.55.2/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py>.

## Execution and validation limits

On 2026-11-12, the supplied supervisor failed both initial read-only requests,
a shell-only retry, and the required test harness:

| Run | Request | Result |
| --- | --- | --- |
| 1 | Default-image listing of the supplied model and scratch directories; 1 shard, 2 min limit | `NotFoundError: No such file or directory` |
| 2 | Train-image package availability and supplied training/model directory listing; 1 shard, 10 min limit | `NotFoundError: No such file or directory` |
| 3 | Shell-only `pwd`; 1 shard, 2 min limit | `NotFoundError: No such file or directory` |
| 4 | Required `bash tools/autolab_tests.sh`; 6 shards, 30 min limit | `NotFoundError: No such file or directory` |

All four runs returned no command output or shard log. Their result files report
$0 charged, with the full $240 GPU budget remaining. The returned error does not
identify the missing path, so its cause cannot be attributed to a particular
volume or staging step. Supervisor restoration or a fuller error log is needed
before the data, model, training and runtime checks can proceed.

No engine behavior has been changed. No pruning, fitting, export, target decode
or practice score has run; all three practice-score opportunities remain. Core
and SDK tests have not been verified. No model bundle or `.autolab/model.json`
has been declared, and no killed outcome has been written. A run of the main
listener would not evaluate this experiment.

After execution is restored, the next prerequisite is to verify the original
teacher under the required streaming geometry and the intended sparse operators,
then build and test occurrence-based pruning and the matched dense control.
Training must validate speaker exclusions and use training-only selection before
exporting both arms and checking the full-session target comparison. A surviving
arm still needs a complete verified bundle, passing core/SDK tests and the
required practice score before it is ready for the fold gate.

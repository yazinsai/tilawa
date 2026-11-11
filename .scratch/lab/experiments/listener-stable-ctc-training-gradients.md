# Stable CTC training gradients

Status: blocked by the supervisor runtime; the numerical premise is unresolved.
The experiment card and engine remain unchanged. This is not a killed experiment
or a completed model refit.

## Prediction and mechanism

The card predicts listener -1 and overall -1, with substantial zero-gain risk.
Its proposed cause is loss of legitimate alternative-alignment gradients through
underflow or cancellation in transcript-conditioned CTC training. The proposed
intervention retains forward/backward accumulators and their gradient combination
in stable log-space arithmetic outside reduced-precision execution, preserving
the objective, valid lengths and recording weights.

Before changing training, the card requires at least two target incidents with
a missing planted consonant whose original-arithmetic training gradient vanishes
or changes sign against a high-precision reference on identical logits. Target
gradients are diagnostics only. A subsequent bounded phone-head refit must use
cached frozen-encoder features from authorized transcribed recordings and
help-train training-speaker none-labelled takes, with an identical-data,
identical-update-count original-arithmetic control. Training-speaker partitions
alone select checkpoints; release-held-out speakers and practice targets cannot
contribute updates or checkpoint selection.

## Source findings

Audited `modal/student_train.py`, SHA-256
`e5b2dc0d99880b686f562a80d9b28714ba2dd086609afc579cac02b67a5cb448`.
`losses` evaluates the model within bfloat16 CUDA autocast, then casts its log
probabilities to float32 and calls native `F.ctc_loss` outside that context.
The loss uses the returned per-recording output lengths, the supplied transcript
lengths, blank 0, summed reduction and `zero_infinity=True`, divided by the sum
of valid output frames. Optional KD retains its existing valid-frame mask and
configured weight. There is no direct bfloat16 CTC accumulator in this source.

This does not disprove the card's cancellation premise. The separately inspected
upstream PyTorch v2.8.0 native CPU and CUDA CTC sources use max-shift log-space
forward/backward sums, but combine probability and alignment occupancy by
subtraction in their backward kernels. Close terms can lose precision. These
upstream sources are contextual evidence, not a verified identification of the
unavailable supervisor runtime or proof of a defect on any target.

## Validation and limits

The read-only source/AST audit passed seven contract checks. An independent
float64 NumPy CTC reference passed 25 feasible short-sequence comparisons against
exhaustive path enumeration and 294 logit finite-difference checks. Cases cover
empty transcripts, repeated labels, skip-transition restrictions and impossible
alignments. The reference retains log-space state occupancy and uses signed
`expm1` subtraction for close probability/occupancy terms.

A synthetic sensitivity control produced float32 zero versus a float64 signed
difference of approximately -1.35e-9 for two deliberately nearby log masses.
This is neither an actual native-PyTorch gradient measurement nor a measured
target failure. It does not establish that such differences survive the
log-softmax Jacobian, head gradients or optimizer updates. A target audit must
compare the complete upstream training-gradient chain on identical logits;
leaf log-probability differences alone are insufficient.

The small reference checks ran locally without training or acoustic inference.
Their disposable scripts remain ignored at `.autolab/ctc_reference.py` and
`.autolab/ctc_source_audit.py`. Local PyTorch and the core/SDK dependencies are
unavailable.

All three authorized supervisor requests failed before command execution with
`NotFoundError: No such file or directory`:

| Request | Requested work | Result |
|---|---|---|
| 1 | Inspect PyTorch/runtime and read-only lab-data layout | No execution |
| 2 | Six shards of `bash tools/autolab_tests.sh` | No build or tests ran |
| 3 | Inspect the training image and read-only lab-data layout | No execution |

The ignored `.autolab/modal-request-{1,2,3}.json` and corresponding result files
retain the requests and errors. No alternative Modal access was attempted.
Core and SDK tests are unverified, not passing. No permanent unit test was
added because no engine behavior changed.

## Outstanding work

The five target incidents have not been acoustically or numerically evaluated.
No target recovery count, practice delta or satisfied kill criterion can be
reported. There were zero practice-score calls. No refit, checkpoint selection,
export or model-bundle verification ran; no `.autolab/model.json` is declared.
The write-up is the only tracked change, and no killed outcome is recorded.

Resume after the supplied supervisor runtime is restored: establish missing
consonants and compare original native gradients with the high-precision
reference, including the actual head/log-softmax chain. If fewer than two
targets meet the numerical premise, record a premise kill. Otherwise implement
and unit-test the stable loss, run the matched training-only head refits, select
within training speakers and verify a complete export. Evaluate target recovery
before spending a practice-score call; three or more target wins require that
score before a gate decision. Infrastructure failure is not evidence for killing
the hypothesis, and scoring the unchanged main model would not evaluate it.

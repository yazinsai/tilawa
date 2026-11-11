# Cross-chunk training credit assignment

Status: killed on premise; no engine change or model refit retained. The experiment card is unchanged.

## Prediction and mechanism

Predict one recovered addition and no losses: listener −1, overall −1 on
lab-v1.5u. The proposed explanation is that detached upper-block streaming
caches preserve forward context while preventing later losses from teaching the
representations that created those caches. The planned intervention is
checkpointed recomputation through adjacent chunks in a bounded upper-block and
phone-head refit, with lower layers frozen and an identical-data detached-cache
control. Training would use authorized recordings and help-train training
speakers' none-labelled takes, excluding release-held-out speakers, with
checkpoint selection restricted to training-speaker partitions.

The card explicitly stops if training already propagates gradients through
retained upper-block context. That stop applies before training or evaluating
planted additions.

## Premise audit

The repository's actual training path does not execute a recurrent chunk loop:

- `modal/student_train.py:134`: `Student.forward` embeds the complete padded
  recording and calls `self.encoder(...)` once with its valid lengths and mask.
- `modal/student_train.py:302`: `losses` calls that model once, then computes
  recording-level CTC and valid-frame teacher distillation on its outputs.
- `modal/student_train.py:460`: training calls `losses` once per batch and calls
  `out["loss"].backward()` before the optimizer update. It does not carry or
  detach any upper-block cache between chunks.
- `modal/student_export.py:56`: explicit `streaming_forward` cache transport
  belongs to the deployment/export wrapper, not the training forward above.

Inspected the exact icefall revision pinned by `modal/student_train.py`:
`3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`. Its
[Zipformer source](https://github.com/k2-fsa/icefall/blob/3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1/egs/librispeech/ASR/zipformer/zipformer.py)
confirms the graph semantics:

- `Zipformer2.forward`, lines 300–368, builds one attention mask and passes the
  full tensor through each encoder stack. It retains the outputs in the same
  autograd graph.
- `_get_attn_mask`, lines 370–411, masks future chunks and chunks outside the
  selected left-context extent. It does not detach allowed earlier chunks.
- `RelPositionMultiheadAttentionWeights.forward`, lines 1602–1755, projects keys
  and queries from that tensor and computes their matrix product before applying
  the mask and differentiable softmax.
- `SelfAttention.forward`, lines 1922–1959, projects the values and multiplies
  them by those attention weights. A later query's loss therefore differentiates
  through the earlier values and keys that query is allowed to use.
- Even the separate streaming attention implementations preserve cache graphs:
  key transport at lines 1801–1803 and value transport at lines 1994–1996 use
  `torch.cat` followed by slicing, without detachment.

An AST scan found **zero `detach` calls in the pinned `zipformer.py`**.
The pinned `scaling.py` contains detach operations inside custom autograd
implementations and diagnostic fixtures; those are not a chunk-cache detachment
boundary. Attention's custom softmax defines a backward operation. A text
search for detach alone would not distinguish these cases.

Source SHA-256 values for reproducibility:

| Source | SHA-256 |
| --- | --- |
| Repository `modal/student_train.py` | `e5b2dc0d99880b686f562a80d9b28714ba2dd086609afc579cac02b67a5cb448` |
| Pinned upstream `zipformer.py` | `4d5365ed19dee07f5a51fcaaa26e18310f7fafbdf2a391e24a7cc01d1b2e8cfe` |
| Pinned upstream `scaling.py` | `d5a5db141cd00b4beaa990fe6827599b3248b2fddd87e8dc7809b0dbd9d6633b` |

## Result and limits

The detached-training-cache premise is absent. Masked full-recording training
already permits backward credit through the same earlier upper-block tensors
that supply its allowed forward context. Checkpointing could reduce activation
memory, but would not restore an interrupted gradient path here. Introducing a
new recurrent training/refit regime would change the forward computation rather
than repair the specified defect while preserving it, so no such intervention
was made.

Targets won: **0**. This is a premise-stop count, not a measurement of trained
model recovery: no refit, target-session decode, or practice score was performed.
No conclusion is drawn about whether another training objective would improve
the targets. All three permitted practice scores remain unused.

The supplied execution service was requested for pinned-source inspection and
runtime dependency availability (Modal run 1) and the unchanged six-shard
core/SDK build/test harness (Modal run 2). Neither request returned a result
before this experiment was closed; these checks are unverified, not passes.
A method-level gradient probe was prepared but was not executed. Static source
inspection is the basis of the kill; it does not claim a runtime gradient
experiment or an acoustic recovery result. No unit-test behavior changed,
because no engine implementation was changed.

No engine changes needed rollback. No model bundle or `.autolab/model.json` is
selected, and no scorer, lab tooling, supplied data, evidence, manifest or card
was edited. No fold was performed. The ignored `.autolab/outcome.json` records
the premise-based killed outcome.

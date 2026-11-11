# Training/export convolution tap audit

Status: killed — no convolution tap-order, dilation, grouping or weight-axis
discrepancy was demonstrated in the pinned training/export path. Engine code and
the experiment card remain unchanged.

## Prediction and proposed mechanism

The card predicts one recovered addition, no losses, listener −1 and overall −1
on lab-v1.5u, with listener
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. Its proposed cause is an
exported temporal filter applying learned taps to different relative input
positions than the training operator, losing brief consonant transitions. A
repair is authorized only for a demonstrated operator discrepancy, preserving
weights, receptive field, frame clock, cache boundaries and all decoding and
warning rules.

The five registered phone targets are p07 `37:36:1` at 144.12 s, p33 `26:208:1`
at 156.16 s, p48 `12:76:1` at 39.98 s, p56 `32:21:1` at 106.76 s, and p57
`51:18:1` at 95.26 s. No target-specific engine logic was introduced.

## Source contract

The starting commit is `f7697c89`. `AGENTS.md` was read first.
`modal/student_train.py` pins Icefall to
`3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1` and PyTorch to 2.8.0. The exact
pinned dependency source was downloaded into ignored diagnostic storage.

Both PyTorch and ONNX use cross-correlation. With temporal stride S, left
padding P and dilation D, weight tap k addresses input `t*S - P + k*D`.
The weight axes are `[output channel, input channel within group, temporal tap]`
for Conv1d, with a trailing frequency-tap axis for Conv2d. No tap reversal is
needed to lower that contract.

In Icefall's `ChunkCausalDepthwiseConv1d`, let `h = K//2` for the odd kernel
size K. The causal filter has h+1 taps, stride/dilation 1, no internal padding,
and groups equal to channels. Training prepends h zeros at utterance startup;
streaming prepends the h cached input frames. Both invoke the same
`self.causal_conv`, so its oldest-to-newest taps address offsets `−h` through 0.
The other filter has K taps and symmetric padding h, also depthwise with unit
stride/dilation. Both paths invoke the same `self.chunkwise_conv` on the current
chunk and apply the same `_get_chunk_scale`. Its offsets are `−h` through `+h`,
with zero padding outside the chunk. Training's chunk reshape does not reverse
time or move channels into the tap axis.

The embedding's ConvNeXt filter is depthwise 7×7. Training uses `nn.Conv2d`
with padding `(3,3)`. Streaming prepends three cached time frames, includes
three right-context frames, and calls `F.conv2d` with the identical weight,
bias and groups, padding `(0,3)`. For valid emitted frames, tap k still
addresses temporal offset `k−3`. Its pointwise filters remain 1×1.
The three initial 3×3 subsampling filters are shared between training and
streaming, with strides `(1,1)`, `(2,2)`, `(1,2)` and padding `(0,1)`, zero,
zero respectively. All these convolutions have dilation 1.

`modal/student_export.py:wrap` invokes these original streaming operators.
Icefall's `get_onnx_export_kwargs` selects the legacy PyTorch exporter, and
`export_onnx` uses opset 17. PyTorch 2.8.0's `_convolution` symbolic lowering
passes `[input, weight, bias]` to ONNX `Conv`, derives the kernel shape from
`weight_size[2:]`, and passes stride, doubled symmetric padding, dilation and
groups directly. It applies no weight flip or axis transpose.
`tools/quantize_onnx.py` quantizes MatMul only, leaving Conv weights fp32.
`packages/sdk/src/ort.ts` supplies tensors to ONNX Runtime directly; there is
no SDK convolution rewrite or weight-layout adapter to correct.

## Lightweight operator check

A read-only local diagnostic inspected the committed tiny ONNX fixture, without
running the acoustic model. It checked every Conv node's kernel shape, pads,
strides, dilation and groups against the independently read source contract.
All 30 matched: six embedding/pointwise filters and 24 encoder depthwise filters.
The tiny encoder uses kernel size 3, rather than the primary model's 15/31.

Each isolated original Conv node was evaluated with ONNX 1.19.0's reference
evaluator on small deterministic random inputs (NumPy seed 0) and compared with
an independent float64 NumPy grouped cross-correlation. Absolute and relative
tolerances were both `1e-5`; maximum absolute error over all operators was
`1.2704370917493435e-6`. Reversing the temporal weight axis was a negative
control: all 28 filters with multiple temporal taps produced discrepancies,
with maximum errors per filter ranging from 1.8804 to 4.4044.

This check validates the fixture's ONNX operator attributes and semantics using
exported weights. It does not independently compare those weights with a
training checkpoint, and cannot alone detect weights reordered before export.
The source-lowering inspection addresses that distinct possibility. Neither
check is a PyTorch-versus-ONNX Runtime replay on target inputs and state.

Diagnostic provenance:

- Icefall `scaling.py` SHA-256:
  `d5a5db141cd00b4beaa990fe6827599b3248b2fddd87e8dc7809b0dbd9d6633b`.
- Icefall `subsampling.py` SHA-256:
  `e04de281f136560e2d44b5f7df174d0364b5feb2c22608173b60de02630c2a66`.
- PyTorch v2.8.0 `torch/onnx/symbolic_opset9.py` SHA-256:
  `33b8b45e0d61a0e0d253cd92b1faf3e3c1ee9a19d531c6d6f067139362925564`.
- Tiny ONNX SHA-256:
  `e28354258202150bd4447470e3a8d9ff5a7a70dadf3113941794ed52bad4c628`.
- Reproduction in this worktree's diagnostic environment:
  `PYTHONPATH=.autolab/audit-python python3 .autolab/audit-taps.py` using the
  Python 3.13 interpreter used to install ONNX. Source, dependencies, script
  and `convolution-audit/result.json` remain ignored under `.autolab/`.

## Validation limits and outcome

Supervisor run 1 requested dependency and authorized-data inspection. Run 2
requested inspection of the shipped primary model's Conv nodes/initializers
and practice-data roots. Run 3 requested the supplied six-shard
`bash tools/autolab_tests.sh` harness. All three returned
`NotFoundError: No such file or directory`, without command output or returned
shard artifacts. Consequently target replay and the existing core/SDK suites
did not run.

The source audit found matching convolution contracts, so there is no
demonstrated discrepancy to repair. The kill is on this unsupported mechanism;
the infrastructure failure is not evidence of target parity. No target recovery
is claimed: targets won 0. No practice score was requested, so overall and
component deltas are unmeasured. The card's all-five-interval numerical kill
test has not been executed. The shipped practice graph, target inputs and
recurrent state remain unverified; this result does not certify that bundle or
the auxiliary teacher's convolution runtime.

There were no engine edits to revert and no changed behavior requiring new unit
tests. No model bundle is declared, no tests were relaxed, and no scorer, lab
tooling, test data, manifests or evidence were modified. No fold was performed.

# Recurrent acoustic-state precision round trips

Status: killed on the inspected premise. No engine changes were made. Shipped-model
runtime validation was unavailable; the conclusion rests on the source and
checked-in export fixture described below.

## Prediction and mechanism

The unchanged card predicts listener −1 and overall −1 on lab-v1.5u, using
listener `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. The five registered
added-word misses are p07 at 144.12 s, p33 at 156.16 s, p48 at 39.98 s, p56 at
106.76 s, and p57 at 95.26 s.

The proposed defect is repeated reduction of the precision of *retained*
acoustic state as a streaming cache advances. The intervention would remove
only demonstrated conversions on the route that carries those existing values
forward. Quantized weights, ordinary activation computation, audio samples,
chunk geometry, state retention, decoding, tracking and warning rules would
remain as shipped. The task requires stopping if that premise is absent or a
built intervention wins fewer than three targets.

## Source audit

`modal/student_export.py` initializes acoustic states as float32 and declares
that dtype in the exported state interface. Its streaming wrapper returns the
encoder's new states and embed pad directly. `stream_onnx` feeds the returned
NumPy arrays into the next call without conversion.

`packages/sdk/src/listener.ts` likewise initializes float32 acoustic tensors
and assigns each returned `new_<state>` tensor directly to the next call's
state map. `packages/sdk/src/ort.ts` constructs ORT tensors from their existing
types and typed arrays and returns the runtime arrays without changing their
precision. The integer processed-length counter is separate. The listener's
four-decimal rounding applies to posterior evidence and token confidence;
those numbers are never fed back as acoustic state.

The context-free gap decoder creates this same listener for each crop. The
auxiliary insertion teacher takes features and an attention mask in a single
call and has no carried acoustic-state interface.

The int8 export recipe in `tools/quantize_onnx.py` dynamically quantizes MatMul
weights and their activation operands. This alone does not establish cache
erosion: a retained float32 cache may be quantized temporarily for a MatMul
consumer while the original value is copied unchanged to the next cache.
Removing that ordinary consumer-side quantization would change the experiment's
mechanism.

## Export fixture validation

A read-only protobuf inspection of the existing
`packages/sdk/test/fixtures/tiny/model.onnx` found 2,379 nodes and verified its
manifest SHA-256:
`e28354258202150bd4447470e3a8d9ff5a7a70dadf3113941794ed52bad4c628`.
This fixture uses the same streaming export and Zipformer state interface,
with smaller dimensions and random weights. Its 37 acoustic-state input and
output ports are FLOAT; the processed-length ports are INT64.

Every acoustic-state copy route consists of `Concat` followed by `Slice`:
24 attention caches, 12 convolution caches and one embedding pad. No retained
state copy route contains `Cast`, `QuantizeLinear`, `DequantizeLinear` or
`DynamicQuantizeLinear`. These operators concatenate old values with newly
computed ones and select the next cache; they do not change retained values'
precision. A convolution or embedding cache can be fully replaced when the
chunk exceeds its capacity, which also cannot cause repeated conversion of
retained values.

The int8 recipe targets MatMul, so it may quantize a cache when using it as an
activation operand. That consumer reads the unchanged float32 value again on
the next call; its quantized approximation is not written back to the retained
cache. Counting repeated quantized *reads* as repeated lossy *writes* would
misidentify the proposed cumulative-erosion mechanism.

## Remote validation limits and outcome

Three supervisor requests returned `NotFoundError: No such file or directory`
without shard results or downloaded output:

- Run 1: environment/model inventory.
- Run 2: shipped int8 graph inspection and twelve-call float32 bit-retention
  probes, with runtime graph optimization disabled and enabled.
- Run 3: the supplied six-shard core/SDK build and test harness.

The run-2 diagnostic was prepared but did not execute. It would have used the
same synthetic feature stream and seeded nontrivial states in both modes,
checking each retained suffix and original values surviving multiple calls.
The actual shipped int8 graph, runtime optimizer behavior and five target audio
streams therefore remain unverified. Existing tests were requested but cannot
be reported as passing. Local diagnostic scripts, the parsed fixture result
and failed-request records are retained under ignored `.autolab/`.

The inspected SDK/export path already retains acoustic state at its original
float32 precision, and its fixture confirms copy-only cache advancement. There
is no demonstrated repeated conversion to remove within the card's mechanism.
Do not widen this into float32 attention computation, an fp16 repair or another
model intervention. Stop on the premise rather than invent such a change.

No implementation was built, so targets won is recorded as zero; that is not
an acoustic replay result. No practice score was used, and no measured overall
or component delta is claimed. There is no engine diff to revert and no new
behavior needing unit tests. Only this report is committed. The ignored
`.autolab/outcome.json` records the killed outcome. The experiment card,
scorer, supplied manifests, evidence and test data are unchanged; no model
bundle is declared and no fold was performed.

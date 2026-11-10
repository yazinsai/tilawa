# Relative-position cache-origin audit

Status: killed — the pinned streaming attention already uses a common query/key
origin after cache truncation. No engine behavior was changed, and the experiment
card is unchanged.

## Prediction and proposed mechanism

The card predicts one recovered addition, no losses, listener −1 and overall −1
on lab-v1.5u. Its proposed cause is retained keys being renumbered after cache
truncation while current queries keep another origin. The intended repair would
carry acoustic positions explicitly and derive relative offsets from them, while
preserving weights, retained context, lookahead, convolution state, output clock,
decoding and warning rules.

The premise requires at least two of the six registered targets to contain an
incorrect query/key offset. The task also directs stopping when the mechanism is
wrong. This audit found that the pinned implementation's offset is already the
actual retained-frame separation. Explicit positions would reproduce its current
index matrix rather than repair it.

## What was checked

The starting commit is `19a94c98`. `AGENTS.md` was read first.

`packages/sdk/src/listener.ts` passes each returned `new_<state>` back to the
next inference call without slicing, renumbering or rebuilding attention caches.
Its rolling feature-buffer compaction changes the local feature-array cursor,
not those recurrent tensors. Neither that cursor nor the SDK output-frame counter
is supplied as a query position.

`modal/student_export.py:wrap` calls the encoder's streaming implementation. Its
`processed_lens` state constructs the startup padding mask; it is not an absolute
query coordinate supplied to positional attention. `student_train.py` pins
icefall to `3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1`.

The exact pinned dependency was downloaded into ignored diagnostic storage and
inspected. Its `RelPositionMultiheadAttentionWeights.streaming_forward`
concatenates cached keys before current keys and retains the last `L` keys for
the next call. Its `CompactRelPositionalEncoding.forward` supplies an ordered
relative table from `−(L + Q − 1)` through `Q − 1`, where `Q` is the current
query count. The traced gather selects table index `Q − 1 − i + j` for query
`i` and key `j`. Therefore the selected offset is:

```
−(L + Q − 1) + (Q − 1 − i + j) = j − L − i
```

If the current chunk starts at absolute layer-frame position `P`, its query is
at `P + i`. A retained key is at `P − L + j`, and a current key uses the same
formula with `j >= L`. Their actual difference is also `j − L − i`.
Suffix retention advances `P` and retains contiguous keys, so this equality holds
after every truncation. The non-traced `as_strided` branch selects the same table
indices. Each stage downsamples both query and key coordinates by the same factor;
the deployed chunk is divisible by all six stage factors.

A lightweight read-only diagnostic then inspected the committed tiny ONNX fixture,
without running acoustic inference. All six `GatherElements` index matrices
exactly matched that formula. The unprojected positional tables matched the
pinned compact-encoding formula with maximum absolute error below `4.4e-7`.
Stage factors were `1, 2, 4, 8, 4, 2`; query lengths were `16, 8, 4, 2, 4, 8`.

An independent simulation assigned absolute frame IDs to keys and applied only
the implementation's concatenate-and-retain update. For each registry tier's
128-frame left context, it checked every available query/key pair through 501
chunks, including startup padding, the first truncation and repeated truncation:
1,878,660 pairs per tier, zero offset discrepancies. At the registered 320 ms
chunk advance, this covers the time range containing all six named targets.
This is a geometry check, not a runtime observation of their audio.

Diagnostic provenance:

- Pinned `zipformer.py` SHA-256:
  `4d5365ed19dee07f5a51fcaaa26e18310f7fafbdf2a391e24a7cc01d1b2e8cfe`.
- Committed tiny ONNX SHA-256:
  `e28354258202150bd4447470e3a8d9ff5a7a70dadf3113941794ed52bad4c628`.
- Reproduction: `PYTHONPATH=.autolab/audit-python python3 .autolab/audit-relative-positions.py`.
  Source, diagnostic script, dependencies and result JSON remain ignored under
  `.autolab/`; none is an engine or scoring change.

## Validation limits and outcome

Supervisor run 1 requested pinned-source and shipped-graph inspection. Run 2
requested only `python -c "print(123)"` as a health check. Run 3 requested the
supplied six-shard `bash tools/autolab_tests.sh` harness. All returned
`NotFoundError: No such file or directory`, with no command output and no returned
shard artifacts. Consequently neither the shipped practice ONNX graph nor target
audio could be inspected, and the core/SDK suites did not run. The tiny fixture
and pinned-source checks do not establish runtime parity for the inaccessible
practice bundle; this limitation is explicit, rather than a claimed target replay.

The kill is on the absent origin-mismatch mechanism in the pinned streaming
implementation, not on those infrastructure failures or collateral losses. No
target recovery is claimed: targets won 0. No practice score was requested, so
there is no measured overall or component delta. No new model bundle is declared.
There were no engine edits to revert, no tests were relaxed, and no scorer,
lab tooling, test data, manifests or evidence were modified. No fold was performed.

# Anti-aliased strided acoustic recheck

Status: implementation prepared; blocked on the supplied Modal runner. Acoustic
premise, target wins and the complete shipped-model bundle are not validated.
The experiment card is unchanged. No practice score or fold has been requested.

## Prediction and mechanism

The card predicts listener −1 and overall −1, with substantial zero-gain risk.
The proposed cause is temporal aliasing of rapidly changing nonlinear acoustic
features inside the shipped student's strided layers.

The SDK graph builder creates an independent ONNX graph from the shipped weights.
Immediately before the embedding's temporal convolution stride and each
Zipformer SimpleDownsample split, it inserts a channelwise temporal FIR filter.
For stride `s`, the fixed kernel is the normalized binomial of degree
`2 * (s - 1)`, with radius `s - 1`. Construction depends exclusively on stride
geometry. Symmetric edge padding centers the filter on the original input
coordinate, compensating the causal FIR delay without moving or deleting output
frames. Existing learned downsample weights remain in place. Original graph
inputs, outputs and state tensors retain their geometry.

The builder copies the entire primary bundle, including existing auxiliary
assets, without changing its manifest. It adds `antialias.onnx` and a separate
`antialias.json` declaring the fixed recipe, parent hash, graph hash and stride
geometry. The ordinary Node loader verifies that declaration and opens a
separate backend. A missing or damaged declared graph fails loading; a model
without a declaration uses its original behavior.

Candidate discovery replays the primary evidence through the ordinary free
tracker and checker. Only posterior-supported single-word string-path
accusations participate. Parallel-phrase graph comparisons, other error classes
and independent ending/duration paths retain their original comparisons. No
label, reader, session or verse selects a runtime window.

When candidates exist, the side listener processes the complete original audio
with independent streaming state and the primary frame clock. This avoids crop
and input-phase changes confounding the anti-aliasing intervention. On each
original candidate's unchanged scoring and word-slot intervals, it uses the
ordinary CTC forward score and the original confirmation bar. A certificate
requires a complete written pronunciation under existing held-sound, pause,
waqf-edge and soft-substitution rules, no complete reading of the accused rival,
finite scores, and a side margin below the original confirmation bar. Missing or
surplus phones do not qualify. The producer also checks the side stream's
origin, frame duration and final frame count against primary evidence.

Certificates travel on a primary token through ordinary JSON evidence transport.
The checker validates their exact word, hypotheses and nested frame intervals
before applying the veto. Primary phones, confidences and posterior rows remain
intact. Audio-backed sessions use ordinary checker replay at `end()` to withdraw
matching string warnings, while preserving the primary tracker and existing
insertion publication path.

## Validation completed locally

The analytic ONNX tests verify unit DC gain, channel independence, zero impulse
delay and rejection of input-Nyquist oscillation at strides 2, 4 and 8. The SDK's
tiny Zipformer fixture exposes seven stride sites: `[2, 2, 4, 8, 4, 2, 2]`.
Its modified graph preserves every original initializer and input/output
declaration, produces finite outputs over three consecutive stateful calls, and
preserves all output/state shapes. A bundle test checks byte equality of every
source file, including an existing auxiliary-file stand-in, and the generated
graph's checksum. All four Python graph tests pass.

Core tests cover complete versus truncated pronunciations, ordinary JSON
certificate transport, unchanged primary evidence values, mismatched intervals,
retained rival support and preservation of parallel-phrase confirmation even
when every eligible string comparison is vetoed. SDK tests cover positive
reading requirements, missing evidence, incompatible inventory/intervals,
optional and damaged asset loading, backend release, private audio retention,
end-of-session withdrawal and retention of insertion/graph decisions.

Core and SDK TypeScript builds pass. The completed full SDK suite passed 34 tests
with one optional integration test skipped. A complete tiny-fixture bundle also
loaded through the real Node loader, verified its graph hash, and processed the
same WAV through primary and side ONNX backends: both returned 145 frames at
40 ms, with equal frame origin and finite side probabilities. This verifies
runtime loading and stream geometry, not acoustic accuracy. A first local core
run had timeout failures and loaded code while the callback-scope change was
still being made. The fresh full core run, with one worker and a 60-second local
default timeout, passed 760 tests and failed one existing startup test on its
explicit 15-second deadline; Vitest also reported two fixed worker RPC deadline
errors. The same startup file still exceeded that explicit deadline when run
alone. With a temporary 60-second deadline, all three startup assertions passed;
the slower case took 17.6 seconds. It supplies no posterior rows, so it never
reaches the new acoustic comparison. The original 15-second timeout was restored
before committing, and the existing test file has no diff. Thus the completed
core run plus that diagnostic cover all 761 assertions, while the ordinary local
suite still has the documented deadline limitation.
These local fixture checks do not establish the shipped graph's acoustic
behavior or its training-recording timing.

## Unresolved acoustic validation

Five requests through the user-supplied `autolab.py modal-run` route failed
before command execution with `NotFoundError: No such file or directory`.
Requests included shipped-graph inspection, the prescribed six-shard test
harness, a minimal working-directory/import probe, and construction of the
complete production bundle under
`/scratch/listener-antialias-strided-recheck/bundle`. Every returned result
reported zero compute expenditure. No alternative credentials, Modal route,
outside-worktree files or release-exam data were accessed.

Consequently the actual shipped graph has not been inspected or transformed,
training-only recordings have not been used to validate graph timing, and none
of the four registered windows has been evaluated:

| Registered target | Complete written reading and rival lost |
| --- | --- |
| p11, 26:106:1, 104.32 s | Not measured |
| p36, 55:48:1, 282.64 s | Not measured |
| p55, 26:174:8, 107.08 s | Not measured |
| realuser-v1-sob-049, 92:13:1, 36.44 s | Not measured |

This is an infrastructure block, not evidence of zero wins or a premise kill.
The user requested continuation to scoring only after at least three target
wins; that prerequisite remains unknown. All three practice-score attempts
remain available. The ignored outcome records `blocked` with no measured values;
no killed outcome is asserted.

## Limits and resumption

Centered filtering preserves filter coordinates analytically; it does not
guarantee that nonlinear acoustic decisions remain aligned or that real
consonants survive smoothing. Edge extension uses only the available tensor at
each chunk boundary. Both effects need the requested training-only recording
checks on the actual shipped graph. Tiny-fixture geometry is not a substitute
for inspecting the production export and confirming every temporal stride.

The current side path buffers session audio and performs another full streaming
decode at completion. It can withdraw a live warning after that warning has
already been emitted. Its storage, compute and end-of-session latency are
experimental limitations; immediate streaming warning suppression is not
established. Population loss rates, clean-session behavior, studio regressions
and aggregate component scores are unmeasured.

Restore the supplied runner, build and verify the complete primary-plus-side
bundle, verify timing on training-only recordings, then evaluate the four
registered windows using ordinary free tracker/checker discovery. If fewer than
three targets are won, revert the engine changes, record the measured premise
kill and write the prescribed killed outcome. If three or more are won, select
the verified bundle and request practice scoring before any narrowing. Do not
evaluate this implementation with the primary model alone: that would omit the
intervention. The ignored `.autolab/model.json` points to the intended `bundle`
directory so validation blocks on the absent production bundle rather than
scoring an inactive recheck. That selection is pending, not a claim that the
bundle exists or passed validation. No scorer, lab-tooling, supplied-manifest,
test-data or evidence edits are part of this branch.

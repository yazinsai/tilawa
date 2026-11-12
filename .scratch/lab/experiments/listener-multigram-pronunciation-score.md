# Multigram pronunciation scoring

Status: implementation prepared; acoustic fitting and target validation blocked by
the supervisor's Modal environment. No practice score has been requested. The
experiment card is unchanged. This is not a measured rejection of the hypothesis.

## Prediction and mechanism

The card predicts one false wrong-word warning removed, zero newly missed
mistakes, listener −1 and overall −1. The task's stopping rule requires at least
three of the four registered targets to improve before a practice score; fewer
than three requires reverting the engine changes and recording a killed outcome.

The implementation exposes frozen, deployment-exact final encoder features in a
separate ONNX graph. For the quantized model these are the dequantized activation
codes actually consumed by its final phone projection. Export checks unchanged
primary probabilities and every streaming cache on eight successive windows.

Two small, identically initialized nonlinear CTC heads consume those features.
The multigram inventory contains singleton phones and up to 48 frequent two- or
three-phone units mined exclusively from fitting speakers' authorized none-labelled
help-train transcripts, requiring support from at least two speakers. Complete
transcripts, not chosen unit segmentations or forced word timings, supervise the
heads. A forward automaton sums every valid segmentation and CTC acoustic placement:
blank states identify consumed phone prefixes; active states additionally identify
the most recent unit. Repeating an active unit consumes no new phones; emitting the
same unit again requires an intervening blank. Only complete pronunciation states
are accepted.

The single-phone control has identical layers, output dimensions, parameter count,
recordings and update budget. Its extra output slots are negative softmax classes
that cannot transcribe phones, rather than additional permissible acoustic units.
Both arms use the same normalization and training-speaker validation split; each
selects its minimum validation-loss checkpoint independently. Release-held-out
speakers, unusable labels, planted edits and repeats are excluded before feature
caching. Fitting is bounded to 120 epochs and 3,000 seconds on two CPU threads by
default, below one CPU day.

The SDK observes complete written-versus-already-accused comparisons only after
the existing checker supports a string-path wrong word. It uses the exact original
frame interval and requires finite multigram preference for the written sequence
while the matched phone control prefers the rival. Exact word, frame, pronunciation
and pronunciation-piece matching binds the resulting certificate. The checker
retains baseline internal confirmation ownership and changes only the public
verdict, preserving subsequent graph and other-class decisions. Primary decoding
and tracker inputs are unchanged. Audio sessions attach certificates and publish
matching verdict downgrades at completion; this experimental path buffers session
audio. Graph confirmations and insertion decisions remain outside its scope.

Declared model side assets must be complete, hashed, and tied to the unchanged
primary graph. The bundle exporter copies the original model directory, preserving
its manifest and existing auxiliary assets, then adds the extractor, fitted heads
and a separate side-asset descriptor. No trained bundle exists yet.

## Validation and limits

The actual TypeScript scorer passed 32 exhaustive small acoustic-path comparisons
locally. A temporary dependency-free smoke harness also executed all seven checked-in
SDK algorithm tests, covering alternate segmentations, repeats, full completion,
singleton CTC equivalence, preference reversals, malformed comparisons and training
eligibility. The Python automaton topology passed 24 exhaustive path comparisons
using a lightweight probability interpreter. Python compilation and `git diff
--check` passed. These local checks do not replace the requested full test suites.

Added tests also cover exact certificate binding, public-verdict-only changes,
graph preservation, missing/corrupt declared assets and audio-session evidence
transport. The differentiable training implementation has exhaustive-path,
ordinary-CTC equivalence and finite-gradient tests ready to run on Modal.

Supervisor runs 1 (data/dependency discovery), 2 (six-shard build and full core/SDK
tests), 3 and 4 (discovery retries) all returned `NotFoundError: No such file or
directory` before any command log or output was delivered. No training recordings,
features, fitted heads, ONNX exports or target comparisons have been obtained.
Consequently, target wins, missed mistakes, clean-session safety and overall change
are unknown. A working supervisor is required to finish; running the unchanged
primary model without the paired side assets would not evaluate this experiment.

## Remaining execution

After authorized data discovery, export with `modal/student_multigram_export.py`,
cache eligible takes with `packages/sdk/src/bench/multigram-cli.ts`, and fit with
`modal/student_multigram.py`. Verify the exported heads and segmentation sums
against the training-only parity fixture through the same CLI; package and verify
the complete bundle through `loadNodeModel`. Run the differentiable scorer tests
and the full six-shard core/SDK validation. Inspect all four registered targets on
their actual checker-requested intervals. Apply the task's target stopping rule;
request a practice score only if at least three targets improve. Declare
`.autolab/model.json` only after the persistent scratch bundle is complete and
verified. No killed outcome or target count can be justified from infrastructure
failure alone.

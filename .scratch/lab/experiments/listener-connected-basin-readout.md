# Connected nonlinear phone readouts

Status: needs-human. Implementation prepared; training and acoustic validation
are blocked by the supplied supervisor execution service. No candidate bundle
is declared and no premise kill is asserted. The experiment card is unchanged.

## Prediction and mechanism

The prediction is listener −1 and overall −1 wrong calls, with no forecast
losses. The proposed cause is a nonconvex CTC alignment basin: independently
trained nonlinear readouts can settle on brittle temporal assignments, while
an arithmetic parameter average can pass through a high-loss region. A learned
quadratic connection might contain an interior readout with a useful written
pronunciation preference. This premise has not been measured on the targets.

`modal/student_connected_basin.py` prepares immutable deployed-encoder features,
fits two independently initialized nonlinear residual readouts, and learns a
quadratic Bézier control point with both endpoints fixed. The readout adds a
64-unit tanh network to the unchanged shipped log probabilities and renormalizes
them. It is a separate scorer; primary decoding uses the shipped model.

The objective is complete-transcript CTC plus KL preservation of the shipped
output. Each connection update samples three stratified positions along the
curve. Endpoints receive no gradients and exact parameter equality is asserted
after fitting. The control starts at the arithmetic endpoint average.

Selection evaluates t = 0.25, 0.5 and 0.75 on reserved whole training speakers,
weighting speakers equally. It freezes the selected parameters and records
validation losses along the curve and the straight chord. Both endpoints,
their arithmetic parameter average, and an ordinary independently trained refit
remain explicit comparison arms. The ordinary refit receives the same total
number of per-recording objective evaluations as both endpoints and all
connection samples combined. This matches dominant CTC/readout work, rather
than claiming identical wall-clock time or optimizer overhead.

## Training population and compute

`modal/student_basin_features.py` requires authorized original clean-match
parquet recordings with speaker-cluster provenance under `/student/s1`.
It excludes an entire speaker cluster if any member is outside the training
split or appears on the existing release exclusion list. Unknown clusters are
not admitted. It deterministically selects at most 128 original recordings,
each at most 30 seconds. The original recording and augmentation mounts have
not been inventoried successfully in this environment.

Help-train supervision admits only training speakers' usable none-labelled
takes without mistakes, repeats or unsupported regions. Any speaker with an
excluded or nontraining take is excluded in full. Other training takes provide
only transcript-free preservation of the shipped output; their planted scripts
are never treated as performed transcripts. Speakers are separated before
feature capture, and duplicate waveforms with conflicting speaker provenance
are rejected. The supplied isolated help-train copy excludes release speakers
by construction; no practice recording or label enters fitting or selection.

Two deterministic views use the existing student augmentation functions and
banks. The ONNX capture graph exposes the actual final-head input, including
rounded/dequantized activations for an integer head. Eight consecutive windows
must retain every primary output and recurrent state exactly under the SDK's
enabled CPU graph optimizations. Final padded rows are trimmed to the deployed
output count. Normalization statistics use training speakers only.

The recipe uses eight endpoint epochs and eight connection epochs. The ordinary
control uses forty epochs to match objective evaluations. Preparation, fitting,
validation and export share a persisted process-CPU counter capped at 86,400
seconds. Computation uses two CPU threads. The budget and training recipe are
fixed before practice inspection; the recipe is not calibrated against scores.

## Engine behavior and assets

`packages/sdk/src/connected-basin.ts` discovers otherwise supported string-path
wrong-word comparisons through the unchanged checker, using the original
primary evidence before gap splicing or teacher enrichment. Requests retain
the complete written and existing rival strings, word identity, and exact
original frame bounds. A separate full-recording streaming state supplies
features on the same frame grid, with the same warm context and final trim.

Both hypotheses use the same complete CTC forward algorithm and the same
run normalization as the shipped CTC training transcript. Distinct phones and
the complete ending remain required; no partial prefix or skipped phone is
accepted. Hypotheses differing only in repeated-phone length tie, so that
difference alone cannot obtain a veto. All five readouts score identical rows.
A finite strict connected-readout preference for the written pronunciation
issues a certificate. The separate `basinUniqueGain` diagnostic requires this
preference to be absent from both endpoints and their arithmetic average, as
specified by the card's premise criterion.

Core binds certificates to both complete strings, the word and exact window.
Graph confirmations are exempt. Internal confirmation and ownership ledgers
remain unchanged; only the public wrong-word verdict can become uncertain.
The audio session publishes a downgrade only for the exact live occurrence
removed by identical-primary-evidence replay. Primary tokens, posteriors,
tracker events, gap decoding and insertion decisions keep their existing paths.

Export copies the primary model and supplied manifest verbatim and requires
every inherited insertion-teacher asset to exist with its declared checksum.
Additional files are `encoder-features.onnx`, `readouts.json` and
`connected-basin-asset.json`. The Node loader checks the primary checksum,
side-file sizes and hashes, readout geometry and inventory. An installed
descriptor with missing or corrupt files fails loading instead of silently
falling back to main. There is no `.autolab/model.json` because no complete
trained and verified bundle exists.

The auxiliary audio route is buffered until session completion. It does not
establish timely streaming warning suppression, mobile/web asset loading,
bounded session memory or device latency. Scripted-take KL preservation does
not prove that genuine substitutions survive the eventual veto.

## Validation and execution failure

Python source compilation, TypeScript Node syntax checks and `git diff --check`
pass. Three pure Python authorization/partition tests passed locally.

Eleven synthetic core/SDK cases executed successfully against temporary source
copies using Node's TypeScript transformation and a small `node:assert` adapter.
These exercised the checked-in test bodies for exact-window binding, malformed
scores, unchanged issue creation, graph exemption, exhaustive complete CTC path
sums, repeated-label blank separation, nonlinear/control preference, incomplete
window abstention, unchanged primary evidence, occurrence-specific retractions,
and native frame/cache/padding behavior. A length-only comparison also ties.
This is narrower than a full Vitest run and is not a TypeScript typecheck.

Additional checked-in fixtures cover endpoint gradient isolation and nonlinear
functional interpolation in PyTorch, verified/missing/corrupt asset loading,
and session-end downgrades with position/insertion preservation. Those fixtures
have not executed: local PyTorch and ONNX runtimes are unavailable.

Supervisor requests 1 and 2 asked for read-only data/runtime inventories.
Request 3 used the explicit node image and prescribed six-shard
`bash tools/autolab_tests.sh`. Request 4 used the explicit training image for
the Python tests followed by prepare, fit and export. All four returned
`NotFoundError: No such file or directory`, recorded zero compute expenditure,
and produced no shard output. No acoustic job or required full test suite ran.
The user was notified through the asynchronous input tool that the execution
service needs restoration.

## Remaining experiment work

After restoring the supplied supervisor, run the core/SDK harness and Python
tests. With `PYTHONPATH=/app/tools:/app/modal`, run the trainer's `prepare`,
`fit`, and `export` commands using
`--out /scratch/listener-connected-basin-readout/refit-v1`; prepare/export also
need `--model /lab/data/models/spliced-clean-int8`. Verify the complete bundle
through `loadNodeModel` and numerical readout parity before declaring it.

Freeze training selection before evaluating the registered targets. Use
`addBasinEvidence`'s inspection callback to record all five scores on actual
engine-requested windows, and count warnings removed through engine replay.
Do not substitute convenient crops or shortened pronunciations. If no target
has a connected-only preference, or fewer than three targets are actually won,
revert the engine changes and record a measured premise kill. At least three
target wins require the official practice score before assessing losses.

Target wins, overall/listener changes, studio preservation and fully clean
session safety are unmeasured. Practice scores used: zero of three. No scorer,
lab tooling, supplied manifest, evidence, test data or experiment card was
changed. Infrastructure failure is not evidence that the acoustic premise is
false; no killed outcome or fold has been recorded.

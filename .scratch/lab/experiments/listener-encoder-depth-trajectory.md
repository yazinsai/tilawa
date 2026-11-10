# Encoder depth trajectory

Status: needs-human. The supplied supervisor cannot launch a Modal run. The
implementation is present, but training, asset verification and target validation
are blocked. No acoustic result or practice score has been obtained. The
experiment card is unchanged.

## Prediction and mechanism

Predict two false warnings removed and one caught substitution lost: listener
−1, overall −1. The hypothesis is that late encoder processing sometimes
replaces useful earlier acoustic distinctions with an incorrect interpretation.
The comparison requires positive evidence for the complete written pronunciation
from an encoder-depth scorer, and an opposing preference from a matched final
layer control; mere disagreement or an uncertain rival cannot veto.

`modal/student_depth_export.py` exposes three architecturally selected completed
Zipformer encoder stages and the final pre-CTC representation in a separate copy
of the supplied frozen ONNX graph. It verifies eight streaming windows of primary
output and recurrent-state parity before writing extractor metadata. The original
model and manifest are never modified. Stage features are mapped to the native
40 ms clock using the architecture's integer subsampling ratio. A fixed shared
four-coordinate projection represents each depth; the observation concatenates
the first projected representation with its three successive depth differences.
The control projects only the final representation to sixteen coordinates.
These projections are fixed before training and target inspection.

Both arms fit sixteen-dimensional, phone-conditioned full Gaussian distributions
with identical model capacity, regularization choices and CTC topology. EM sums
phone-boundary paths instead of imposing one forced segmentation. Legal timing
arcs have normalized probabilities; sequence scores condition the timing prior
on completing the entire pronunciation to avoid rewarding a different number
of paths. Parent posteriors initialize the training boundary distribution only.
Speaker-grouped, training-only three-fold validation separately selects each
arm's shrinkage from the fixed grid, averaging validation likelihood by speaker.
Neither targets nor practice wrong-call counts select any training choice.

The preparation command accepts help-train `none` takes on the `train` split
only, excludes every speaker with any non-training take, and requires an explicit
release-speaker exclusion file. The trainer independently rejects ineligible
records. Mistake scripts are not treated as verified performed transcripts.

The SDK discovers comparisons only when the unchanged checker would confirm a
non-graph wrong word. It uses the exact complete written and accused
pronunciations and frame bounds already used for acoustic confirmation, including
identical neighbouring text. It carries a certificate only for a finite strict
preference reversal. Core matches word identity, both complete strings and exact
window coordinates before changing the public verdict to uncertain. The internal
issue ledger stays unchanged to preserve subsequent insertion/merge decisions.
Primary phones, posteriors and tracker input are preserved.

The Node loader checks the descriptor, both side-file hashes and the primary
model identity. A declared incomplete bundle fails loading. Undeclared ordinary
models use the existing path. The audio-backed SDK computes this auxiliary
recheck at end of session, retains certificates for stored-evidence replay, and
publishes explicit downgrades for certified existing live warnings. This buffered
end-of-session route is an experiment limitation: it is not an immediate warning
veto and has not demonstrated device latency or memory suitability.

## Validation and limits

The four Python numerical tests pass locally, using the installed x86 Python
with its matching NumPy architecture. They check exhaustive short CTC paths,
normalized posterior occupancy, path-count cancellation, and positive-definite
regularization. The core and SDK TypeScript builds pass. Tests additionally cover
exact certificate transport, unchanged baseline candidate discovery, graph
exemption, candidate exchange symmetry, abstention on missing/nonfinite data,
and rejection of incomplete/corrupt declared assets. The session integration test
uses a synthetic substitution and a mocked auxiliary score, exercising the real
tracker/checker and session completion route: it observes an explicit downgrade,
retained stored certificate, identical primary position events and unchanged
insertion verdicts. It does not validate the trained acoustic scorer.

Final local SDK run: 36 tests passed, one optional help-data test skipped, using
one worker and a 60-second default timeout. Both TypeScript builds pass. The
core full run with two workers passed 759 tests, but two existing startup tests
exceeded their explicit 15-second limits, with two worker-update timeout errors.
An isolated final run passed all five depth tests and two of three startup tests;
the remaining startup test still timed out. Replacing only the checker with its
original HEAD version for an isolated comparison produced the same startup
timeout (two passes, one timeout); the experiment implementation was restored.
No core assertion failure was observed, but the complete core suite is not
certified passing here. The prescribed remote harness never executed.

Supervisor requests 1 and 2 attempted read-only input inspection, request 3
attempted the prescribed six-shard `tools/autolab_tests.sh` harness, and request 4
retried the infrastructure with `pwd`. All four
returned `NotFoundError: No such file or directory` before executing commands;
all reported zero compute cost and produced no remote command output. Consequently
no supplied training label, practice audio, encoder graph or target window has
been inspected on Modal. This is an infrastructure blocker, not evidence against
the acoustic premise. Target comparisons attempted: zero. Practice scores: zero
of the permitted three. No fold or discard was performed.

There is no `.autolab/model.json` declaration and no purported trained model
bundle. Running the main listener would not evaluate this implementation. Before
validation can proceed, the supplied supervisor must be restored; then inspect
eligible labels and the supplied model, export and verify the extractor, prepare
eligible observations, fit/freeze both arms, and validate SDK/Python score parity
and the complete bundle. Only then inspect the registered engine-requested
windows. Apply the task's explicit continuation requirement: fewer than three
recovered targets stops the experiment, despite the card's two-target criterion.
If at least three recover, run practice-score before making a gate decision.

The generic preparation and comparison command is
`packages/sdk/src/bench/depth-trajectory-cli.ts`; training is
`modal/student_depth_trajectory.py`. Both have explicit path arguments and no
incident list or target-based inference path. A completed export must include
`depth-trajectory.onnx`, `depth-trajectory-emissions.json` and
`depth-trajectory-asset.json` alongside the unmodified primary bundle, including
its existing auxiliary assets. No scorer, lab tooling, supplied manifest,
evidence, test data or experiment-card file was edited.

The preparation utility currently expects normalized label fields `speaker`,
`split`, `kind`, `wav` and `ayahs`. Its compatibility with the supplied help-train
labels has not been established because the data inspection never ran. Likewise,
the exporter has not established that the supplied frozen graph exposes the
assumed completed-stage names, shapes and clock ratios. These are unresolved
implementation checks, not successful acoustic measurements.

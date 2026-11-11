# Multiscale waveform regularity evidence

Status: needs-human — implementation draft; Modal execution is unavailable.
The training fit, registered target comparisons, complete model bundle and
required build/Vitest validation remain incomplete. This branch is not ready
for scoring. The experiment card is unchanged.

## Prediction and premise

Predict two false wrong-word warnings removed and one caught substitution lost:
listener −1, overall −1. Spectral averaging may confuse a sharp interruption
with sustained periodic motion or irregular turbulence, while local waveform
scaling may retain the distinction without estimating pitch.

The card requires at least two targets to prefer the complete written
pronunciation under regularity scoring while the energy control prefers the
complete accusation. The task additionally requires at least three actual
target warnings removed before continuing to practice scoring. Neither has
been measured. An infrastructure failure establishes neither zero target
wins nor a scientific premise kill.

## Implementation

`packages/sdk/src/regularity.ts` observes the original waveform in 25 ms windows
at the native 10 ms frontend clock. Window DC removal and division by window
RMS normalize gain. A centered Haar wavelet bank uses symmetric window-edge
extension and prefix sums; no pitch estimate, reconstructed speech, student
logit or encoder feature is an observation.

For every adjacent scale pair, the representation measures pointwise
log-magnitude change divided by log scale change. Its mean and standard
deviation over the analysis window give twelve regularity coordinates.
Magnitude floors are relative to the same RMS normalization. The control
pools squared coefficients across all scales first, then records log energy
in twelve temporal subdivisions of the same analysis window. It retains local
energy dynamics while discarding coefficient scaling relationships. Both arms
have twelve coordinates and identical phone-conditioned Gaussian parameter
counts; the control is independently fitted on the identical recordings.

Streaming observation extraction is independent of microphone push sizes.
An existing 40 ms evidence interval indexes four native observations per frame.
The SDK compares the complete written and already-accused pronunciations on
exactly those existing endpoints. Non-finite waveform observations remain
unavailable evidence, rather than becoming silence. Missing phones, incomplete
windows and windows exceeding 2,048 observations abstain.

Each complete sequence has a mandatory state for every phone. Self loops sum
all monotonic phone boundaries. An emission-free forward partition conditions
the common geometric timing prior on consuming the complete pronunciation in
the supplied interval. Neither arm makes a phone optional or accepts a prefix;
repeated labels remain separate mandatory states. The same duration law is
used for both candidates and both feature arms.

`modal/student_regularity_data.ts` prepares SDK features from the supplied
help-train labels. It requires an explicit speaker exclusion list, additionally
excludes every speaker with any non-training recording, and accepts only
none-labelled training takes without events or repeats. Their canonical ayah
pronunciations come from the generic repository reference. Recording scripts
are never converted into performed transcripts. Audio paths must remain under
help-train; non-native sample rates use the existing SDK resampler.

`modal/student_regularity.py` fits both emission models with forward/backward
EM, marginalizing phone boundaries throughout fitting. Uniform placement is
only initialization. Sparse phone statistics shrink toward manner/place
groups. Training inputs must explicitly carry none/train/non-excluded
provenance; the separate exclusion list is checked again. Three hash-ordered,
speaker-disjoint training folds select between two wavelet geometries
(2–128 or 4–256 sample half widths) and shrinkage values 16 or 64. Selection
maximizes equal-weight speaker-averaged predictive likelihood across the two
arms. Eight EM iterations, the standardized variance floor and feature
definitions are fixed before any target inspection. The final fit uses all
eligible training speakers and records input hash, exclusions, folds and every
selection result. Actual supplied metadata has not yet been audited because
no container executed.

The checker route is reused from the existing intraphone experiment. It exposes
an otherwise-supported string-path comparison without changing its candidates
or acoustic bars. A veto requires a strict regularity preference for the
written pronunciation and a strict control preference for the accusation;
ties abstain. The public final wrong-word event becomes uncertain while the
internal confirmation and ownership ledger remain unchanged, preserving later
candidate generation, insertion handling and merges. Graph confirmations are
exempt. Primary phonemes, posterior rows and tracker events are not replaced.

Live audio sessions use accumulated waveform observations before publishing
the final verdict. Recorded evidence carries exact-word, exact-pronunciation,
exact-endpoint score certificates through ordinary token metadata. A generic
replay discovers those original comparisons after existing gap and teacher
insertion enrichment; it does not generate an alternative decode. Evidence-only
clients can consume the serialized certificates without waveform access.

The optional Node loader checks `regularity-asset.json`, its fixed
`regularity-emissions.json` filename, byte count, SHA-256, inventory, parameter
shapes and clocks. A declared missing or corrupt asset fails loading and
releases the primary backend. Primary-only bundles keep their existing path.
The trainer exports this descriptor, both fitted models and a training report.
No supplied model manifest or registry was changed.

## Validation and limits

All three authorized supervisor requests failed before container execution:

1. Run 1 requested a read-only help-train/data layout and numerical-library
   inspection.
2. Run 2 retried that inspection.
3. Run 3 requested the supplied six-shard `bash tools/autolab_tests.sh` harness.

Each `.autolab/modal-result-<n>.json` reports
`NotFoundError: No such file or directory`, zero compute cost, and no container
log. This is not a failure of a model fit or a tested engine behavior. No
alternative Modal access or outside-worktree data was attempted.

Thirteen focused SDK/core cases passed locally using Node's native TypeScript
transformation on ignored workspace-local source copies, adapting import
suffixes and using a small assertion adapter for the existing tests. These
cover gain/polarity/DC invariance, finite silence, differences between periodic,
interrupted and turbulent waveforms, scale pooling in the control, arbitrary
push sizes, exhaustive complete boundary sums, strict preference reversal,
unavailable observations, repeated labels, model parameter validation,
exact-certificate transport, unchanged issue creation, and graph preservation.
These checks are not a TypeScript build or a full Vitest run. Added Node-loader
mock tests and the Python forward/backward/provenance tests remain unexecuted
on their intended runtimes. Python source compilation passed; the local NumPy
installation cannot load its binary extension due to an architecture mismatch.

No help-train audio was accessed, no distribution was fitted, no registered
target was tested, and no practice score was requested. All three practice
score opportunities remain. There is no validated model bundle and no
`.autolab/model.json` selector: a primary-only decode would not evaluate this
experiment. Do not interpret such a decode as zero gain. No killed outcome
was written because neither the premise nor target continuation criterion
could be evaluated. No fold or discard was performed.

After Modal execution is restored, audit the supplied training/exclusion
metadata, prepare observations, fit and freeze the paired asset, and verify
normal SDK loading. Then inspect all four registered targets using the actual
engine-requested windows and serialized certificate consumption, comparing
primary rows, tokens, tracker events and insertion decisions with baseline.
If fewer than three target warnings are removed, revert the engine draft and
record the prescribed killed outcome. Otherwise finish and verify the bundle,
run the required checks and request practice scoring before drawing a gate
conclusion. No parameter may be selected from those target comparisons.

Small-data emission error, unmodelled phone realizations and gain normalization
that loses useful intensity cues can all favor the wrong pronunciation. The
positive comparison and matched control constrain when a veto applies but do
not guarantee preservation of genuine substitutions. Observation accumulation
also needs bounded-history engineering for a production streaming client;
this draft buffers session observations alongside the existing audio capture.

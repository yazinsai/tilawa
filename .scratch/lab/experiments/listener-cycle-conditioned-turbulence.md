# Cycle-conditioned turbulence

Status: implementation drafted; training and target validation blocked by supervisor infrastructure. The experiment card is unchanged.

## Prediction and premise

Predict listener −1 and overall −1, with no predicted losses. The registered
targets are the phone wrong-word false warnings at p11 26:106:1, p36 55:48:1,
and realuser sob-049 92:13:1. The hypothesis concerns information discarded
when aperiodic high-band energy is averaged over voicing cycles: its distribution
within each normalized cycle may still distinguish the written pronunciation
from the already accused pronunciation.

No target has been inspected or scored in this implementation. Unknown gains
are not zero gains, and infrastructure failure is not a premise kill. The task's
three-target continuation requirement still applies: after a training-frozen
fit, fewer than three actual target vetoes requires rollback and a killed
outcome; three wins requires practice scoring before narrowing on losses.

## Implemented mechanism

`packages/sdk/src/cycle-turbulence.ts` reads the original 16 kHz waveform.
Two fixed high-band filters cover approximately 2.5–7 kHz. A low-band signal
provides a normalized-autocorrelation pitch estimate in a fixed 60–400 Hz range.
Negative-derivative minima, localized between samples, provide closure proxies.
At least three complete cycles, correlation of at least 0.6, and intervals within
15% of the estimated period are required. These geometric and numerical choices
are fixed in source before any practice inspection.

Each 50 ms window advances on the native 10 ms front-end grid. Complete cycles
are resampled to a common phase grid; the phase-locked mean waveform is removed
separately in each high band. The remaining squared waveform is the aperiodic
energy observation. Its pooled energy, normalized by waveform power, supplies
two control features. Eight normalized phase-bin energies per band supply seven
independent Helmert log-ratio contrasts per band. Pure deterministic harmonics
should disappear into the mean cycle. Subsample closure localization was added
because the synthetic harmonic control exposed integer-grid jitter producing a
spurious residual.

When cycle estimates or residual energy are unreliable, phase observations are
unobserved and integrate out exactly. Both arms then use the same ordinary
high-band pooled observations. A comparison abstains if the whole window has no
reliable cycle, a complete pronunciation is unsupported by training data, or
the available window cannot contain every phone. No pitch is manufactured in
unvoiced intervals.

`modal/student_cycle_prepare.ts` extracts the SDK's exact observations from
explicitly none-labelled, train-split, non-excluded help-train recordings only.
It excludes every speaker appearing in a non-training split and excludes
recordings carrying mistake, repeat or unsupported annotations. It records
audio and label hashes and the excluded-speaker list. Scripted mistakes are
never turned into performed transcripts.

`modal/student_cycle.py` fits regularized phone-conditioned Gaussian emission
distributions by forward-backward EM, marginalizing phone boundaries in a
complete-phone chain. Every phone emits at least once, including repeated labels.
Phase statistics use only reliable observations. Sparse estimates shrink to
global feature distributions. Three deterministic speaker-grouped folds choose
between regularization strengths 16, 64 and 256 using speaker-balanced recording
likelihood; the selected fit is then frozen on all eligible training speakers.
The pooled control uses the identical pooled parameters and normalized timing
prior, with only phase identity evidence removed. Both arms subtract the same
standardized phase background, so missing phase contributes zero log evidence.

At an existing confirmed string-path wrong word, the checker exposes its exact
complete written and rival sequences and frame endpoints. The SDK compares both
on the identical original audio observations. Only a strict reversal qualifies:
the pooled control prefers the rival and cycle conditioning prefers the written
pronunciation. Both scores must be finite. The live audio session can apply this
comparison when the requested observations are available; completed-session
evidence also carries exact-window certificates through ordinary token JSON.
Stored-evidence replay requires a certificate matching the word, both complete
pronunciations and both endpoints exactly. Discovery covers both direct core
consumers on the native complete-token grid and SDK consumers on their existing
evidence grid and acquisition cadence. The SDK grid implementation is shared
without changing its behavior. Each comparison retains its own requested window;
certificates are never transferred to a nearby or resized window.

Only the public wrong-word verdict is downgraded to uncertain. The internal
confirmation and ownership ledger retain their baseline state, preserving
subsequent insertion decisions. Primary phonemes, confidences, frame bounds,
posterior rows, tracker events and parallel-graph confirmations are untouched.
No session, reader or verse IDs participate in engine or training code.

The normal Node directory and tier loaders support an optional
`cycle-turbulence-asset.json` descriptor, containing a fixed emission filename,
byte count and SHA-256. They verify the model inventory, clock, dimensions,
positive variances and support counts. Missing/corrupt declared files fail and
release the backend. Ordinary bundles have no side declaration and keep their
existing behavior. Supplied manifests and registry data are unchanged.

## Validation and execution limits

Supervisor run 1 requested a read-only storage and help-train inventory. Run 2
requested the supplied six-shard test harness. Both returned `NotFoundError`
without container output. A third request running only `print(123)` failed in
the same way. These requests provided no data, audio or actual engine window.
Run 4 retried the complete supplied six-shard test harness after implementation
and also returned `NotFoundError`. Run 5 retried only `print(123)` after resuming
the task; it failed identically. All five supervisor requests failed before
container execution, at zero reported compute cost.

The TypeScript core and SDK builds pass locally, as does standalone type-checking
of the training preparation utility. Six focused core tests pass, covering
read-only candidate discovery, strict finite reversal, exact window and
pronunciation matching, unchanged internal issue creation, additions and graph
preservation. The SDK's seven acoustic tests pass, including a matched-energy
phase contrast, a genuine-rival control, exact missing-phase marginalization,
exhaustive complete-boundary summation, repeated-phone requirements, harmonic
removal and arbitrary audio chunking. Five asset-loader tests cover valid,
absent, missing, corrupt and inaccessible side declarations. Existing teacher
loader tests preserve their original behavior.

The initial complete local SDK suite passes: 37 tests, with one existing optional
integration test skipped. The added session transport test passes on a small
reference and an unrelated synthetic lexical substitution: discovery supplies
the SDK comparison, JSON and SDK rechunking retain the certificate, primary
evidence and tracker events match, and only the public confirmation changes.
Its first fixture exercised the protected parallel graph, which correctly kept
the warning. This synthetic test is not evidence of a registered target win.
The final full SDK suite after sharing the evidence grid passes all 38 tests,
with the same optional integration test skipped. Three small Python numerical
tests pass in an ignored NumPy environment: forward/backward values and occupancies match exhaustive
paths, missing phase matches the control, and synthetic EM produces finite
positive variances while enforcing training provenance. No recording was
trained on the host. Four cross-language numerical comparisons agree within
1.78e-15 absolute log-score error. Python compilation and `git diff --check` pass.

The first host core run hit the existing 5 s deadline on four tests in
`checker.test.ts`. A new full run uses a 30 s test deadline and one worker,
without changing test files or configuration. The full run passed 760 tests
across 53 files; two tests in the unchanged
`startup-persistence.test.ts` exceeded their explicit 15 s deadlines, which
override the command-line timeout. There were also two worker `onTaskUpdate`
timeouts during synchronous replay. Those are timing failures, not assertion
mismatches; the full run is not a pass. An isolated rerun passed two of that
file's three tests; one still exceeded its 15 s deadline (19.47 s). No assertion
mismatch was reported. These host timing failures remain unresolved; the
required Modal test execution is unavailable.

Zero practice scores have been requested; all three allowances remain. Listener
and overall changes, target wins, loss rates, clean-session behavior and studio
behavior remain unknown. No help-train model has been fitted, and there is no
complete deployable bundle. `.autolab/model.json` is not declared: decoding with
the main model alone would not evaluate this new scorer. The ignored outcome
records the infrastructure block rather than a kill or an invented zero gain.

## Resume and risks

Restore supervisor execution first. Audit the actual help-train speaker and
exclusion metadata, then run `modal/student_cycle_prepare.ts` with
`--help-train`, `--reference` and `--out`. Fit `modal/student_cycle.py` with that
input, its emitted excluded-speaker list and the reference inventory. Retain the
training report and freeze the chosen fit before requesting target diagnostics.
Inspect every actual confirming window obtained through `sequenceEvidence`,
including growing-window variants, using the complete sequences the engine
requests rather than hand-cut target tokens. Check certificate transport and
compare primary evidence and tracker events with and without enrichment.

Before declaring a bundle, copy the exact baseline primary ONNX and manifest
and preserve all installed baseline teacher assets. Add and verify the trained
cycle emission file and descriptor under this experiment's persistent scratch
directory. Declare only that complete verified directory in untracked
`.autolab/model.json`. Run the required Modal suites and the target test; apply
the user-directed kill or practice-score rule only after those measurements.

The detected closures are acoustic proxies, not independently verified glottal
closure instants. Filter delay, source-amplitude changes, period errors and
interpolation can contaminate the estimated aperiodic component. Low-pitch
speech may lack three cycles in 50 ms; breathy or irregular speech may abstain.
Independent Gaussian contrasts omit phase-bin dependence, and overlapping
windows retain temporal correlation in both arms. Limited clean learner data,
uniform boundary initialization and utterance-edge silence may produce poor
phone assignments. Predictive recording likelihood may select regularization
that does not distinguish lexical rivals. The session retains observations and
audio; real-time CPU and memory budgets remain unmeasured. Web and native asset
installation are not implemented. Positive cycle evidence for the written word
can still veto a genuine substitution, especially if the distinguishing phones
have unreliable cycles while shared vowels provide the phase evidence.

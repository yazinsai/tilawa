# Anticipatory gesture acoustic score

Status: needs-human — implementation is committed, but the supplied Modal
execution environment failed before every requested command. Training, target
validation, the complete exported bundle, and the required core/SDK checks remain
pending.
The experiment card is unchanged. This is not a premise kill or a measured gain.

## Prediction

The card predicts two false substitutions removed and one caught substitution
lost: listener −1, overall −1. Its four registered targets are the phone warnings
at 26:106:1, 55:48:1, 26:174:8 and 92:13:1. The task's continuation rule overrides
the card's two-reversal criterion: fewer than three recovered targets requires
rollback; three or more requires a supervisor practice score before narrowing.
No target measurement or practice score has run (all three scores remain).

## Mechanism and implementation

`modal/student_gesture.py` fits two independent, equal-capacity acoustic models
on the same normalized spectral observations. Every phone has ordered attack,
sustain and release spectra with regularized diagonal residual covariance.
Observed power is a normalized mixture of the current phase's spectrum, the
preceding phone's release and the following phone's attack. Two positive weights
are fitted jointly with emissions; fixed phase kernels fade the preceding
gesture and increase anticipation of the following gesture. The weights are
shared across all phones and all candidate pronunciations.

The non-overlap control has exactly the same parameter dimensions, kernel
weights, observations, phase topology and regularization. Its auxiliary kernel
contributions come from the current phone's own attack/release spectra rather
than adjacent phones. It is independently fitted, rather than disabling overlap
in the treatment's fitted weights. Both models use identically structured
padding at sequence edges.

Exact forward/backward sums marginalize every phone and phase boundary. Every
phase and phone is mandatory, including repeated equal phones. A fixed common
self-loop probability makes the conditional timing prior uniform over complete
segmentations; subtracting the timing partition normalizes that prior separately
for each complete pronunciation. No duration prior is fitted on practice.

The input utility accepts only help-train none-labelled training recordings,
excludes any speaker with a non-training recording as well as an explicit
exclusion list, and requires at least three eligible speakers. It reads the
unchanged SDK fbank's 10 ms frames, sums groups of four mel-band powers, and
normalizes spectral mass. Three deterministic speaker folds choose one shared
regularization strength from 8, 32 and 128 by the mean held-speaker likelihood of
both arms. Both arms are then refitted using all eligible training speakers.
The utility records input hash, speaker partitions, selection results, parameter
count and optimizer history beside its output asset. It accepts no practice
incidents or script-corrected performed transcripts.

The SDK compares the complete written and already-accused strings on the exact
windows requested by the ordinary confirming checker. Evidence frame coordinates
remain on the native 40 ms clock; each selects the corresponding four 10 ms
spectral observations. A certificate requires the overlap model to strictly
prefer the written pronunciation and the matched control to strictly prefer the
rival. Impossible pronunciations, unsupported phone coverage, non-finite results
and missing observations abstain.

The certificate includes the complete strings, location, exact frame bounds and
four likelihoods, and survives ordinary token JSON serialization and rechunking.
It changes only the public verdict of an existing non-graph wrong-word
confirmation to uncertain; the internal confirmation/ownership ledger stays
intact to preserve insertion decisions and subsequent candidate discovery.
The audio session runs the same comparison before publishing a live confirmation,
including when evidence recording is disabled. At completion it also annotates
the existing enriched evidence for downstream replay. Primary decoding,
posterior rows, tracker events and graph confirmations are preserved.

The normal Node bundle loader accepts `gesture-asset.json`, validates the
declared `gesture-emissions.json` byte length and SHA-256 and checks inventory,
spectral normalization and clocks. Missing or corrupt declared files fail
loading. No supplied manifest or registry entry was modified.

## Validation and current blocker

Local numerical tests passed under the existing x86 Python interpreter:

- Exact acoustic and timing partitions agree with exhaustive path enumeration
  for both arms; posterior occupancies and transition counts normalize.
- The joint emission/kernel analytic gradient agrees with numerical derivatives.
- Complete repeated phones cannot be erased to fit too few observations.
- Mistake-labelled, held-out, excluded and practice-source rows are rejected.

The initial native Python attempt could not import the installed x86 NumPy
extension on the ARM interpreter; `arch -x86_64 python3 -m unittest discover -s
modal -p test_student_gesture.py` passed all four tests in 0.371 seconds. This was
a tiny synthetic numerical test, not training or acoustic inference on the Mac.
Python syntax parsing and `git diff --check` also passed.

Added TypeScript tests cover exhaustive complete-sequence sums, normalized
anticipation/release mixtures, abstention and asset validation, push-independent
observations, exact certificate matching, unchanged internal issue creation,
graph preservation, live veto publication with recording disabled, and checksum
verified bundle loading. These tests have not executed.

Supervisor Modal requests 1 and 2 were read-only environment/data checks;
request 3 used the required six-shard `bash tools/autolab_tests.sh` command. All
three returned `NotFoundError: No such file or directory` before container
execution, with no command log or model output. There was no data inspection,
training, target probe, full SDK/core test result or practice score. Restoring
the supplied supervisor execution environment is required to continue.

No model bundle exists and `.autolab/model.json` was not created. Evaluating the
main model without a fitted gesture asset would not evaluate this experiment.
Do not fold or report a score from this incomplete implementation. After the
runner is restored: extract the eligible none-labelled recordings, complete the
speaker-grouped fit, verify Python/TypeScript likelihood parity and normal bundle
loading, inspect the actual requested target windows and preservation checks,
then apply the three-target continuation rule and practice-score requirement.

## Limits

The proposed overlap kernels are a compact acoustic approximation, not measured
articulator motion. Three mandatory phases and only 20 spectral dimensions may
be too restrictive; sparse phone coverage can force abstention. The independent
control and strict reversal rule test the overlap mechanism but do not guarantee
retention of genuine substitutions. Session observation buffering and CPU score
latency are unmeasured. No listening judgments, benchmark labels, evidence,
scorers or lab tooling were changed, and no fold was performed.

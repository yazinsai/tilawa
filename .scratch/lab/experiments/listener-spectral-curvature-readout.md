# Spectral curvature pronunciation readout

Status: needs-human — draft implementation checked locally; supervisor container
execution failed before training or target inspection. The experiment card is
unchanged. There is no fitted real-data bundle and this branch is not ready for
population scoring.

Implementation commit: `5d6ef413`.

## Prediction and premise

Predict listener −1 and overall −1: one false wrong-word warning removed, with
no predicted losses. Second differences of a log spectral envelope annihilate
an additive constant and a locally linear frequency slope while retaining
localized peaks and notches. The proposed evidence is therefore frequency-local
shape rather than smooth recording coloration.

The card requires a target to reverse from rival preference in a matched
undifferenced control to written-pronunciation preference in the curvature arm,
on the actual engine-requested window. The task additionally requires at least
three target warnings removed before proceeding to practice scoring. Neither
condition has been measured. Infrastructure failure is not a zero-win result,
evidence that the mechanism is wrong, or a premise kill.

## Implementation

`packages/sdk/src/spectral-curvature.ts` observes the native frontend's power
spectrum without changing its mel output. Triangular smoothing samples the log
envelope on an equally spaced 125–7000 Hz grid. Four successive native 10 ms
analyses produce one 40 ms observation, anchored at original sample zero.
Resolution candidates are 16, 24 and 32 envelope samples.

The control uses the interior envelope samples; curvature uses their second
frequency differences. Both arms have resolution minus two dimensions, the
same number of phone-conditioned diagonal Gaussian parameters, identical blank
and phone state topology, and the same fixed geometric duration prior. An
ordered CTC-style forward sum marginalizes boundaries and optional blank
placements. Repeated labels require separation. Only complete pronunciations
accept; scores condition on observed duration through an identity-neutral timing
partition. Unknown or unseen phones, incomplete placements, missing observations,
nonfinite inputs and windows over 2048 observations abstain.

`modal/student_curvature_extract.ts` exports all three resolutions from
single-ayah, none-labelled help-train training recordings only. It requires an
explicit speaker exclusion list, unions non-training speakers from metadata,
rejects contradictory mistake/repeat labels and paths outside the help root,
and supplies the canonical pronunciation with the supplied final isolated word
form. The extractor's assumptions about the actual supplied label layout and
split metadata still require a container audit; no real recording was accessed.

`modal/student_curvature.py` fits emissions with 12 fixed soft-EM iterations.
Phone occupancy updates integrate all boundaries, starting from uniform phone
placement and edge silence. Gaussian estimates shrink toward pooled moments;
fixed variance and scale floors prevent degenerate estimates. Three hash-ordered
speaker-disjoint training folds select resolution and shrinkage jointly from
the frozen grid (16/24/32 and 8/32/128). Selection minimizes the speaker-balanced
mean contrast log loss of both arms against four deterministically sampled
single-phone rivals. Both arms use identical validation alternatives. Ties
favor lower resolution and stronger shrinkage. All choices were specified
before accessing training or target recordings. The selected configuration is
then refitted on eligible training speakers once. No configuration has yet been
selected from real data.

The trainer independently checks source, none/train/non-excluded provenance,
exclusions, complete-label duration and finite observations. It exports emissions,
a descriptor and a report recording input hash, speaker folds, exclusions,
selection losses and fit history. The optional Node bundle loader checks filename,
length, SHA-256, inventory, sample rate, frame clock and parameter shapes.
Declared missing or corrupt assets fail loading and release the backend.

The checker exposes the complete written and already-accused hypotheses at an
otherwise confirmed string-path wrong-word event. It reports the exact native
frame endpoints used by the current support calculation. Audio sessions compare
both hypotheses before publishing confirmation. Recorded evidence receives
certificates after existing gap-splice and teacher insertion enrichment.

A certificate requires finite, strict control preference for the rival and
strict curvature preference for the written sequence. Word, both complete
pronunciations and both endpoints must match exactly. Ties abstain. A veto changes
only the public wrong-word verdict to uncertain and suppresses its mistake-list
entry. The private confirmation/ownership ledger retains baseline state, so later
insertion merges and candidate generation retain their original decisions.
Graph confirmations and primary decoding/tracker input remain untouched. No
benchmark session, reader or verse selector occurs in engine/training code.

## Validation and execution block

Four requests used only the authorized supervisor route:

1. Run 1: read-only training-data layout and scratch-capacity probe.
2. Run 2: retry of the help-train layout/capacity probe.
3. Run 3: supplied six-shard build/core/SDK test harness.
4. Run 4: updated six-shard harness, Python numerical tests and help-train
   metadata audit.

Every result was `NotFoundError: No such file or directory`, with zero reported
compute cost and no container log. These results establish neither a code/test
failure nor absent training data. The user was asked to restore container
execution; no alternative Modal access was attempted.

Nineteen focused Vitest tests passed locally: six core certificate tests, five
SDK numerical/frontend tests, three new asset-loader tests, two session
integration tests and three existing insertion-loader tests. They cover:

- additive gain/linear tilt cancellation and retained notch structure;
- finite, audio-chunk-independent observations and unchanged primary fbank;
- exhaustive small-path agreement, completion and repeated-phone constraints;
- unsupported-phone/input abstention and strict matched-control reversal;
- complete candidate discovery, JSON transport and exact certificate matching;
- retained issue creation, graph confirmations and a neighboring addition;
- live audio use with recording disabled, original tracker events, unchanged
  recorded primary tokens/posteriors and certificate enrichment;
- verified, missing and corrupt asset loading with backend release.

Five Python tests passed, including independent complete-path enumeration,
normalized soft occupancies, repeated-label impossibility, provenance rejection
and finite equal-capacity soft-EM fits. A tiny synthetic fit was accepted by SDK
model validation; six Python/SDK complete-pronunciation scores matched exactly
(worst absolute difference zero). This toy asset remains ignored and is not a
trained Quran pronunciation model.

A strict local TypeScript check passed for core/SDK source, the extractor and
new tests. The temporary environment supplies declarations for unavailable ONNX
packages; loader tests mock the runtime. Python syntax compilation and
`git diff --check` passed. These checks do not establish actual ONNX execution,
full build/suite success or full-session population preservation. The required
remote suites remain unverified.

No target recording or target comparison was inspected. Target reversals and
warnings removed are unknown. No practice score was requested, leaving all
three allowances. No supplied manifest, scorer, lab tooling, test data, evidence
or card was edited. No fold occurred. There is no `.autolab/model.json`: declaring
the primary-only or toy bundle would not evaluate this experiment. The ignored
outcome records the infrastructure block without asserting a kill or numerical
gain.

## Resume and limits

Restore authorized supervisor execution, audit help-train metadata and exclusion
provenance, then export eligible observations and fit/freeze the two scorers.
Replay the card's selected primary evidence through the ordinary free tracker
and checker. Use generic confirming-candidate discovery to obtain actual
hypotheses/windows, with target labels selecting reports only. Measure both
scores and whether each target warning is actually removed.

If fewer than three targets are removed, revert the draft engine changes,
record the measured premise kill and commit the result. If at least three are
removed, preserve the original primary ONNX/manifest and any installed teacher
files in the persistent experiment bundle, add the verified curvature assets,
verify ordinary Node loading and evidence transport, and only then declare that
complete bundle in untracked `.autolab/model.json`. Request practice scoring
before narrowing on an aggregate gate failure.

Invariance is exact for additive gain and linear tilt applied to the sampled
log envelope. It is approximate for coloration before spectral smoothing,
windowed real audio, additive noise and time-varying handset processing. Curvature
can amplify narrowband noise or remove useful broad spectral cues. Diagonal
Gaussian emissions, sparse training phones, fixed geometric timing and EM local
optima can confuse real substitutions. Positive written preference is not proof
of mistake preservation. The draft duplicates frontend work, retains session
observations/audio and replays discovery; device CPU/memory costs are unmeasured.
Node asset loading is implemented, while web/native installation is not.

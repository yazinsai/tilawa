# Elastic frequency correspondence

Status: needs-human — committed implementation draft; all three authorized
supervisor requests failed before container execution. Training, target
validation and the production bundle remain incomplete. Do not score the main
listener as though it includes this change. The experiment card is unchanged.

## Prediction and continuation criterion

Predict listener −1 and overall −1: one false wrong-word warning removed, with
no predicted losses. The speculative cause is a local shift in spectral
resonances which fixed-frequency matching interprets as a different phone.
Joint temporal and frequency correspondence should explain that shift while
preserving genuine changes in pronunciation.

Registered targets are the string-path phone warnings on p11 at 26:106:1,
p36 at 55:48:1, p55 at 26:174:8 and sob-049 at 92:13:1. Their labels do not enter
training, model selection, candidate discovery or runtime scoring. No target
audio or evidence was accessed during this draft.

The card requires at least two rigid-to-elastic preference reversals. The task
also requires at least three actual target warnings removed to continue.
Neither condition has been measured. An unavailable execution environment is
not a premise kill, and zero tested targets is not a measured zero gain.

## Implementation

`packages/sdk/src/elastic-frequency.ts` computes the existing 80-bin log-mel
frontend with its original 25 ms Povey window, 10 ms hop, DC removal and
pre-emphasis. It averages adjacent coordinates into 24 spectral bands and
subtracts each frame's mean to remove overall gain. No separate per-band
standardization distorts the meaning of the frequency coordinate. Observations
retain their native 10 ms progression; existing 40 ms evidence windows index
four observations per frame, with both endpoints unchanged.

Each pronunciation phone has three mandatory ordered spectral knots, with
diagonal Gaussian emissions. A left-to-right chain integrates over all knot
durations and phone placements. Repeated labels require separate complete
articulations. Both complete written and already-accused pronunciations are
scored, including shared context present in the original checker comparison.
No target phone is optional, and no new lexical rival is selected.

The elastic arm augments that chain with a frequency-displacement state on
the fixed grid −2, −1, 0, 1, 2 spectral bands. A displacement acts through a
sinusoidal correspondence pinned at both spectral endpoints. Every permitted
map remains monotonic. Mean and variance templates use the same interpolation.
Displacement can change by at most one band per observation, with a normalized
Markov law penalizing changes and displacement from identity. That state
continues across every knot and phone boundary: a mismatching consonant does
not receive a fresh independently initialized warp. The grid describes one
smooth field over frequency and time, rather than arbitrary independent bin
movement.

Forward summation jointly integrates temporal placement and deformation over
the complete pronunciation. The emission-free completion partition is
subtracted to remove timing-path multiplicity. The normalized warp prior is
identical for both hypotheses and has no phone-specific initialization or exit
reward. The rigid control uses the same fitted spectral templates, variances,
timing topology and normalization, restricted to displacement zero. A veto
requires a strict rigid preference for the rival and a strict elastic preference
for the written pronunciation. Ties, unsupported phones, incomplete windows,
non-finite observations and windows above 2,048 observations abstain.

`modal/student_elastic.py` fits the spectral trajectories with forward/backward
EM on canonical complete pronunciations from none-labelled training-speaker
recordings. Initialization spreads observations uniformly across the transcript;
later updates integrate all phone/knot boundaries. Sparse emission statistics
share Arabic manner, place and voicing attributes. A scalar acoustic scale,
positive variance floor and shared knot-duration probabilities are fitted
without accessing warning windows.

The trainer rejects mistake-labelled, unmarked or excluded-speaker inputs.
`modal/student_elastic_features.ts` extracts the exact SDK observations from an
explicitly audited JSONL inventory, with an independent exclusion list and
audio hashes. It accepts only records marked none/train/non-excluded and their
canonical pronunciations. Neither utility mechanically corrects a scripted
mistake transcript. These guards do not replace auditing the supplied labels
and speaker split; that audit could not execute here.

Three hash-ordered speaker-disjoint training folds select template shrinkage
from 8, 32 and 128, and deformation regularization from 0.5, 2 and 8, using
speaker-balanced complete-recording predictive likelihood. Each fold fits its
own acoustic scale and templates. Ties favor stronger regularization. The final
fit uses all eligible training speakers and records the input hash, exclusions,
fold membership and every selection value. No selection used practice data.

The checker exposes the complete pronunciations and original endpoints only
at otherwise supported non-graph confirmations, covering ordinary, word-local
and paired string paths. It accepts serialized elastic certificates only when
the word, written and rival pronunciations and both endpoints match exactly.
Certificates must contain four finite scores with the stipulated reversal.
Only the public wrong-word verdict and returned mistake list change to
uncertain; the original internal confirmation and ownership ledger remain,
preserving insertion merges and subsequent candidate generation. Graph
confirmations are exempt. Primary tokens, confidences, posteriors and tracker
events are not edited.

Audio sessions can consult the scorer before publishing confirmation.
Recorded evidence is enriched after existing gap decoding and teacher insertion
enrichment, retaining certificates in ordinary token metadata for evidence-only
replay. The Node directory and tier loaders verify the optional
`elastic-asset.json` descriptor and `elastic-templates.json` checksum, bytes,
inventory, clocks and parameter shapes. Missing or corrupt declared assets
raise an error and release the primary backend; primary-only bundles remain
available. Supplied model manifests and registry data are untouched.

## Validation and execution block

Three requests were made exclusively through the authorized supervisor:

1. Run 1 requested a read-only data/help-train/model layout and scratch-space
   audit.
2. Run 2 requested the supplied six-shard build/core/SDK test harness.
3. Run 3 retried that harness with the assembled scorer, training utility and
   tests, also requesting Python tests and a help-train directory listing.

All returned `NotFoundError: No such file or directory`, zero compute cost and
no container logs. No request reached the data audit, training, builds or
target diagnostics. Later small source corrections were verified locally;
no Modal snapshot has been validated.

Eleven SDK/core cases passed locally using Node's TypeScript transformation
on ignored workspace copies and a minimal assertion adapter. These include
exhaustive joint-path enumeration across a phone boundary, normalized warp
mass, agreement with the rigid arm on frequency-invariant emissions, a toy
shifted-trajectory reversal and a real-rival toy control, mandatory repeated
articulations, abstention, monotonic pinned warps, frontend chunk invariance,
exact-window JSON certificate matching, unchanged issue creation and an
exempt parallel-graph confirmation. This adapter is not the complete Vitest
suite or a TypeScript build.

Five Python numerical tests passed in an ignored workspace-local NumPy
environment. They check exhaustive timing/warp paths, constant-emission
normalization, pinned monotonic maps, exclusion/provenance rejection and a tiny
finite EM fit, and vectorized-versus-enumerated likelihood agreement. No real
recordings were fitted on the Mac. Eight complete-sequence Python/SDK scores
on toy fitted parameters matched to a maximum absolute error of
1.1368683772161603e−13, across both rigid and elastic arms.

A separate loader check using actual temporary files and a mocked ONNX backend
passed optional, verified, missing and corrupt asset cases, primary-manifest
preservation and failure cleanup. Unit tests for those cases are also added
to the existing Node loader suite. Python compilation and `git diff --check`
passed. None of these checks establishes full-stream behavior, target recovery,
population losses or an independent listening judgment.

Zero practice scores were requested; all three allowances remain. Target wins,
losses, overall change, clean-session behavior and studio behavior are unknown.
There is no training-frozen help-train model, no production bundle under the
persistent Modal scratch directory, and no `.autolab/model.json` declaration.
Ignored local toy parameters are numerical fixtures, not deployable assets.
The ignored outcome records the infrastructure block without claiming a kill.

## Resume and limits

Restore the supervisor execution path and audit the supplied speaker and label
provenance. Construct the audited none-labelled training inventory, extract
native SDK observations and freeze the speaker-grouped fit before querying
practice windows. Verify the complete exported bundle with the original primary
ONNX, manifest and any installed teacher files intact. Only then declare its
relative persistent scratch path through untracked `.autolab/model.json`.

Run the required builds/tests and actual SDK target-window comparisons, checking
primary evidence, tracker events, graph confirmations and insertion decisions
against baseline. If fewer than three actual target warnings are removed or
the premise is disproved, revert the engine draft and record the measured kill.
If at least three targets are removed, request practice scoring before deciding
whether to narrow anything on an aggregate gate failure. Do not tune against
target preferences or count ordinary main-model scoring as evaluation of this
asset.

The one-dimensional deformation family may be too restrictive to explain real
local shifts or too flexible to retain weak substitutions. Latent alignment
from uniform initialization may converge poorly. Three mandatory knots and
expanded duration labels may misrepresent short phones, while utterance-edge
silence has no separate background state. Diagonal emissions and sparse
training coverage can cause acoustic errors or abstention. Tiny toy tests
cannot establish these protections on real learners. Session observations and
audio are currently retained for evidence enrichment; runtime, memory and
device budgets remain unmeasured. Node loading is implemented, but web/native
asset installation is not. No release-readiness or loss-free claim is made.

# Spectral ridge connectivity

Status: needs-human — implementation draft, blocked before Modal execution.
The experiment card is unchanged. No fitted scorer or complete deployment bundle
exists, so this branch is not ready for scoring.

## Prediction and stopping rule

Predict listener −1 and overall −1: one false wrong-word warning removed, with
no predicted losses. The hypothesis is that correct speech retains connected
spectral trajectories that distinguish it from a rival supported by misleading
local energies.

The card names zero preference reversals on the four actual engine-requested
windows as its kill criterion. The task requires at least three target warnings
removed to continue to practice scoring. Neither condition has been evaluated:
no target audio or training recordings could be accessed. An execution failure
is not evidence for killing the acoustic premise or a measured zero gain.

## Implementation

`packages/sdk/src/spectral-ridge.ts` extracts a separate acoustic representation
from the original waveform. The existing 25 ms Povey window, 10 ms hop, DC
removal, pre-emphasis and 512-point FFT are reused through an optional spectrum
observer on `Fbank`. Installing this observer leaves the primary mel calculation
and output unchanged. Four FFT bins are grouped per coordinate, excluding DC,
into 64 log-power coordinates; this is a compact spectral representation rather
than an LPC formant estimate or a second recognition network.

Within each column the log spectrum is centered and scaled to describe spectral
shape. At five fixed normalized energy levels (−2, −1, 0, 1, 2), contiguous
superlevel frequency intervals form ridge nodes. Nodes in adjacent columns are
connected when their intervals overlap, allowing one coordinate of frequency
motion. Both forward and backward edges are inspected, including splits,
merges and interruptions. Summing observations over all levels approximates
superlevel persistence; no single level decides whether a ridge is present.
This is a finite quadrature approximation, not an exact persistent-homology
barcode or a guaranteed tracker of physiological formants.

Eight spectral regions each retain four control coordinates: mean original
log power, ridge count, within-region ridge location and ridge prominence.
The connected arm additionally retains three coordinates per region: connected
time extent, branching, and interrupted edges. The resulting 56 observations
contain 32 control and 24 connectivity coordinates. The control integrates out
the connectivity coordinates of the same fitted diagonal Gaussian emissions,
retaining identical location and energy summaries, scaling, phone parameters,
complete-pronunciation topology and timing prior. It is an ablation of the
connection information, not a separate threshold or differently fitted rival.
These summaries are lossy; they do not preserve every FFT bin or ridge identity.

Each complete pronunciation traverses an ordered chain with one mandatory state
per phone. Forward summation marginalizes every monotonic phone boundary.
Repeated pronunciation labels require separate states, and no interior phone
is optional. The emission-free completion partition is subtracted separately
for each pronunciation to remove differences in admissible timing path counts.
The shared self-loop probability cancels after this completion conditioning;
this is effectively a uniform prior over complete segmentations, not a learned
phone-duration model. Both arms treat pronunciation expansion identically.
Unknown or unsupported phones, incomplete observations, non-finite inputs and
windows over 2,048 observations abstain.

`modal/student_spectral_ridge.py` fits diagonal phone emissions by
forward/backward EM, with moments shrunk toward shared manner, place and voicing
classes. Initialization uses equal-duration placement; subsequent updates
integrate all boundaries. `packages/sdk/src/bench/spectral-ridge-cli.ts` prepares
SDK observations for none-labelled help-train takes only. Speakers with any
non-training label are excluded from every take, and an explicit exclusion list
is required. The trainer independently rejects non-training, non-none,
non-help-train or excluded rows. Actual split metadata and release-speaker
exclusions still need auditing against the supplied dataset before extraction.

The proposed training-only selection compares temporal radii of one and three
observations, and shrinkage strengths 8, 32 and 128. Three hash-ordered,
speaker-disjoint folds rank their marginal predictive likelihood, averaging
within speaker first. Ties prefer stronger shrinkage and shorter radius.
Final fitting uses all eligible training speakers. Export records the input
hash, exclusions, fold membership, all validation values, selected configuration
and EM history. This selection procedure is implemented but has not run on
recordings; no training-frozen configuration has been established.

The checker exposes the exact complete written and already-accused
pronunciations at an otherwise confirmed string-path comparison, including
word-local and paired support paths. The original frame endpoints are retained:
one native 40 ms frame maps to four 10 ms observations. A veto needs the control
to strictly prefer the rival and the connectivity arm to strictly prefer the
written pronunciation. Ties and missing support abstain. Certificates match
both full pronunciations, word identity and both endpoints exactly.

Primary tokens, posterior rows, tracker decisions and graph confirmations are
untouched. The internal checker confirmation and ownership ledger retain their
baseline state; only the public wrong-word verdict becomes uncertain and its
id is excluded from the returned mistake list. This avoids changing inputs to
existing insertion merges and later candidate generation. The metadata travels
on existing decoded tokens through ordinary JSON serialization and native
frame-cell rechunking. Live audio sessions score available observations before
publishing confirmation; evidence recording adds certificates after existing
gap decoding and teacher insertion enrichment.

The normal Node loader accepts an optional `ridge-asset.json` descriptor and
verifies the declared `ridge-emissions.json` file's bytes, SHA-256, model shapes,
inventory, geometry and clocks. A declared missing or corrupt scorer fails and
releases the backend. Primary-only bundles continue to load. Neither the
supplied primary manifest nor the registry is edited.

## Checks completed and execution block

Three supervisor requests failed before any container executed, each with
`NotFoundError: No such file or directory`:

1. Run 1 requested the read-only help-train layout, splits, examples, numerical
   libraries and available scratch storage.
2. Run 2 requested the supplied six-shard build/test harness during assembly.
3. Run 3 retried that harness plus the Python scorer tests after the scorer,
   training utility and initial tests were present.

No container log or training-data layout was returned. Required TypeScript
builds and complete core/SDK Vitest suites therefore remain unverified. The
additional bundle-loader unit test was added after the last request's snapshot.

Twelve focused TypeScript checks passed locally using Node's type transformation
with workspace-local module-resolution hooks and a small assertion harness.
They cover identical local energies with different connectivity and branching, the required
matched-control reversal, exhaustive segmentation, complete repeated-phone
requirements, unsupported-input abstention, identical primary fbank outputs,
irregular audio push sizes, parameter validation, exact certificate identity,
unchanged internal issue generation and graph-confirmation preservation.
These checks execute the engine source and test cases, but do not replace the
required full Vitest suites or a TypeScript type check.

Three Python numerical unit tests passed on tiny synthetic arrays. The installed
NumPy extension is x86_64, so the default arm64 Python import initially failed;
running the same tests through `arch -x86_64 python3` succeeded. Tests verify
forward/backward likelihood and state marginals against exhaustive enumeration,
training/exclusion guards, finite EM parameters, and the control's independence
from connectivity coordinates. No recording was trained locally.

Eight Python/SDK complete-sequence scores from synthetic parameters agree to a
maximum absolute difference of 2.2737367544323206e−13 across both arms. A separate
mock-ONNX check using real temporary files passed optional, verified, missing
and corrupt Node bundle loading, with backend release on failure and no primary
inference. The session module imported successfully under the same local hooks.
Python source compilation and `git diff --check` passed. Temporary harnesses and
toy models are ignored under `.autolab`; none is a deployable acoustic asset.

Zero target recordings were accessed and zero practice scores were requested.
All three practice-score allowances remain. Target reversals, target wins,
component and overall changes, clean-session warnings, studio regression and
population preservation are unknown. There is no `.autolab/model.json`: running
the primary model alone cannot evaluate the new scorer, and missing assets must
not be presented as a completed model bundle. The ignored outcome records an
infrastructure block without a premise kill or invented target-win count.

## Resume and limits

Restore the authorized supervisor execution path; inspect help-train split and
exclusion provenance, extract observations with the SDK utility, then fit and
freeze geometry and emissions before inspecting any target comparisons.
The utility can probe arbitrary audio/evidence pairs using the normal free
tracker/checker; labels may select and report targets but must never affect
candidate discovery. Verify all four actual engine-requested comparisons and
full-stream preservation of primary evidence, tracker events, graph decisions
and insertion decisions.

Build the complete persistent bundle under this experiment's scratch directory
with the existing primary model and manifest unchanged, all required existing
teacher assets preserved, and both ridge files verified through the normal Node
loader. Only then declare the verified bundle in untracked `.autolab/model.json`.
Run the required full build/tests. If fewer than three target warnings are won,
revert the engine draft and record a killed outcome; if at least three are won,
run practice scoring before considering any narrowing based on its gate result.

Sparse training data, coarse frequency grouping, finite energy-level quadrature
and diagonal emissions may erase relevant distinctions. Equal-duration EM
initialization may converge poorly. There is no separate edge-silence model and
no independently verified phone boundary supervision. The observer needs
10–30 ms of right context for its finalized rows; sessions currently retain
observations and audio for evidence enrichment, so memory and CPU budgets still
need measurement. Node bundle loading is implemented; web/native auxiliary
installation is not. Positive written-pronunciation preference does not prove
that weak genuine substitutions survive. No acoustic success, release readiness
or scored gain is claimed, and no fold was performed.

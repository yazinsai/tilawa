# Consonant-conditioned vowel micropitch evidence

Status: blocked by supervisor container execution; draft implementation retained.
The scorer has not been fitted on real recordings, the targets have not been
inspected, and the experiment is not ready for scoring or folding.

## Prediction and mechanism

The card predicts one false warning removed, no losses counted, listener −1 and
overall −1. Its premise is that brief pitch disturbances in vowels can retain
consonant identity when the primary listener's spectral evidence prefers a rival.
The task's continuation rule requires three target wins, despite the smaller
prediction; fewer than three actual wins requires rollback and a killed outcome.
No target win or premise failure is established here.

`packages/sdk/src/micropitch.ts` estimates pitch directly from 40 ms waveform
windows every 10 ms using normalized lag correlation and interpolated periods.
It fits one robust local linear melodic contour outside a central 60 ms collar.
The contour is estimated from audio alone and shared by both pronunciations.
Unreliable periodicity, missing two-sided contour support, and gross period jumps
produce marginalized observations rather than fabricated pitch.

Every complete pronunciation symbol remains obligatory. Vowels have ordered
attack, sustain and release states; only edges immediately adjoining a consonant
condition their residual-pitch means on its identity. Length-expanded vowel
interiors do not create additional consonant disturbances. Other states use one
shared residual distribution. A forward sum integrates all phone/phase boundaries
under an identical geometric duration law. Subtraction of its identity-neutral
timing partition removes a path-count advantage. The control uses the same pitch,
contour, vowel phases, variance and duration law, with consonant identity removed.

`modal/student_micropitch_extract.ts` exports waveform-only pitch tracks and
complete canonical single-ayah pronunciations from none-labelled help-train
training takes. It rejects contradictory none labels and excludes the union of
explicitly excluded speakers and speakers with any non-training label metadata.
`modal/student_micropitch.py` independently checks that provenance. Soft EM
integrates boundaries during fitting. Hierarchical shrinkage pulls sparse
consonant-conditioned means toward their vowel/phase control means.

Three deterministic speaker folds select contour radius, periodicity reliability
and shrinkage solely by speaker-balanced training-only predictive likelihood.
The fixed grid is radii 0.20/0.35 s, reliability 0.8/0.9, and shrinkage 16/64;
not every radius/reliability combination is tried. Variance, phase durations and
the exclusion collar are fixed architecture choices. No target identifiers,
practice labels or practice score enter fitting or selection.

The checker exposes only complete comparisons from already supported non-graph
wrong-word confirmations, using exactly its existing frame bounds. A certificate
requires a strict written preference under the conditioned scorer and a strict
rival preference under the control, with reliable adjacent-vowel occupancy in
both complete paths. Missing training contexts and inadequate pitch abstain.
Certificates match the word, complete pronunciations and exact frame interval.

Only the public confirmation and `mistakes()` result can be vetoed. The internal
confirmation/ownership ledger remains intact, preserving subsequent insertion
merges and candidate discovery. Primary decoding, tracker events, graph
confirmations and confirmation constants are untouched. SDK evidence enrichment
carries certificates on existing tokens through ordinary serialization. The
Node loader reads an optional checksum-verified side descriptor and rejects
declared missing or corrupt side files.

## Validation and execution block

All three authorized supervisor requests failed before container execution:

1. Run 1: training-data layout, numerical-library and scratch-capacity audit.
2. Run 2: the supplied six-shard `tools/autolab_tests.sh` harness.
3. Run 3: that harness, Python numerical tests and a training-label audit.

Each result contains `NotFoundError: No such file or directory`, zero compute
cost, and no container log. This establishes an execution block, not missing
training data, a test failure, storage exhaustion or an acoustic kill.

Thirteen focused SDK/core tests passed locally under Node's TypeScript transform
with an ignored minimal assertion harness. They cover waveform pitch and silence,
chunk invariance, linear-contour removal, retention of a brief disturbance,
expanded-vowel adjacency, exhaustive timing summation, coverage abstention,
strict control reversal, parameter validation, generic candidate discovery,
exact serialized certificates, unchanged issue creation, and preservation of
confirmed additions and graph decisions.

Three small Python numerical/provenance tests passed in a workspace-only NumPy
environment. They verify contour separation, exhaustive forward/backward
summation and occupancy normalization, exclusion enforcement and finite toy EM.
A separate actual-filesystem Node-loader check, with ONNX inference mocked,
passed optional, verified, corrupt and missing side-asset cases and backend
release. A strict local TypeScript source check passed with workspace source
mapping and declarations for unavailable ONNX modules. `git diff --check` passed.
These checks do not replace the required full builds or core/SDK Vitest suites.

An end-to-end training CLI run on six synthetic tracks exercises all three
training-only folds, export and SDK validation. Four Python/SDK sequence scores
agree within 1e−9. This toy model is solely a numerical check; it is not a trained
pronunciation asset and is never declared as the experiment model.

Zero practice scores were requested; all three scoring allowances remain. No
real recording was processed locally. Target pitch reliability, preferences,
warning recovery, clean-session behavior and studio losses remain unknown.
Ignored verification artifacts are under `.autolab/micropitch-local/`.

## Resume and limits

Restore the authorized supervisor's container execution, audit actual training
and exclusion metadata, and run the required suites. Export eligible help-train
tracks and fit/freeze the scorer before inspecting the three actual
engine-requested target comparisons. Check certificates through ordinary JSON
serialization/rechunking and compare complete primary evidence, tracker events,
graph decisions and insertion decisions against the same baseline.

If fewer than three targets are removed, revert the engine changes, record the
measured premise result and commit a killed outcome. If three are removed,
finish a persistent bundle containing unchanged primary ONNX/manifest and
installed teacher assets plus the verified micropitch descriptor/emissions.
Verify that complete bundle before declaring untracked `.autolab/model.json`,
then request practice scoring before narrowing on gate losses.

There is currently no `.autolab/model.json` and no real trained side asset.
Running the main model without that asset would not evaluate this experiment.
The ignored outcome records an infrastructure block rather than a kill or gain.
No supplied manifest, evidence, test data, scorer, lab tooling or experiment card
was edited. No fold was performed.

Lag correlation can confuse harmonics with pitch, and melodic ornaments can
overlap the presumed consonant timescale. The fixed geometric duration law and
obligatory expanded labels can misfit rapid or very long renditions. The model
uses scalar Gaussian residual emissions, not a calibrated full acoustic
likelihood; soft EM has no independent phonetic boundary supervision and can
learn poor placements. The consonant-independent control cannot distinguish
sequences with identical vowel/phase structure, causing abstention on such
contrasts. Sparse training contexts can also abstain or generalize poorly.

The draft duplicates waveform analysis and buffers audio/evidence. CPU, memory,
full-stream preservation and live-versus-enriched decision consistency remain
unmeasured. The two-sided contour requires local future audio; end-of-session
enrichment can have more contour context than an early live decision. Automatic
side-file loading is implemented only for Node. A positive pitch preference
does not establish that a real weak substitution will survive the veto.

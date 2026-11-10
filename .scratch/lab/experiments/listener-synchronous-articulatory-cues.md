# Synchronized articulatory acoustic cues

Status: needs-human — implementation draft, blocked before Modal execution.
The scorer has not been fitted on real recordings and the branch is not ready
for practice validation. `lab-experiment.json` is unchanged.

## Prediction and premise

Predict listener −1 and overall −1: two false substitutions removed, with one
caught substitution lost. The proposed defect is that place, manner and voicing
support from different moments can jointly describe a consonant never spoken.
The test requires a complete written pronunciation to beat the complete current
accusation under synchronized timing, while the same emissions prefer the
accusation when feature streams choose their own timings.

The registered windows are the four incidents in the task: p11 at 26:106:1,
p36 at 55:48:1, p55 at 26:174:8, and sob-049 at 92:13:1. They appear only in
this description, not in implementation or training selection. No target audio,
labels or evidence was accessed. Target recoveries and preference reversals are
unknown, not measured zeroes.

The card asks for at least two target preference reversals. The task's later
instruction requires at least three actual target warnings removed before
practice scoring. Neither threshold has been evaluated. A supervisor execution
failure does not establish that the acoustic premise is false.

## Implementation

`packages/sdk/src/synchronous-cues.ts` extracts twenty coordinates by averaging
adjacent groups of four bins from the existing 80-bin log-mel frontend. Native
25 ms windows, 10 ms shifts, pre-emphasis, DC removal and timing are retained.
Four observations correspond to one 40 ms evidence frame. The same SDK extractor
supplies training and inference observations, independently of student logits.
Streaming extraction is invariant to audio push size.

Every original phone maps to three categorical articulatory attributes: place,
manner and voicing. Each attribute class has a diagonal Gaussian emission over
the acoustic coordinates. The model includes a background class. Pronunciation
graphs retain every original phone, with optional intervening background states,
self loops and forward edges. Repeated identical articulations require
background separation; phones with identical attributes retain their original
identities for this topology.

The synchronized score sums the three attribute emissions at each graph state
before marginalizing all boundaries. Thus one consonant interval must supply all
three cues together. The independently timed control runs the same complete
graph separately for each attribute stream and adds its three marginal scores.
Both arms use identical observations, emissions, pronunciation labels and
transition probabilities. Each graph score subtracts its emission-free
completion partition, conditioning the normalized timing prior on completing
the pronunciation in the available window. This tests synchronization of
feature support, rather than changing the written-versus-rival correspondence.
Shared vowels and context phones also use shared boundaries in the synchronized
arm; the restriction is stronger than coupling only changed consonants.

Unsupported phones, incomplete or nonfinite windows, more than 4,096
observations, and identical written/rival feature predictions abstain. Ties
abstain. A veto requires a strict independently timed preference for the rival
and a strict synchronized preference for the written pronunciation.

`modal/student_synchronous_cues.py` fits the three emission streams with shared
latent phone boundaries by forward/backward EM. Initial uniform placements are
only initialization; subsequent sufficient statistics integrate every valid
phone/background path. Attribute sharing and shrinkage toward a global
distribution regularize sparse classes. The variance floor is fixed at 0.05 in
training-standardized coordinates. Only none-labelled, explicitly non-excluded
training speakers supply transcripts; mistake scripts are never used as
performed transcripts.

Model selection uses three SHA-256-ordered speaker-disjoint folds entirely
inside the training split. Shrinkage candidates 8, 32 and 128 and self-loop
probabilities 0.7 and 0.9 are selected by mean held-speaker marginal likelihood,
with deterministic ties favoring stronger regularization. Eight EM passes are
fixed in advance. Final fitting uses all eligible training speakers. The export
records input hash, exclusions, fold membership, selection scores, final fit
history and attribute names. These are an unexecuted recipe, not frozen learned
parameters: actual split provenance still needs auditing on Modal.

`packages/sdk/src/bench/synchronous-cues-cli.ts` extracts canonical ayah
pronunciations and SDK observations only from none-labelled training records
with empty error/repeat lists and a required external speaker exclusion list.
It rejects fewer than three eligible speakers. The supplied dataset layout and
metadata could not be checked, so this extraction utility has not run on
help-train.

The checker callback exposes only an already-confirming non-graph comparison,
including its complete written and accused pronunciations and exact original
frames. This covers ordinary, word-local and paired string comparisons.
The SDK either performs the comparison before live confirmation or discovers
the same comparisons in a read-only replay and serializes certificates on token
metadata after existing gap decoding and teacher insertion enrichment.
Evidence-only sessions consume those certificates without waveform access.

Certificates must match word, both pronunciations and both endpoints exactly,
and carry finite scores satisfying the reversal. The checker preserves internal
confirmation/ownership state so the substitution-tail insertion merge continues
to use baseline decisions. Only the public wrong-word confirmation becomes
uncertain and leaves the returned mistake list. Primary decoding, tracker
inputs/events, graph confirmations and insertion decisions have no new decision
route. Full-stream equality on actual recordings remains unverified.

The Node loader optionally loads `cue-asset.json` and its fixed
`cue-emissions.json` file next to the primary model. It validates byte count,
SHA-256, inventory, clocks, class indices, parameter shapes, variances and
transition probabilities. Declared missing or corrupt assets throw and release
the backend. Primary-only bundles retain their existing loader behavior. No
supplied manifest, registry data, scorer, lab tool or test data was modified.

## Validation and execution block

Supervisor run 1 requested read-only inspection of help-train, dataset layout,
numerical libraries and available output storage. It failed with
`NotFoundError: No such file or directory`, before any container execution.
Runs 2 and 3 requested the supplied six-shard `bash tools/autolab_tests.sh`
harness and failed identically. All three result files report zero compute cost
and contain no container logs. This is different from the
earlier experiments' in-container storage exhaustion; no storage-recovery
command can run through this failed execution path.

Local checks used ignored, workspace-contained dependencies. Seventeen focused
Vitest cases passed: six SDK numerical/frontend cases, five core certificate
and graph-preservation cases, and six mocked bundle/teacher loading cases.
These include a synthetic rival assembled from three differently timed cues,
retention of simultaneous spoken-rival cues, exhaustive normalized path sums in
both arms, mandatory repeated articulation, JSON certificate transport, changed
window/pronunciation rejection, and unchanged primary backend loading.

Three Python numerical tests passed: forward/backward and state marginals match
exhaustive path enumeration; original-label repetition topology is retained; and
training/exclusion provenance plus a tiny synthetic EM fit are valid. Eight
Python/SDK scores from that toy fitted model agreed to a maximum absolute error
of 2.2737367544323206e−13. No recording was fitted on the Mac.

Strict TypeScript checking of core and SDK sources passed with temporary
module aliases and stubs for unavailable ONNX runtimes. Python numerical tests
used an isolated NumPy/SciPy environment because the system NumPy binary has
the wrong architecture. These local checks establish mathematical and transport
properties; they do not replace the required Modal build and full core/SDK
suites, real acoustic validation or an independent listening opinion.

Zero practice scores were requested; all three allowances remain. No model
bundle or `.autolab/model.json` is declared. Running the main listener without
the unfitted side scorer would not evaluate this change. The branch is held as
a draft, with an ignored infrastructure-blocked outcome; it is not marked
killed or ready to fold.

## Remaining work and limits

Restore supervisor container execution, audit the actual help-train split and
exclusions, extract eligible recordings, then fit and freeze emissions before
inspecting any target comparison. Verify a complete persistent bundle containing
the unchanged primary listener/manifest, existing installed teacher files and
the new cue asset/descriptor before declaring `.autolab/model.json`.

Run the required Modal suites and compare full-stream primary posteriors,
tracker events, graph confirmations and insertion decisions with the side
scorer enabled. Discover target windows through the actual engine callback,
measure the four complete preference comparisons and resulting warnings, and
apply the task's continuation rule. Fewer than three target recoveries requires
rollback and a measured killed outcome; three or more requires practice scoring
before considering any narrowing.

The current categorical classes may be too coarse, diagonal emissions can
misread correlated acoustic features, and uniform initialization can converge
to poor boundaries. The three heads reuse the same spectral observations, so
their emission product is a discriminating score rather than a calibrated joint
acoustic density. Timing normalization does not eliminate that limitation.
Feature synchronization can remove useful timing evidence from genuine weak
substitutions. The implementation buffers session observations/audio and has
unmeasured runtime cost; browser/native bundle installation is not implemented.
No measured claim about target wins, losses, clean sessions or studio audio is
possible without the real fit and execution checks.

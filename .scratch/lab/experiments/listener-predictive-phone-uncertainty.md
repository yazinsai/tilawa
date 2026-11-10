# Posterior-predictive phone uncertainty

Status: implementation draft; training, target validation and complete model bundle
blocked by the authorized Modal execution path. The experiment card is unchanged.
No acoustic premise result or practice gain is claimed.

## Prediction and mechanism

Predict listener −1 and overall −1, removing one false wrong-word warning with no
predicted losses. The card requires at least two written-versus-accused preference
reversals. The task's stricter continuation rule requires at least three actual
target warnings removed before practice scoring; both remain untested.

The CPU trainer fits full-covariance, eight-dimensional phone emissions on frozen
native log-mel features. Each coordinate averages ten adjacent mel bins. It uses
only none-labelled help-train training-speaker recordings. Mistake scripts are
not assumed to be performed transcripts. Input provenance is checked explicitly;
any speaker with a non-training take is excluded during extraction, and an
explicit exclusion list is required. Release-held-out speakers must also be
absent from that input and exclusion provenance must be verified before training.

Normal/inverse-Wishart priors share their moments across manner, place and voicing
classes, backing off to global moments. A phone's own observations are excluded
from its class pool. Forward/backward EM integrates ordered phone boundaries and
produces fractional sufficient statistics. Three prior strengths, 2, 8 and 32,
are selected by three speaker-disjoint, training-only validation folds, ranked
by mean posterior-predictive likelihood with equal speaker weighting. The
selected prior is then frozen and all eligible training speakers are refitted.
The trainer exports the posterior, its descriptor and a provenance/selection
report. None of this selection uses practice windows or release data.

The SDK's point control uses the posterior mean emission mean and covariance.
The predictive arm samples both parameters jointly from the same fitted phone
posterior, sharing each sampled distribution across every frame and every
repeated occurrence of that phone in the complete pronunciation. Each draw
marginalizes all ordered phone boundaries; the result averages likelihoods in
log space, rather than averaging log likelihoods. Both hypotheses use the same
frames, duration prior, shared phone draws and complete pronunciation semantics.
The duration prior is normalized conditional on complete pronunciation; every
phone must emit at least one observation. No target phone is erased or optional.

A single-class complete pronunciation has an exact conjugate joint predictive
calculation; its boundaries cancel. Other pronunciations use deterministic
128-draw integration. A certificate requires the point estimate to prefer the
accusation, and both the first 64 draws and all 128 draws to prefer the written
pronunciation. These limits and the random streams are fixed in the source;
they have not been selected against any incident. Adding or reversing a candidate
does not change its shared phone draws.

The checker exposes the complete comparison and exact endpoints from an already
supported non-graph wrong-word confirmation. This includes its ordinary,
word-local and paired string comparisons. The SDK computes the live veto before
publishing the warning, and enriches recorded evidence through the ordinary
primary-token metadata route after existing gap and teacher-insertion processing.
An evidence-only client can consume that certificate without waveform access.
A certificate must match the word, both complete pronunciations and both endpoints
exactly, and contain finite scores with the required reversal.

Only the public confirmed wrong-word verdict becomes uncertain. Internal issue
creation, confirmation and ownership remain baseline, preserving insertion-tail
merges and subsequent candidate generation. Primary phone values, posterior rows,
tracker events, parallel-graph confirmations and all other warning classes are
preserved by this route. Missing coverage or unsupported phones abstain.

The Node model-directory loader verifies an optional `predictive-phone-asset.json`
and its fixed `predictive-phone-emissions.json` filename, byte count, SHA-256,
inventory, frontend clock, posterior shapes and positive-definite matrices.
A declared missing or corrupt file fails and releases the backend. Legacy bundles
without a descriptor retain their normal behavior. No supplied manifest or
registry data was edited.

## Validation and execution block

Three authorized supervisor requests were made:

1. Run 1 requested a read-only help-train/data layout and scratch-capacity check.
   It failed before container execution: `NotFoundError: No such file or directory`.
2. Run 2 requested `bash tools/autolab_tests.sh` on six shards while the draft was
   being assembled. It failed before container execution with the same error.
3. Run 3 requested the six-shard harness after the implementation and initial
   focused checks were complete. It failed before container execution with the
   same `NotFoundError: No such file or directory`.

No alternate Modal path was attempted. No help-train recording, supplied model
or target waveform was accessed. No training fit or speaker-grouped validation
on real audio executed. Target eligibility, reversals, warning recovery, losses,
clean-session behavior and studio behavior are unknown. Zero practice scores
were requested; all three practice-score allowances remain.

Local checks use only small synthetic inputs and ordinary repository fixtures:

- 19 focused Vitest cases cover complete-boundary sums, candidate-order invariance,
  repeated-phone completion, an actual sparse-posterior reversal, full covariance
  sampling, unsupported-input abstention, native frontend rechunking, inventory
  and covariance validation, exact certificate matching, unchanged issue creation,
  graph preservation, and verified/missing/corrupt optional assets. The loader
  tests mock ONNX execution; they do not validate a real ONNX bundle.
- Three Python numerical/provenance tests cover exhaustive boundary enumeration,
  normalized forward/backward marginals, exclusion and training-only enforcement,
  positive-definite toy EM fits and a finite toy predictive validation score.
- The exact one-frame conjugate predictive matches an independent SciPy
  multivariate Student-t calculation: −23.575775721061724 nats for the recorded
  synthetic parameters. No recorded audio was fitted on the Mac.
- TypeScript checking passed over core and SDK sources with temporary ambient
  declarations for unavailable ONNX peer packages. Python syntax compilation and
  `git diff --check` passed. These are not the required complete Modal suites or
  real-runtime integration checks.

All local environments and diagnostic artifacts are ignored under `.autolab`.
No fitted side asset or complete export bundle exists. No `.autolab/model.json`
is declared: decoding the main model alone would not evaluate this experiment.
The ignored outcome records an infrastructure block, without claiming a kill
or reporting untested targets as zero wins.

## Resume and limits

Restore the supervisor execution path, inspect help-train's actual metadata and
training/held-out provenance, then use the SDK observation utility and CPU trainer
to fit and freeze the scorer. The generic comparison probe discovers every actual
engine-requested window from supplied evidence; labels may select cases for
reporting, but cannot choose pronunciations or endpoints. Validate all four
registered targets and replay enriched evidence to measure actual warning removal,
while checking primary evidence, tracker events, graph confirmations and insertion
decisions. Kill and revert engine changes if the premise fails or fewer than three
target warnings are removed. Otherwise request practice scoring before narrowing.

Before declaring a model selection, assemble and verify the complete bundle in
this experiment's persistent Modal scratch directory. Preserve the original
primary ONNX, manifest and installed teacher assets, and add the verified scorer
and descriptor. Only then write the untracked model selection and run the required
complete build/test harness. Missing declared files must fail validation.

The emission posterior is empirical Bayes conditional on fixed class priors and
EM fractional statistics; it is not a full joint posterior over priors and
training alignments. Correlated 10 ms observations can overstate effective sample
size. Eight pooled coordinates may discard useful distinctions, and uniform
initialization may give poor boundaries. The current chain has no independent
background state for edge silence and uses one fixed duration persistence.
Monte Carlo integration can miss likelihood-dominating draws, particularly in
long, high-dimensional windows; agreement of 64 and 128 draws is only a sign
stability check, not a bound on integration error. Small training coverage may
cause abstention or misleading shrinkage. The session retains observations and
audio for enrichment; real-device memory and latency have not been measured.
Node installation is implemented; web/native auxiliary installation is not.
Positive predictive support cannot guarantee preservation of a genuine weak
substitution. These limits require real training and target validation before
any release or fold conclusion.

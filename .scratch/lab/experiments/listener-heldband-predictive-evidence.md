# Held-band predictive acoustic evidence

Status: needs-human — implementation committed, but the supplied Modal supervisor
cannot start commands. Training, full unit tests, the complete model bundle and
target validation remain unfinished. This branch is not ready for scoring or
folding. The experiment card is unchanged.

## Prediction

The card predicts two false wrong-word warnings removed and one caught
substitution lost: listener −1 and overall −1. It proposes that an erroneous
pronunciation can fit dominant spectral energy while failing to predict the
remaining observed spectrum.

The task's continuation rule is stricter than the card's two-target prerequisite:
at least three of the four registered targets must actually be vetoed before
practice scoring. Neither prerequisite has been measured. Infrastructure failure
is not evidence that the premise is false, so this experiment has not been killed.

## Implementation

`modal/student_heldband_prepare.ts` extracts disjoint FFT-power bands and
full-inventory primary-listener posteriors from authorized, none-labelled
help-train recordings. Recordings with event or repeat labels are excluded.
The help-train directory excludes release speakers by contract; the preparer
additionally excludes every speaker explicitly marked eval, test or release.
It refuses insufficient training-speaker coverage. None of the target sessions,
reader identities, verse references or their labels enters the training code.

`modal/student_heldband.py` initializes latent acoustic occupancies using a
canonical-transcript CTC forward-backward calculation, then fits a small acoustic
model by EM without fixed phone boundaries. It fits conditional Gaussian
predictors for each held-band partition, shrinking phone-specific coefficients
toward a pronunciation-independent predictor. A deterministic speaker partition
selects band grouping and regularization by held-out predictive likelihood, with
equal speaker weighting. The selection grid is fixed in code; selection sees only
training speakers. The selected configuration is then refitted on all authorized
training speakers. The exporter checks the primary model's supplied checksum and
copies its graph, manifest and existing auxiliary files byte for byte before
adding `held-band.json`.

The SDK extracts the same disjoint spectral observations on the normal 10 ms
analysis grid. Each complete candidate's timing prior uses only the visible
bands. Its withheld-observation likelihood is the joint forward partition divided
by the visible-only partition. Subtracting the common pronunciation-independent
prediction prevents predictable background from becoming lexical advantage.
The two pronunciations use identical observations, normalized timing priors and
complete-label acceptance. Unsupported phones, missing frames, impossible
pronunciations and nonfinite computations abstain.

The ordinary Node model loader recognizes the side asset in the primary bundle,
validates its partitions and parameters, and rejects an incompatible inventory.
The checker supplies the exact complete written and already-accused pronunciations
and its existing acoustic window. A positive written-over-accused likelihood
margin in both complementary partitions supplies a veto certificate. There is no
new fitted confirmation margin. Certificates retain both full hypotheses and
frame bounds through JSON and ordinary token-preserving evidence rechunking.

Only the published non-graph wrong-word verdict becomes uncertain. The checker's
internal confirmation and ownership state remains unchanged so subsequent
insertion and graph decisions retain their original inputs. Primary listener
phonemes, confidence values, posterior rows and tracker events are preserved.
Audio-backed SDK sessions also discover certificates against their final recorded
evidence without feeding replay events into the primary tracker.

## Validation completed

- Python syntax compilation passed for the trainer and its numerical tests.
- `git diff --check` passed.
- Lightweight local Node smoke checks passed for a complete-window core
  substitution certificate, disagreement abstention, identical tracker events,
  exclusion of the vetoed warning from `mistakes()`, spectral chunk invariance,
  opposite written/rival acoustic preferences, and incomplete-label abstention.
- Added core and SDK unit tests cover those behaviors, exact window and hypothesis
  matching, nonfinite evidence, background cancellation, exhaustive normalized
  timing paths, sparse-phone abstention, model loading, and evidence serialization.
  Added Python tests compare forward-backward probabilities and occupancies with
  exhaustive paths and check conditional predictor input dimensions and shrinkage.

These smoke checks use synthetic observations and unrelated unit fixtures. They
are not target measurements, trained-model validation or practice scores.

## Infrastructure block and remaining work

Three requests through the supplied supervisor failed before command execution:

1. Modal run 1: read-only inspection of authorized training inputs.
2. Modal run 2: minimal `pwd` environment check.
3. Modal run 3: the required six-shard `bash tools/autolab_tests.sh` run.

Each returned `NotFoundError: No such file or directory`, with no command output
or shard logs. The raw supervisor results remain in the ignored
`.autolab/modal-result-{1,2,3}.json` files. The restoration request was sent to the
user while implementation continued.

No authorized recording was inspected or decoded, no scorer was trained, no model
bundle was exported, and no target preference or target recovery count is known.
The full core/SDK tests and Python numerical tests still need successful remote
execution. All three permitted practice scores remain available.

After the supervisor environment is restored, prepare and train the scorer using
the commands in the trainer's module docstring, placing outputs beneath
`/scratch/listener-heldband-predictive-evidence/`. Verify the complete bundle through
the normal Node loader, acoustic feature and scorer parity, certificate transport,
and unchanged primary and insertion behavior. Freeze the scorer before evaluating
the four targets on the actual checker-requested complete windows. Targets are
selected for reporting only. If fewer than three are vetoed, roll back the engine
change, record the measured kill and commit the result. If three or more are
vetoed, run the authorized practice score before deciding whether to narrow it.

`.autolab/model.json` has not been declared because the complete trained bundle
does not exist. Decoding the main model alone would not evaluate this experiment.
Finite training-data coverage, conditional-Gaussian model error, sparse phones,
and weak true substitutions remain unmeasured risks.

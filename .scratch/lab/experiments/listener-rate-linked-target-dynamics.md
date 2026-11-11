# Rate-linked spectral target dynamics

Status: needs-human — the scorer, training utility and tests are implemented,
but all three supervisor Modal requests failed before command execution with
`NotFoundError: No such file or directory`. Training, the complete model bundle,
TypeScript checks and acoustic target validation remain pending. This is not a
premise kill or a measured result. The experiment card is unchanged.

## Prediction

Predict two false warnings removed and one caught substitution lost: listener
−1 and overall −1. Registered windows are p11 26:106:1, p36 55:48:1,
p55 26:174:8 and sob-049 92:13:1. The card asks for two preference reversals;
the task's continuation rule takes precedence: fewer than three recovered
warnings requires rollback, while three or more requires a supervisor practice
score before deciding or narrowing. No target has been tested, and no practice
score has been requested; all three scoring opportunities remain.

## Mechanism

`modal/student_rate_dynamics.py` fits a conditional spectral-state model on
none-labelled help-train training-speaker recordings. On each 10 ms observation,
the state approaches the current phone's spectral target:

    x[t] = rho * x[t-1] + (1-rho) * target[phone] + innovation

Retention is shared across phones, with one coefficient per spectral band,
bounded below one. A rendition lasting d observations contracts its expected
shortfall by rho^d. The spectral state is the observed, normalized acoustic
spectrum, not a listener posterior, phone spike or measured articulator position.
Thus this is a compact conditional Markov approximation to articulatory motion,
not a hidden continuous-state articulatory reconstruction. Short sounds can end
before reaching their targets; stretching playback cannot supply the missing
spectral trajectory.

The independent-realization control uses the same phone-target parameterization,
observations, complete-sequence topology, fixed innovation reference and target
pooling. Instead of learning bandwise retention, it learns bandwise log variance
scales around stationary targets. Each arm has exactly
`(inventory size + 1) * 20 + 20` free parameters. Both are independently fitted;
the control is not made by disabling the treatment's learned coefficients.
They have equal parameter counts, although retention and dispersion are
different kinds of flexibility, so equal counts do not prove equal statistical
capacity.

Observations use the primary frontend's 25 ms analysis windows and 10 ms hop.
Groups of four mel-band powers are summed into 20 bands and normalized by total
spectral mass before logging. Training-only means and scales normalize these
observations. A fixed innovation reference is estimated from training temporal
differences; neither normalization nor noise scale is estimated on practice.

Exact forward/backward sums marginalize phone boundaries. Every phone is
mandatory and occupies at least one observation, including adjacent identical
phones. The acoustic partition is divided by the complete timing partition,
giving a normalized uniform prior over complete nonempty segmentations.
There are no optional target consonants or candidate-specific dynamics.

Three deterministic speaker folds select one regularization strength from
8, 32 and 128 using the mean held-speaker density of both arms. Target means
shrink toward training-only phonetic-group pools; shared coefficients receive
the same regularization unit. Both arms refit on all eligible training speakers
with that frozen strength. The extractor rejects non-training takes and excludes
all takes of any speaker with a non-training label, in addition to an explicit
exclusion list when provided. The supplied help-train mount is stipulated to
omit release-held-out speakers; its contents could not be checked here.
Mistake-labelled recordings and practice inputs never enter this estimator.
The utility records input hash, speaker partitions, selection, optimizer history,
parameter count and asset hash.

## Engine path

The checker exposes the complete written and already-accused strings and exact
frame bounds of an existing, supported string-path confirmation. This hook and
its transport follow the existing anticipatory-gesture experiment's route;
its acoustic model is replaced by the rate-linked target model.

The SDK scores both pronunciations over the identical native window. Four
10 ms observations correspond to one existing 40 ms evidence frame. The immediately
preceding acoustic observation initializes the dynamic state for both hypotheses;
at recording start the training mean supplies the common initial state.
A veto requires strictly greater dynamic likelihood for the complete written
pronunciation and strictly greater control likelihood for the rival. Missing
observations, unsupported phones, impossible complete pronunciations and
non-finite scores abstain.

Certificates retain both strings, the occurrence, exact bounds and all four
likelihoods. The consumer checks the strict finite reversal and exact comparison
identity before changing the public confirmation to uncertain. Its internal
confirmation and ownership ledger retains the baseline result, so subsequent
candidate generation and insertion merges keep their original inputs. The live
session performs this check before publication, including when evidence recording
is disabled. End-of-session enrichment annotates stored tokens so ordinary scored
replay can consume the same veto. Primary tokens and rows, tracker events,
parallel-graph confirmations and insertion decisions have no new scoring rule.
These preservation properties still need runtime verification.

The normal Node loader accepts a `rate-asset.json` side descriptor and checks
`rate-dynamics.json` byte length, SHA-256, inventory, clock, covariance and
transition bounds. Missing or corrupt declared files fail loading. No existing
model manifest or registry was changed. No trained asset or bundle currently
exists and `.autolab/model.json` was deliberately not written: the main model
without these assets does not evaluate the acoustic change.

## Validation and blocker

Five local synthetic numerical tests pass using:

    arch -x86_64 python3 -m unittest discover -s modal -p test_student_rate_dynamics.py

They verify exhaustive complete-segmentation likelihoods and posterior
occupancies, duration-dependent approach to a target, independence of control
realizations from preceding observations, both analytic fitting gradients,
mandatory repeated-phone completion and rejection of ineligible training data.
The gradient test uses central differences; forward differences initially had
numerical error slightly exceeding its tolerance, so the numerical reference was
corrected without changing the optimizer. Python syntax and `git diff --check`
also pass. These are tiny synthetic numerical checks, not training on the Mac.

Added TypeScript tests cover exhaustive partitions in both arms, a short
undershot rendition versus its completed counterpart, shared incoming state,
coverage and finite-score abstention, model validation, push-independent spectral
observations, exact certificate matching, graph preservation, unchanged internal
issue creation, live publication without evidence recording, and checksum-verified
bundle loading. They have not executed.

Supervisor requests:

1. Read-only storage/data/model inspection: failed before execution.
2. Required six-shard `bash tools/autolab_tests.sh`: failed before execution.
3. Python tests and training-observation preparation: failed before execution.

All returned `NotFoundError: No such file or directory`, with no command logs,
data or training outputs. No acoustic inference, independent listening review,
target recovery measurement or practice score ran. The third request's original
command also named an exclusion file that had not been created; the extractor
now derives and exports exclusions from label splits, so use the corrected
commands below when the supervisor is restored. That command defect was not
reached by the reported infrastructure failure.

## Resume after supervisor recovery

Run the required six-shard tests. Extract eligible observations under Modal:

    mkdir -p /scratch/listener-rate-linked-target-dynamics
    pnpm --filter @tilawa/core build
    pnpm --filter @tilawa/sdk exec tsx src/bench/rate-dynamics-cli.ts \
      --labels /lab/data/help-train/labels.jsonl \
      --out /scratch/listener-rate-linked-target-dynamics/train.jsonl

The extractor writes `train.jsonl.metadata.json` with inventory/tokens and
`train.jsonl.excluded.json` with excluded speakers. Feed those exact values into
`student_rate_dynamics.py --input ... --tokens ... --inventory ...
--excluded-speakers ... --out ...`, using structured Python arguments to retain
the pronunciation alphabet literally.

Before target inspection, complete and verify a directory under the experiment's
persistent scratch path containing the unchanged shipped `model.onnx` and
`manifest.json`, all shipped teacher files, the fitted `rate-dynamics.json` and
its `rate-asset.json` descriptor. Load it through `loadNodeModel`, verify the
primary and teacher hashes, and compare Python/SDK likelihoods. Only then declare
the complete relative directory in untracked `.autolab/model.json`.

Use `rate-dynamics-cli.ts --emissions ... --audio ... --evidence ... --out ...`
to inspect ordinary engine-requested comparisons. The probe asserts primary
rows, tracker events and non-wrong-word decisions remain identical. Count actual
registered warnings removed, not merely likelihood reversals on hand-cut windows.
Apply the task's three-target continuation rule; if it passes, request practice
scoring before deciding whether any narrowing is needed.

## Limits

Observed spectral-state dynamics also capture acoustic autocorrelation, so a
reversal would support this acoustic representation without proving physical
articulatory undershoot. Normalization discards loudness; 20 bands can discard
place cues. Mandatory phone segments attribute pauses to adjacent phones rather
than modelling a separate silence process. Phone coverage may force abstention;
small-data targets and dynamic persistence can forgive genuine substitutions.
Equal parameter counts, shared dynamics and strict reversals do not guarantee
preservation. Training runtime, SDK latency, long-session memory, target gain
and full-set collateral effects are unmeasured. The supplied execution service
must work before the experiment can be completed or judged.

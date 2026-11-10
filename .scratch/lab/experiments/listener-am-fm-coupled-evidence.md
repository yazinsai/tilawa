# Coupled amplitude and instantaneous-frequency evidence

Status: needs-human — implementation draft. All three authorized supervisor
requests failed before container execution with `NotFoundError: No such file or
directory`. Training, target validation and the complete exported bundle remain
blocked. This branch is not ready for practice scoring. The card is unchanged.

## Prediction and mechanism

Predict two false warnings removed and one caught substitution lost: listener
−1, overall −1. The speculative premise is that a misleading pronunciation may
fit amplitude structure and frequency motion separately while failing their
within-band coordination. The registered card requires at least two preference
reversals; the task's continuation rule additionally requires at least three
actual target warnings removed before requesting practice scoring. Neither
condition has been measured. Infrastructure failure is not a premise kill.

The SDK extracts waveform observations with eight fixed complex two-pole bands
between 250 and 5,500 Hz. Complex phase increments, after removing the filter's
centre frequency, measure within-band instantaneous frequency. Over the native
25 ms windows advanced by 10 ms it retains changes in log amplitude and in
amplitude-weighted frequency. Filter state and window accumulation are independent
of caller push size. Existing 40 ms warning frames index four observations;
the acoustic window endpoints and both complete pronunciations are unchanged.

Each band has a phone-conditioned bivariate Gaussian. Reliability decreases
when either of two successive amplitude observations is weak relative to local
waveform power. The likelihood integrates an unreliable frequency observation
out, using a phone-independent standardized frequency background to keep the
observed and missing branches on a common measure. Zero reliability removes
frequency and covariance evidence exactly. Intermediate reliability mixes the
observed and marginalized explanations; it does not replace an unknown frequency
with zero or delete a pronunciation phone.

The matched independent control has exactly the joint model's amplitude and
frequency means, marginal variances, observations, reliability, duration prior,
phone inventory and completion requirement. It sets the within-band covariance
to zero. Every phone must emit at least once; self loops integrate all complete
phone boundary placements, including separate repeated articulations. An
emission-free partition normalizes the timing prior separately for each complete
pronunciation. A veto needs a strict joint preference for the written sequence
and a strict independent preference for the already-accused sequence. Ties,
unsupported phones and invalid or oversized windows abstain.

## Training and integration

`modal/student_am_fm_prepare.ts` extracts the exact SDK observations from
help-train labels and waveform files. It accepts only explicitly none-labelled,
non-excluded training-speaker single ayahs without mistake/repeat annotations,
rejects speakers appearing in any non-training split, and retains source label
and audio hashes. Other sample rates use the SDK resampler. No scripted mistake
is converted into a performed transcript. The actual supplied metadata still
needs auditing because the read-only data request never executed.

`modal/student_am_fm.py` fits complete-sequence phone boundaries by
forward/backward EM. Missing-frequency posterior moments are integrated in the
sufficient statistics. Covariance estimates shrink toward a shared diagonal
prior, with positive marginal variance floors and positive-definite covariance
bounds. Duration treatment is fixed from training transcript/frame lengths and
shared between comparison arms. Three hash-ordered speaker folds choose between
four/eight bands and shrinkage 16/64/256 by held-speaker predictive likelihood;
ties favour stronger shrinkage and fewer bands. The final training-only fit
exports parameters, a byte-count/SHA-256 descriptor, and a report containing fold
membership, exclusions, selection scores and input hash. No target or practice
score participates in selection.

The checker exposes its actual supported non-graph written-versus-rival
comparison, including word-local and paired paths. Its internal confirmation,
ownership and issue ledger keep the baseline verdict, preserving insertion
merges and later candidate generation. Only the public wrong-word confirmation
and returned mistake list can be vetoed. Suppression attaches to the specific
emitted event, so a subsequent graph event is not suppressed by sharing an ID.

Audio sessions compute the recheck before publishing confirmation. Evidence
recording attaches exact-window certificates to ordinary token metadata after
existing gap and teacher-insertion enrichment. Evidence-only replays consume
those certificates; they must match the word, complete sequences and both frame
endpoints exactly and contain finite scores with the required reversal. Primary
phoneme identity, confidence, timestamps and posteriors are never overwritten.
Graph confirmations and other warning classes do not consult the scorer.

The normal Node model loader accepts an optional `am-fm-asset.json` descriptor
and verifies the fixed `am-fm-emissions.json` file, byte count, SHA-256, inventory,
clock, shapes and positive-definite covariance. Missing/corrupt declared files
raise an error and release the backend. Primary-only bundles load normally.
Supplied manifests and registry data are untouched.

## Validation and execution limits

Supervisor run 1 requested a read-only help-train/data/storage inventory.
Runs 2 and 3 requested the supplied six-shard `bash tools/autolab_tests.sh`
harness. Each returned `NotFoundError` without container logs. No training data,
real target window or deployed model was accessed through these requests.

Local focused Vitest checks passed: six core tests cover discovery without
verdict changes, finite strict reversal, exact window/pronunciation matching,
unchanged internal issue creation, added-word preservation and graph preservation. Thirteen SDK checks
cover covariance-sensitive comparisons, retention of the rival when its coupling
matches, exact unreliable-frequency marginalization, exhaustive complete-boundary
summation, mandatory repeated phones, invalid-input abstention, native observation
clock and arbitrary audio chunking, covariance/inventory validation, verified
asset loading, missing/corrupt asset rejection and existing insertion loading.

Three Python numerical tests passed in an isolated ignored NumPy environment:
forward/backward values and occupancies match exhaustive paths, unreliable
frequency is marginalized symmetrically, and a tiny synthetic EM fit enforces
provenance and produces finite positive-definite parameters. No recording was
trained on the host. Eight synthetic Python/SDK sequence scores, covering both
arms and different complete pronunciations, agree to maximum absolute error
3.55e−15. Both TypeScript builds and standalone type-checking of the training
preparation utility passed. `git diff --check` and Python syntax compilation
passed.

The full SDK suite passed locally with one worker and a 120 s process timeout:
35 passed, one optional integration test skipped. Its first two-worker run hit
a 30 s timeout in the existing evidence-chunking session test; the unchanged
test passed on the final run. The initial unrestricted core run also hit
worker-contention timeouts and was interrupted. The constrained core rerun completed with 759 passed and two existing
startup-persistence cases timing out at their explicit 15 s limit, plus two
Vitest `onTaskUpdate` RPC timeouts. The startup file also timed out on both
cases when run alone. The additional six-case core AM-FM file passed after
the insertion-preservation test was added. No test implementation or repository
timeout setting was changed. An ignored Vitest alias substituted the unchanged pre-experiment checker
for the standalone startup test: that baseline also timed out on one of the
two cases. Its aggregate test time was 34.47 s versus 34.38 s for the draft.
This does not establish a passing core run; the required Modal suites remain
unverified. These host checks do not replace Modal validation.

Zero practice scores were requested; all three allowances remain. Target wins,
losses, overall change, clean-session behavior and studio behavior are unknown.
There is no fitted help-train scorer or complete deployable side-asset bundle.
No `.autolab/model.json` is declared: scoring the main model alone would not
evaluate this change. The ignored outcome records the infrastructure block,
without claiming a kill or a measured zero gain.

## Resume and risks

Restore supervisor execution, audit help-train speaker/exclusion metadata,
extract observations and freeze the speaker-validated fit before inspecting the
actual confirming windows of the four targets. Verify a complete bundle under
`/scratch/listener-am-fm-coupled-evidence/` containing the original primary ONNX
and manifest, existing installed teacher assets, and the new scorer/descriptor.
Only then declare that bundle through untracked `.autolab/model.json`.

Run the required Modal suites and full-session preservation checks. If fewer
than three target warnings are removed, revert the draft engine changes and
record the measured premise kill. If at least three are removed, request practice
scoring before narrowing on an aggregate gate failure.

Small training coverage and uniform boundary initialization may yield poor phone
assignments. The simple complete-phone chain has no separate background states
for extra utterance-edge silence. Fixed complex bands have differing group delays;
there is no independently established physiological interpretation of the fitted
covariance. A missingness/background approximation and local reliability estimate
may misjudge noisy weak sounds. Independent bands omit cross-band dependence,
and overlapping windows still contain temporal correlation in both arms.
Speaker-held-out predictive likelihood need not select features that distinguish
lexical rivals. Session observation/audio buffering and duplicate frontend work
need CPU/memory measurements; Node asset loading is implemented, web/native asset
installation is not. Positive joint support for the written pronunciation does
not establish preservation of a weak genuine substitution.

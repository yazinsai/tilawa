# Noise-power observation law

Status: implementation draft; blocked before training and target validation by the supervisor's Modal infrastructure. This is not a killed experiment or a scoring-ready model bundle. The experiment card is unchanged.

## Prediction and premise

Predict listener −1 and overall −1: one false wrong-word warning removed, with no predicted loss. The hypothesis is that additive complex noise becomes a biased, signal-dependent observation after power and logarithmic compression. A zero-mean error model in logged power can therefore prefer a rival even when the actual noisy-power likelihood prefers the written pronunciation.

The card's reversal premise names four targets. The task's continuation rule requires at least three actual target recoveries before practice scoring, overriding the card's weaker two-reversal prerequisite. Neither criterion has been measured. No target waveform or supplied listener evidence was accessed; the infrastructure failure does not establish absent noise, an incorrect premise, or zero wins.

## Implementation

The SDK observes individual positive-frequency FFT-bin powers through a read-only callback on its existing frontend. Windowing, DC removal, pre-emphasis, FFT geometry, mel features and primary inference remain unchanged. Sixteen fixed bins exclude DC and Nyquist, whose noise would not follow the complex circular model. Observations retain the native 25 ms window and 10 ms shift; warning endpoints remain on the existing 40 ms evidence clock.

For measured noise power `n = E|N|²` and latent clean power `s = |S|²`, the observation density is

`p(y | s, n) = exp(-(y+s)/n) I0(2 sqrt(y s)/n) / n`.

Equivalently, `2y/n` follows a noncentral chi-square distribution with two degrees of freedom and noncentrality `2s/n`. Evaluation uses the scaled Bessel function and squared difference of square roots to avoid high-SNR cancellation. Each phone predicts a Gaussian distribution of log clean power. Gauss-Hermite quadrature integrates its likelihood, rather than subtracting an estimated spectrum or averaging log likelihoods.

The matched control uses the same clean-power distribution and its quadrature. Conditional on `s`, it models `log(y)` as an unbiased additive Gaussian error around `log(s)`, with delta-method variance `2n/s`. Its density includes the same conversion to power units. The two arms share phone parameters, audio, noise estimates, context pronunciations, mandatory phone states, optional noise-only states and normalized timing priors. Every articulation in both complete pronunciations must occur; repeated labels remain distinct mandatory states. Dynamic programming sums all monotonic phone boundaries. Shared noise-only states can explain pauses without imposing a speech phone on them.

Noise estimation uses completed preceding gaps between primary tokens satisfying the engine's existing pause-length criterion. Full analysis windows near their bordering speech are excluded. All qualifying preceding pause observations contribute to one mean spectrum. No observation at or after the complete comparison's start enters this estimate, and the same frozen spectrum serves both hypotheses. No qualifying pause, missing training coverage, nonpositive powers, incomplete observations or unsupported phones cause abstention.

A certificate requires the log-power control to strictly prefer the accusation and the explicit noisy-power model to strictly prefer the complete written pronunciation. Both signs must survive increasing quadrature from 17 to 33 points. This sign check detects some integration instability; it is not an error bound.

The core exposes comparisons only when a non-graph wrong word would otherwise confirm, using that actual decision's complete written/rival strings and exact acoustic endpoints, including its word-local and paired forms. Certificates match the location, both pronunciations and both endpoints exactly. A veto changes the public verdict to uncertain while retaining the baseline internal confirmation/ownership ledger. This preserves insertion-tail merges and subsequent candidate generation. Graph confirmations and other warning classes do not use the veto.

Audio-backed sessions perform the check before warning publication. Recorded evidence receives certificates through primary-token metadata after the existing gap and teacher-insertion enrichment. Primary token values, posterior rows and tracker events remain the evidence source. Optional Node model-directory loading verifies the side descriptor, fixed filename, byte count, SHA-256, inventory, frontend clock, dimensions and parameter validity. A declared missing/corrupt scorer fails and releases the primary backend. Bundles without this optional descriptor retain their ordinary behavior; running one of those bundles does not evaluate this experiment.

## Training utility and unavailable bundle

`modal/student_noise_power.py` requires authorized original Quran training recordings with verified complete transcripts. It also accepts none-labelled help-train training takes. Practice sources, non-training splits, scripted mistake recordings and speakers in the required exclusion list are rejected. The source/transcript/authorization flags are provenance assertions that must be established from the authorized dataset, not invented by a preparer.

The utility reproduces the SDK FFT features, measures noise from completed recording pauses and fits latent clean log-power moments through forward/backward boundary EM. Optional noise states keep pause frames out of phone fitting. It uses eight EM passes, a fixed intrinsic variance floor, and pseudo-observation strengths 8, 32 and 128. Hash-ordered speaker groups form three disjoint validation folds; mean held-speaker noisy-power likelihood selects shrinkage, with stronger shrinkage breaking ties. All choices were specified without accessing targets or a practice score.

The intended outputs are `noise-power-emissions.json`, `noise-power-asset.json` and a training report containing input hash, exclusions, folds and selection results. They must be fitted and verified in a persistent scratch bundle alongside unmodified primary and existing teacher files before declaring `.autolab/model.json` and evaluating the candidate. No fitted asset or complete export bundle exists. No `.autolab/model.json` was created, and this draft must not be interpreted as a validated deployment of the scorer.

## Validation and execution block

Three authorized supervisor requests failed before container execution with `NotFoundError: No such file or directory`:

1. Run 1 requested a read-only data/model/training layout and scratch-capacity audit.
2. Run 2 requested the supplied six-shard `bash tools/autolab_tests.sh` harness.
3. Run 3 requested that harness again after the implementation and initial focused checks.

No alternative Modal path, outside-worktree data search, release-held-out access or practice score was attempted. All three practice-score allowances remain. Training-source availability, target eligibility, preference reversals, population losses, clean-session behavior and studio behavior are unknown.

Local validation uses synthetic numerical inputs and ordinary repository fixtures only. Four Python checks passed: the density agrees with SciPy's noncentral chi-square distribution and integrates to one; forward/backward scores agree with exhaustive small-path enumeration; a tiny synthetic EM fit remains finite; unauthorized sources and excluded speakers are rejected before audio access. No real recording was fitted on the Mac.

The complete SDK Vitest suite passed locally: 38 tests passed, one optional integration test skipped. This includes ten new acoustic tests and three new verified/missing/corrupt side-asset loading checks. Five focused core checks passed, including exact-window certificate transport, strict reversal requirements, unchanged issue creation and preservation of an actual parallel-graph confirmation under a rejecting string-path callback.

The full local core run passed 753 tests and failed eight on test deadlines, with two additional Vitest worker RPC timeouts. A serial retry of the four affected files with a 60-second default test deadline passed 58 tests; one startup-persistence test still exceeded its explicitly configured 15-second deadline, and the retry reported one worker RPC timeout. Running that remaining test alone also exceeded its 15-second deadline (21.36 seconds). No assertion failure was reported, but the required complete core suite has not passed. Existing test deadlines were not changed. Logs remain in the ignored `.autolab/core-tests.log`, `.autolab/core-retry.log` and `.autolab/core-startup-isolated.log`.

Core and SDK TypeScript compilation passed with the repository's TypeScript 5.9.3, installed only in the ignored local test environment. The unavailable optional React Native runtime used an ambient module declaration in that environment; its actual runtime was not tested. The Node ONNX fixture tests use the real installed runtime, while side-asset loader tests mock inference. Python syntax checking and `git diff --check` passed. Local tests do not replace the required Modal checks or real model-bundle validation.

## Limits and next required work

The likelihood assumes approximately circular additive complex Gaussian noise with a spectrum stationary across the disputed interval. Real microphone processing, nonstationary breaths, correlated FFT bins and overlapping windows can violate that model. Token gaps can contain missed quiet speech and contaminate the noise estimate. Sixteen bin observations omit much of the spectrum; pooled phone emissions can miss channel and speaker effects. Quadrature and boundary EM provide approximations, and empirical shrinkage does not integrate uncertainty in fitted parameters. None of these limits has been calibrated on real authorized audio here.

The session retains audio and spectral observations for enrichment; device memory, streaming latency and compute cost are unmeasured. Node auxiliary installation is implemented; web/native auxiliary installation is not. Strict positive written-pronunciation support limits the intervention but cannot guarantee preservation of weak real substitutions.

After infrastructure recovery: audit original training sources and exclusions; fit/select solely on training-speaker partitions; export and verify the complete persistent bundle; exercise the actual SDK/evidence path on the four registered targets. Apply the task's premise stopping rule to those measured results. If at least three targets are recovered, run the authorized practice score before considering a narrower change. Until then the experiment is blocked, not killed, and no aggregate gain is claimed.

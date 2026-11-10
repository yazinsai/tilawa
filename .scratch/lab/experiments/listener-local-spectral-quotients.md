# Local consonant / vowel spectral quotients

Status: needs-human — implementation draft, blocked before Modal execution.
The experiment card is unchanged. There is no fitted real-data asset bundle;
this branch is not ready for population scoring.

## Prediction and premise

Predict listener −1 and overall −1: one false substitution removed, with no
predicted losses. A stationary linear recording channel approximately adds the
same frequency response to consonant and adjacent-vowel log spectra. Their
difference can retain articulation cues while cancelling that coloration.

The card kills the experiment if none of its four registered windows reverses
the absolute-spectrum control toward the complete written pronunciation. The
task additionally requires at least three actual target warnings removed before
continuing to practice scoring. Neither condition has been evaluated. Failed
infrastructure is not a measured zero gain or a premise kill.

## Implementation

`packages/sdk/src/spectral-quotients.ts` reads 32 exact FFT-bin log powers from
the native frontend, at frequencies 125–6906.25 Hz. It retains the same 25 ms
Povey window, 10 ms shift, DC removal and pre-emphasis. Four successive analyses
are averaged on the existing 40 ms evidence clock. The optional spectral
observer is separate from the frontend's original mel output; the primary
listener consumes precisely its original features.

The complete pronunciation is covered by disjoint consonant-plus-following-vowel
units and residual single phones. Repeated labels are retained. A CV unit has
two mandatory latent intervals; both its external boundaries and internal cut
are summed over by a bounded semi-Markov dynamic program. Each interval permits
1–32 observations. No existing CTC/edit backtrace supplies those intervals.
Inference requires every pronunciation phone, exact original support-window
endpoints and observed training coverage for every constituent phone; unavailable
parameters, incomplete placements, non-finite inputs or windows exceeding 1,024
observations abstain.

The quotient emission is the interval-mean consonant log spectrum minus the
interval-mean vowel log spectrum. The matched absolute control observes the
interval-mean consonant log spectrum, conditioned on the same CV identity.
Both have 32-dimensional diagonal Gaussian emissions, identical unit inventories,
boundary search, normalized internal-cut and duration priors, and completion
normalization. Residual single-phone emissions are shared with the absolute
arm. This control isolates the effect of the adjacent-vowel subtraction rather
than adding a second vowel emission or changing model size.

`modal/student_quotients.py` fits the two emission families by hard EM: uniform
complete-transcript initialization, followed by latent MAP unit boundaries and
CV cuts. Inference and validation integrate boundaries, rather than using the
MAP path. Sparse estimates shrink toward phone-pair pools shared by consonant
manner, consonant identity and adjacent vowel; fixed variance floors prevent
degenerate estimates. Three hash-ordered speaker-disjoint training folds select
shrinkage from 8, 32 and 128 by mean held-speaker predictive score of both arms,
with ties favoring stronger regularization. Final fitting uses all eligible
training speakers, with five fixed EM iterations. These choices were made
before accessing any real recording or target result.

`modal/student_quotients_extract.ts` exports the SDK observations from single-ayah
none-labelled help-train training takes only. It requires an explicit exclusion
list, unions it with non-training speakers in the labels, rejects contradictory
none labels and paths outside the help directory, and supplies canonical phones
with the final word's supplied pausal pronunciation. The Python trainer checks
training/none/non-excluded provenance and the exclusion list again. Scripted
mistakes are not treated as performed transcripts. Actual label layout, split
and exclusion provenance still need auditing in the authorized container.

The trainer writes `quotient-emissions.json`, `quotient-asset.json` and a training
report recording input hash, exclusions, folds, selection scores and fit history.
The optional Node bundle loader verifies the descriptor, filename, byte count,
SHA-256, inventory, clocks and parameter shapes. A declared missing or corrupt
asset fails loading and releases the backend; ordinary primary-only bundles
continue to load normally. Supplied manifests and registry data are unchanged.

The checker exposes its actual complete written and already-accused comparison
at a confirming string-path warning, including ordinary, joint and word-local
support paths. Audio sessions compare those hypotheses before reporting a
confirmation. Recorded evidence is enriched after the existing gap and teacher
insertion stages. A serialized certificate requires finite scores with a strict
absolute preference for the rival and strict quotient preference for the written
pronunciation. It matches the word, both complete strings and both endpoints
exactly; ties abstain.

Only the public wrong-word verdict becomes uncertain and its mistake-list entry
is suppressed. The private confirmation and ownership ledger retain their
baseline state, preserving inputs to later insertion merges and candidate
generation. Primary tokens, tracker input and graph confirmations are unchanged.
No new warning is created. The callback and certificate are ordinary listener
evidence, with no session, reader or verse selectors in engine code.

## Validation and execution block

The authorized supervisor route failed before container execution:

1. Run 1 requested a read-only data/help-train layout, numerical-library and
   scratch-capacity audit. Result: `NotFoundError: No such file or directory`.
2. Run 2 requested the supplied six-shard build/test harness. Same error.
3. Run 3 retried that harness with the draft source/tests. Same error.
4. Run 4 retried the read-only training-data audit. Same error.

All four result files report zero compute cost and contain no container
logs. None establishes a test failure, data absence or storage exhaustion.

Twelve focused core/SDK cases passed locally with Node's TypeScript transform on
ignored workspace copies and a minimal assertion harness. They cover strict
control reversal, arbitrary stationary frequency-response cancellation,
exhaustive unit/cut summation, complete repeated-phone constraints, coverage and
input abstention, identical primary fbank output, audio rechunking, parameter
validation, complete candidate discovery, exact serialized certificate matching,
unchanged issue creation, preservation of a confirmed addition and graph
preservation. These are small unit checks, not the full Vitest suites.

Three Python numerical tests passed using an ignored workspace NumPy environment:
the latent boundary sum matches exhaustive enumeration, training and exclusion
provenance is enforced, and a small synthetic hard-EM fit has finite positive
parameters. No real recording was fitted on the Mac. A separate mocked-runtime
check using real temporary files passed optional, verified, corrupt and missing
asset loading, checked backend release and preserved the primary manifest.
The session module imports under the same TypeScript transformation. A full
synthetic three-fold training/export run produces a model accepted by SDK
validation; four Python/SDK sequence scores match within 1.07e−14. This toy
asset remains ignored and is not a deployable pronunciation model.

A strict local TypeScript check passed for the changed SDK/core source and the
training extractor, using workspace source mappings and declarations for the
unavailable ONNX runtime modules. Python syntax compilation and `git diff
--check` also passed. Required full builds, complete core/SDK Vitest suites,
actual ONNX execution and full-stream population preservation remain unverified.

No target recordings were accessed; target wins and preference reversals are
unknown. Zero practice scores were requested, leaving all three allowances.
There is no `.autolab/model.json`: declaring a primary-only bundle would not
evaluate this change, and declaring an unfinished side bundle would violate the
handoff requirement. The ignored outcome records the infrastructure block
without claiming a kill or numerical gain. No scorer, lab tooling, supplied
manifest, test data, evidence or card was edited; no fold was performed.

## Resume and limits

Restore the supervisor's container execution, audit the supplied label and
exclusion metadata, and export eligible none-labelled observations. Fit and
freeze the model before inspecting the four actual engine-requested target
windows. Copy the original primary ONNX and manifest plus any installed teacher
files intact into a persistent experiment bundle, add the verified quotient
descriptor/emissions, and verify the complete bundle through the Node loader
before declaring it in untracked `.autolab/model.json`.

Run the required suites and compare complete streams for primary evidence,
tracker events, graph confirmations and insertion decisions. If fewer than
three actual target warnings are removed, revert the draft engine changes and
record a measured killed outcome. If at least three are removed, request
practice scoring before narrowing on any aggregate gate failure.

The cancellation is exact for an additive log-frequency response applied to the
observations, but only approximate for windowed real audio, additive noise,
time-varying gain and handset DSP. Unusual vowels can change the quotient and
hide a substitution. Disjoint CV grouping leaves codas and unpaired geminate or
duration labels with absolute evidence; it is not an exhaustive physiological
syllabifier. Pair-conditioned segment features are a pronunciation score, not
a calibrated waveform likelihood. Hard EM can converge to poor boundaries;
the model lacks an explicit edge-silence state and uses diagonal covariances.
Sparse pair sharing can erase articulation distinctions. The fixed 40 ms
observations and bounded durations may abstain on fast or very long renditions.

The draft buffers session observations/audio, duplicates frontend work and
replays candidate discovery for evidence enrichment. Device memory and CPU
budgets have not been measured. Node bundle loading is implemented; web/native
asset installation is not. Positive quotient preference is not proof that a
genuine weak substitution will survive.

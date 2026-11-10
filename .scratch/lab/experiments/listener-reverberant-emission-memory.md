# Reverberant phone-emission memory

Status: needs-human — implementation draft with focused numerical and transport
checks; Modal execution, training and target validation are blocked. This branch
has no fitted side-scorer bundle and is not ready for practice scoring. The
experiment card is unchanged.

## Prediction and stopping rule

Predict listener −1 and overall −1: one false wrong-word warning removed, with
no losses predicted. The proposed cause is lingering spectral energy from
earlier phones being interpreted as the current consonant's own spectrum.

The card's premise kill is no written-versus-rival reversals on the four complete
target comparisons. The task also requires at least three target warnings won
before continuing to practice scoring. Neither rule has been evaluated. A
container-launch error provides no evidence of a premise kill or zero gain.

## Mechanism

The SDK extracts a separate spectral view with the original Fbank, preserving
its Povey window, DC removal, pre-emphasis, mel bank and sample clock. It sums
linear power over four adjacent mel bins and four 10 ms observations, then
normalizes spectral mass and takes logs. This gives twenty spectral coordinates
on the primary evidence's 40 ms clock. Microphone push size does not change
observations. No audio samples, primary features or decoded phones are replaced.

Each phone has a learned positive, normalized spectrum. Its observation mean is
a mixture of its own spectrum and the spectra of the preceding four phone
states. The current coefficient is `1 − mix`; the four tail coefficients are
proportional to `mix × decay^lag` and normalized to sum to `mix`. The kernel
therefore has unit mass. Its four-frame, 160 ms support is fixed before any
recording analysis; both mixing fraction and decay are learned on training
audio. Diagonal Gaussian errors model the logged normalized spectrum.

The likelihood uses an exact augmented-state forward sum. A state carries the
pronunciation position and the most recent four stay/advance decisions. Those
decisions identify the phones whose spectra still contribute to the current
observation. During the first four frames a latent preceding-phone spectrum is
integrated under a normalized training-derived occupancy prior shared by both
hypotheses. After frame four that incoming state is summed out exactly. All
reachable recent-state histories remain; no beam or selected boundary path is
used in inference.

Every phone of the complete pronunciation must be visited, including separately
represented repeated labels. Per-phone geometric duration priors are the same
for both hypotheses; subtracting the emission-free partition conditions each
score on completion within its fixed observation window. The matched
instantaneous control uses exactly the same fitted spectra, variances and
duration priors, disabling only the cross-frame convolution. A certificate
requires a strict instantaneous preference for the rival and a strict memory
preference for the written pronunciation. Ties and unsupported phones abstain.

The checker exposes its actual confirming complete non-graph comparison,
including its written and already-accused pronunciation and its original frame
endpoints. Ordinary, word-local and paired support windows are covered without
moving their boundaries. Certificates must match the word, both complete
strings and both endpoints and contain four finite scores with that reversal.
Graph confirmations are exempt. The internal confirmation/ownership ledger
retains its baseline decision, preserving insertion merges and later candidate
generation; only the public wrong-word verdict becomes uncertain and that
occurrence leaves the returned mistake list.

Audio-backed sessions score before publishing confirmation. Recorded sessions
also add certificates after the existing gap and teacher-insertion enrichment,
using normal token metadata serialization. The Node directory and tier loaders
verify an optional `reverberant-asset.json` descriptor, the fixed
`reverberant-emissions.json` filename, byte length, SHA-256, inventory, clocks,
normalized spectra and parameter dimensions. Declared missing or corrupt assets
raise an error and release the backend. Primary-only bundles load normally.

## Training and reproducibility

`modal/student_reverberant.py` implements CPU training. Generalized EM uses exact
forward/backward marginals for phone boundaries and recent/incoming states.
The emission M-step optimizes normalized phone spectra and the normalized decay
kernel with analytic gradients. Covariance and duration updates use posterior
counts. Training drops posterior weights below 1e−9 for numerical economy;
inference retains every reachable state. Initialization uses uniform transcript
placement, without claiming performed frame boundaries.

Sparse phone spectra share information through fixed Arabic manner/place
attributes. Shrinkage candidates 8, 32 and 128 are chosen by three hash-ordered,
speaker-disjoint training folds using per-speaker mean marginal likelihood;
ties select stronger shrinkage. A fixed mild Beta prior avoids a degenerate
all-memory fit. The final model is fitted on all eligible training speakers.
The report records input SHA-256, speaker folds, exclusions, every validation
score, optimizer diagnostics and the exported asset hash.

Training input requires explicit training/exclusion provenance. help-train
accepts only none-labelled takes; authorized original Quran recordings require
an explicitly verified transcript. Scripted mistake and repeat recordings are
not performed-transcript supervision. A required exclusion list is checked
before fitting. The generic SDK preparation CLI additionally excludes every
speaker with any non-training take in the supplied labels, before opening audio.
The supplied split and release exclusions still need an audit against the actual
data metadata: no container reached that data in this turn.

The generic CLI can also inspect all actual confirming comparisons for a supplied
recording and evidence file. It checks that enrichment preserves primary tokens,
posteriors, tracker events and every other error class. It has no incident list,
target-specific vocabulary or scoring logic. No scorer, lab tool, test data,
supplied manifest, evidence or experiment-card edits were made.

## Validation and limits

Eighteen focused SDK/core Vitest cases passed locally in an ignored isolated
dependency environment. They cover a synthetic cross-phone reversal, retention
of an actually different spectrum, exhaustive boundary/incoming-state sums past
the memory horizon, exact zero-memory control equivalence, repeated-phone
completion, unsupported-input abstention, native power extraction and
rechunking, malformed assets, verified/missing/corrupt bundle loading, exact
certificate identity, unchanged issue creation, graph preservation, and live
verification before publication with recording disabled. ONNX is mocked for
the loader tests; these do not establish inference with a deployed bundle.

Four Python numerical tests passed on small synthetic arrays: forward/backward
values and phone marginals match exhaustive paths, normalized emission/decay
gradients match finite differences, training provenance rejects ineligible
records, and a tiny EM update produces finite normalized spectra and positive
variances. Eight additional complete-sequence Python/SDK likelihood comparisons
agreed to ten decimal places. No recording was trained on the Mac.

Both core and SDK TypeScript builds passed locally. Full Modal build/test results
and all existing-suite results remain unavailable. The first local SDK build
attempt lacked the optional React Native peer package; installing that dependency
only in the ignored test environment allowed the unmodified build to pass.

Three authorized supervisor requests were made: a read-only data layout check,
the supplied six-shard build/test harness, and the six-shard harness plus the
Python tests after implementation. All three failed before container execution
with `NotFoundError: No such file or directory`. No training or target audio has
been accessed, no fitted model exists, and no practice score has been requested.
All three practice-score allowances remain.

These checks establish numerical and transport behavior, not acoustic accuracy,
target wins, population losses, clean-session behavior or studio behavior.
The compact stationary spectra, finite memory horizon and diagonal covariance
could fail to explain real reverberation or could explain away a genuine swap.
Normalized spectral shape discards absolute energy. Phone coverage, long
silences and initialization quality require training-data inspection. Device
latency and memory have not been measured; Node bundle installation is
implemented, while web/native side-asset installation is not.

## Required next steps

Restore the supervisor execution path and audit the supplied training split and
exclusion metadata. Prepare observations with the SDK CLI, train and freeze the
model before looking at target comparisons, then inspect the four actual
engine-requested comparisons and their corresponding warning verdicts. If
fewer than three targets are won, remove the engine changes and record a premise
kill. Otherwise finish a persistent bundle with the supplied primary listener,
manifest and existing teacher assets preserved, verify every file and the normal
Node loading path, and declare only that verified bundle through untracked
`.autolab/model.json`. Run the required suites and the authorized practice score
before judging losses. No bundle is currently declared: decoding the main
listener alone would not evaluate this experiment. No fold was performed.

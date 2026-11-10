# Nonlocal acoustic denoising

Status: implemented draft; blocked on supervisor Modal infrastructure. Training
calibration and target premise are unmeasured. The default engine path remains
disabled until a training-only geometry is frozen. This is not a killed premise
and must not be scored as a completed experiment.

The target keys supplied with the task are:

- `lab-v1.5-p11/phone/warning:wrong_word/26:106:1/104.32`
- `lab-v1.5-p36/phone/warning:wrong_word/55:48:1/282.64`
- `lab-v1.5-p55/phone/warning:wrong_word/26:174:8/107.08`
- `realuser-v1-sob-049/phone/warning:wrong_word/92:13:1/36.44`

## Prediction

The card predicts listener −1 and overall −1, removing one false wrong-word
warning with no predicted losses. Independent noise might obscure consonant
structure that recurs in short patches elsewhere in the same recording.

The card's premise requires at least two targets with nonoverlapping donors that
restore a written distinction absent in the identical raw replay. The task's
continuation rule additionally requires at least three target warnings removed
by the complete implementation. Neither threshold has been measured. There have
been zero practice scores; all three remain available. The card is unchanged.

## Implementation

`packages/sdk/src/nonlocal-acoustic.ts` builds a recording-local bank of 25 ms
spectra at a 10 ms stride. A query is matched using four surrounding spectral
trajectories over 16 bands. The inner flanks are 30 ms away, sharing no samples
with its central observation. The central spectrum does not rank candidates.
Context energy supplies a shared gain normalization. Donors must end before
the entire disputed word, before the query patch, and be mutually disjoint over
their complete context supports.

A bounded shortlist supplies at least three donors. Every donor pair must have
consistent normalized central spectra. Otherwise that query is unchanged. The
central power estimate averages the original observation with donor powers,
weighted exponentially by context distance. Original complex phase is retained.
Delta overlap-add keeps all samples outside the disputed interval exactly fixed;
abstention is an exact PCM passthrough. There is no word-template bank, text or
phone identity in retrieval or reconstruction.

The core exports a read-only callback for actual string-path wrong-word
evaluation windows. The SDK discovers those requests using the existing tracker
and checker, then decodes paired raw and reconstructed warm crops with the
shipped acoustic model. Their origin, duration, streaming state initialization,
chunk grid and padding agree. A certificate requires the reconstructed slot to
spell the complete written pronunciation under existing soft substitutions,
without spelling the rival, and its CTC likelihood must also prefer the written
sequence over that rival. Both likelihoods must be finite. An otherwise identical
raw crop that already spells the written word disqualifies the certificate.

Certificates are tied to written and rival strings, exact frame bounds and word
occurrence, and carried on primary tokens through ordinary JSON evidence.
The core checks complete pronunciation identity and acoustic preference again.
Only the string-path wrong-word support decision can be vetoed. Primary tokens
and posteriors, tracker inputs and events, graph confirmations and insertion
decisions are preserved. Audio-backed sessions can withdraw a certified primary
wrong word at completion; recorded evidence carries the same certificate for
ordinary downstream replay.

## Training-only freeze protocol

`packages/sdk/src/bench/nonlocal-calibrate.ts` reads only explicitly `train` or
`training` split, `none`-labelled help-train recordings. It refuses missing
training partitions instead of falling back to other splits. It never reads
their Quran transcripts or recognition outputs. The supplied help-train input
is required to exclude release-held-out speakers, as stipulated by the task.

The preregistered search uses radii 5/8 analysis frames, context and donor
consistency limits 0.25/0.5/1, and three donors. These are training candidates,
not a profile fitted to practice. Three speaker-grouped folds select geometry
on the other speakers and report held-speaker reconstruction error. Each speaker
contributes equally. Paired independent noise injections at 0.1 times recording
RMS provide a waveform reconstruction loss relative to the original recording.
Noisy and reconstructed observations share their exact samples and durations.
The training-only final selection must be recorded before any target inspection,
then copied into `NONLOCAL_GEOMETRY` in the engine source. It is currently `null`.

After calibration, `nonlocal-recheck-cli.ts` accepts any WAV, primary evidence,
model directory and completed calibration JSON. It discovers the actual core
requests and reports donor intervals, raw/reconstructed readings, complete-word
identity, likelihoods and reconstruction-specific recovery. An optional generic
word argument selects a diagnostic location; no target IDs or verse references
are embedded in engine or utility code.

## Validation

Four supervisor requests all failed before execution with the same output:

```
autolab: modal run N: NotFoundError: No such file or directory
```

Runs 1 and 2 requested read-only input inspection, run 3 requested the supplied
six-shard `bash tools/autolab_tests.sh` harness, and run 4 requested the training
calibration. No remote logs, calibration artifact, target audio observations or
target decoder results were returned. These failures establish an infrastructure
blocker, not absence of eligible donors or zero target recovery.

Local validation used dependencies isolated under ignored `.autolab/check-tools`.
The complete SDK suite passed: 32 tests, one optional integration test skipped.
The final focused SDK runs passed eight acoustic tests and two replay tests,
including an independent central-contamination reduction test and the two
replay tests added after that complete suite. The final focused core run passed
five certificate tests. They cover center-blind retrieval, disjoint causal
donors, donor inconsistency, abstention, phase/timing and outside-span preservation,
native-clock written/rival likelihoods, pronunciation completeness, genuine
doubling, JSON transport, stale certificates and a still-supported rival.
After the calibration guard and session changes, the two affected session test
files also passed again (six tests).

The core TypeScript build passed. SDK compilation reported only the two optional
platform peers missing from this local toolchain (`onnxruntime-web` and
`onnxruntime-react-native`); a separate check covers SDK source excluding those
two platform entry points. The broad core run overlapped the last source edits
and finished with 759 passing tests, one rival-support guard failure and one
existing startup test exceeding its explicit 15-second timeout, plus two worker
RPC timeout errors. A fresh run of all five certificate tests passes with the
final finite-likelihood guard. An isolated startup-file rerun passes two tests
and again exceeds the third test's explicit 15-second timeout (about 18 seconds).
No assertion failure was reported for that startup test. Its timeout remains
unresolved; a completely passing full core run has not been established here.
No scorer, lab tooling, supplied manifests, reference assets, labels or
evidence were edited, and no model bundle or `.autolab/model.json` is declared.

## Limits and remaining work

The injected-noise calibration does not prove that the original recordings'
background noise is independent, nor that spectrally similar flanks identify
the same consonant. Mutually consistent donors can still reinforce a different
consonant. Original phase may retain corrupted information, and a positive
written reading could still erase a genuine substitution. Those risks require
target diagnostics and the ordinary fold gate, not assertions from unit tests.

This draft uses the existing completion-time auxiliary evidence route and keeps
recording audio privately when enabled. It does not yet meet immediate streaming
warning or bounded recording-memory requirements. The raw replay control guards
against mistaking crop-induced recovery for a reconstruction benefit, but it
does not prove full-stream state equivalence.

Once supervisor execution is restored: run training calibration, freeze its
profile, inspect all four registered targets on actual checker requests, and
test the complete certificate route. If the premise or fewer-than-three-wins
rule fires, revert engine changes and record a killed outcome. If at least
three targets are won, run the authorized practice score before deciding.
Until then the hypothesis remains open, and no score or target gain is claimed.

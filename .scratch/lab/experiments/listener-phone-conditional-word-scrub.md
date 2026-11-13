# listener-phone-conditional-word-scrub

Prediction: listener −1 and overall −1, with no forecast losses. The proposed
failure is lexical context in the acoustic representation that supplies a word
shortcut beyond the information required to recognize its phones. Status:
implementation checkpoint; supervised execution is blocked, and the acoustic
premise and target gains have not been measured.

## Implementation

A separate copy of the inherited streaming encoder exposes the float features
immediately before the CTC readout. Export checks all original outputs and caches
bit for bit over eight successive streaming windows. A deterministic 32-dimensional
projection bounds feature caching; the inherited primary graph stays unchanged.

Preparation accepts authorized help-train training speakers' none-labelled takes.
Any excluded or non-training row excludes its entire speaker group. Missing split
provenance is rejected, waveform paths must remain inside the authorized training
root, and repeats, unsupported labels and scripted events cannot become canonical
phone supervision. Complete mushaf word pronunciations supply lexical labels;
homophonous forms share a label. Adjacent held phones collapse as in the shipped
training recipe, and ambiguous word-boundary labels abstain from the adversary.
Exact complete-transcript CTC forward/backward probabilities give soft phone
occupancies, including blanks, rather than engine word boundaries.

Each arm learns the same 32-unit nonlinear residual projection and phone readout,
with complete CTC loss and soft occupancy supervision. The conditional word
adversary receives pooled projected acoustic features and an explicit one-hot
transcript phone identity. Only phones occurring in multiple words qualify, and
each phone-word pair must be attested by at least two fitting speakers. Outcomes
are balanced within phone. The adversarial arm reverses the lexical gradient
while the phone loss preserves recognition. The ordinary arm has identical
capacity and phone supervision without that reversed gradient; the shuffled arm
uses word labels permuted within phone and speaker, retaining their conditional
marginals while destroying the acoustic pairing. Initialization, data order,
regularization, learning rates and update counts are shared.

A deterministic speaker-grouped training partition selects a common checkpoint
from 80 epochs using the mean phone loss across arms plus a fixed weak substitution
retention penalty. The controls are already-caught string-path substitutions at
help-train scripted labels on validation speakers. They retain the primary
checker's exact hypotheses/windows; their recordings never receive an assumed
performed transcript or canonical phone training loss. Missing validation-speaker
controls prevent fitting. Complete-transcript consonant counterfactual comparisons
provide an additional training-only diagnostic. No practice example participates
in training, control construction or checkpoint selection.

Fresh, independently initialized conditional lexical probes are fitted after
checkpoint selection with each projection frozen. They predict true word labels
on held-out training speakers. A phone-only probe measures lexical predictability
available from identity alone. The report requires acoustic predictability in the
ordinary arm and reduced predictability in the adversarial arm relative to both
ordinary and shuffled controls. The probe result cannot change the checkpoint.
Export, extraction and fitting CPU time are bounded to one day.

The checker exposes both complete pronunciations and its existing native-frame
support window only for otherwise confirmed string-path wrong words. The SDK sums
complete CTC chains, marginalizing boundaries on the identical window for every
arm. A certificate requires a strictly positive written-versus-rival margin in
the adversarial readout and strictly negative margins in both controls. Nonfinite,
unknown-phone or incomplete-chain results abstain. The certificate binds the word,
both strings and frame bounds. It changes the public accusation while preserving
internal ownership and structural decisions; graph confirmations are exempt.

Audio sessions retain primary evidence and publish a matching uncertainty revision
at completion. Primary tokens, posterior values, tracker events and insertion
confirmation stay intact. Node loading checks every declared side asset and binds
the readout to the inherited primary hash, inventory and clock. Bundle assembly
uses a fresh directory and verifies all inherited files, including the insertion
teacher and external weights, byte for byte. It publishes a bundle path only after
exported-runtime numerical parity succeeds.

## Validation and limits

Four focused Python unit tests passed under the installed x86_64 Python runtime:
independent lexical attestation and speaker exclusions, within-phone-and-speaker
shuffling, exhaustive complete CTC path/occupancy sums, and inherited bundle
preservation with missing-asset rejection. The ordinary arm64 Python cannot import
the installed NumPy binary. Python compilation, shell syntax, changed TypeScript
syntax and whitespace checks passed.

Local assertions against stripped TypeScript sources passed for complete CTC
scoring, both-control agreement, exact-window certificate replay, baseline equality
when a control fails, graph-confirmation exemption, and SDK completion revisions
with tracker/insertion preservation. The session check uses mocked acoustic
certificates. These checks do not establish TypeScript type correctness or replace
the required SDK/core Vitest suites. Added unit tests also cover certificate
serialization and missing, corrupt or incompatible declared model assets.

The authorized supervisor inventory request and six-shard test request returned
`NotFoundError: No such file or directory` before execution, with zero compute
reported and no container logs. The complete 60-minute training/export/bundle/parity
request also returned the same error with zero compute and no bundle. The final
six-shard test retry returned the same error. All four requests failed before
execution; no full test results or fitted artifacts were produced. Untracked request/result files remain in `.autolab`.

No training data or target audio has been inspected, no readout has been fitted,
and no complete model bundle or `.autolab/model.json` is declared. No practice
score was attempted; all three allowances remain. Target wins, losses and
conditional predictability are unknown. Infrastructure failure does not establish
a premise kill, so no killed outcome is claimed. Scoring the inherited listener
would not evaluate this change. The experiment card, scorer, lab tooling, supplied
manifests and test evidence remain unchanged; no fold was performed.

## Remaining validation

After supervisor execution is restored, run `bash tools/autolab_tests.sh` with six
shards and `bash modal/student_word_scrub_run.sh` through the authorized runner.
Inspect the speaker/provenance and substitution-control coverage, fresh conditional
probes, inherited output parity, bundle checksums and runtime parity before copying
the returned bundle path into ignored `.autolab/model.json`.

Only then inspect actual checker-window comparisons for these four targets:

- `lab-v1.5-p11/phone/warning:wrong_word/26:106:1/104.32`
- `lab-v1.5-p36/phone/warning:wrong_word/55:48:1/282.64`
- `lab-v1.5-p55/phone/warning:wrong_word/26:174:8/107.08`
- `realuser-v1-sob-049/phone/warning:wrong_word/92:13:1/36.44`

If fewer than three targets are won, revert engine changes and record the measured
kill as instructed. If at least three are won, practice-score before deciding;
other incident losses belong to the fold gate. Training-speaker lexical coverage
and coarticulation preservation remain unmeasured. The fresh probes are finite
models, and their failure to predict does not prove information is absent. The
second encoder pass, whole-session audio retention and completion-time revisions
also need device cost and streaming-latency assessment. This checkpoint is not a
completed or score-ready experiment.

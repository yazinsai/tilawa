# listener-duration-matched-hard-negatives

Status: needs-info

## Prediction

The card predicts one fewer listener false warning and one fewer overall wrong
call (listener -1, overall -1). Duration and vowel shape may let an acoustic
pronunciation verifier distinguish easy negatives without learning the
consonants that separate two legitimate words. Real different-word recordings
matched on these nuisance properties should make those consonants necessary.

The four registered targets are the false wrong-word warnings at 26:106:1,
55:48:1, 26:174:8 and 92:13:1 in the card. These references appear only in this
write-up, not in training, matching or inference code. No target comparison or
win has been measured. The user's final instruction requires at least three
actual target wins to continue to practice scoring; the card's zero-reversal
criterion does not replace that instruction.

## Implementation

A separate copy of the shipped int8 streaming graph exposes its frozen pre-CTC
features. The export utility checks the primary file checksum and requires
bit-for-bit equality of every original output and cache over eight successive
windows with graph optimization disabled. A fixed 32-dimensional projection
limits the head's cached-feature budget; the primary graph and manifest are
copied unchanged when assembling a bundle.

The preparer accepts only help-train training speakers' none-labelled,
non-excluded takes without planted events or repeats. The supplied help-train
export excludes release-held-out speakers. The preparer additionally excludes
any speaker appearing on a non-training row before extracting features.
Canonical full-take CTC alignment supplies recorded word spans, with midpoint
boundaries in the surrounding blank intervals. Every positive contains all
phones of a complete recorded pronunciation. Mistake-labelled recordings and
practice recordings do not enter fitting or checkpoint selection.

Speaker identity is assigned to one of five partitions by a fixed hash: three
fit, one selects checkpoints, and one audits the frozen result. Feature
normalization uses fitting speakers only. Matching happens independently inside
each partition. It pairs different complete pronunciations within the same
speaker and capture category, differing consonant identities, identical vowel sequences after collapsing
held vowel runs, duration ratio at most 1.3, log-RMS difference at most one and
mean frozen-feature descriptor distance at most three. Same-take matches are
preferred. These rules, the 4,000-word partition cap and the 100-frame word cap
are fixed in code, not selected on targets.

Matched swaps are disjoint and reciprocal: pronunciation A has its own actual
positive recording and word B's actual negative recording, and pronunciation B
has the opposite pair. The random-negative control starts with precisely the
same retained recordings and uses fifty sweeps of symmetric random pair-switch
proposals. Its only constraints are different pronunciations and enough frames
to consume the complete candidate. Both arms retain exactly the same positive
multiset, balanced positive/negative examples, initial weights, architecture,
optimizer, batch size, epoch count and update budget. Each selects its checkpoint
by its own training-speaker selection BCE. The independent audit partition
cannot select a checkpoint or veto margin.

The small head combines frozen features and phone embeddings through a shared
nonlinear readout. A forward sum marginalizes all ordered, nonempty phone
segments: every phone consumes frames, every frame belongs to a phone, and
normalization removes the number-of-timings advantage. A second equal-prior
mixture includes the exact span and one-frame onset/ending boundary alternatives.
The JSON-exported heads and the SDK implement the same calculation; the pipeline
requires agreement on training-only parity fixtures before its bundle is usable.

The checker exposes only existing confirmed string-path wrong-word accusations.
A listener certificate carries the complete written and rival pronunciations,
word location, exact native-frame interval and both score margins. It can
reduce a matching public warning to uncertain only when the matched head strictly
prefers the written pronunciation and the random control strictly prefers the
rival. Ties, nonfinite scores, changed bounds and changed pronunciations preserve
the warning. Graph confirmations retain their existing path. Internal issue
ownership stays at its baseline state to preserve insertion merging.

The SDK preserves primary tokens, posteriors and tracker input. Audio sessions
retain an auxiliary audio copy and publish applicable uncertainty revisions at
completion. The Node loader validates all four declared auxiliary assets by
byte size, checksum, geometry and primary-model identity; missing or corrupt
control assets reject a declared bundle. The assembly script copies all existing
primary assets, including insertion assets, unchanged.

## Validation and limits

Local checks passed:

- Four pairing tests: actual audio swaps, duration/vowel/capture constraints,
  exclusion of identical pronunciations, deterministic speaker assignment and
  identical positive recording multisets in the random control.
- Python compilation, shell syntax, TypeScript syntax checks on eleven changed
  or new files, and `git diff --check`.

Core/SDK unit tests were added for forward-sum boundary marginalization,
complete-phone consumption, strict control preference, certificate serialization
and exact identity, graph preservation, end-of-session revisions, insertion
preservation and missing/corrupt bundle rejection. They have not run successfully;
syntax checks are not TypeScript typechecking or passing unit tests.

Four authorized supervisor submissions failed with
`NotFoundError: No such file or directory`, no container log and zero compute
charge:

1. Read-only LAB_DATA/help-train inventory.
2. Minimal environment probe (`pwd` and an environment-variable read).
3. The required six-shard `bash tools/autolab_tests.sh` harness.
4. `bash modal/student_duration_negatives_run.sh`, the complete preparation,
   fitting, assembly and export-parity pipeline.

This is an infrastructure blocker, not a measured acoustic kill. No real
training recording was inspected, no head was fitted, and no auxiliary bundle
was generated or declared in `.autolab/model.json`. The changes are an
unvalidated implementation checkpoint, not a score-ready experiment. Evaluating
the primary model alone would not evaluate these changes.

Unknowns include whether enough matched pairs exist in all speaker partitions,
whether the head discriminates consonants in unseen speakers, whether any of the
four actual complete comparisons reverses, and whether genuine substitutions
are lost. Completion-time revisions do not establish timely suppression of a
live warning, and the second encoder pass plus full audio retention are not a
device-budget result. No loss-free or release-ready claim is made.

After the supervisor is restored, rerun tests on the final code and run the
fixed training pipeline. Require its primary-output and exported-head parity
checks, inspect its training-only pair counts and audits, then compare the four
actual engine requests using the generic SDK diagnostic. If fewer than three
targets are won, remove engine changes and record the measured killed outcome.
If at least three are won, declare the verified scratch bundle and practice-score
before deciding. All three practice-score allowances remain. No fold was done,
and the experiment card, scorer, data, evidence and manifests were not edited.
